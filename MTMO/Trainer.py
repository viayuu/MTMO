import hashlib
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from objectives import MultiObjectiveEnv
from utils import get_env, get_model


class RolloutModel(nn.Module):
    """Keep parameter generation, encoder and rollout inside one DDP forward."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, data, env, preference):
        batch_size = data.size(0) if isinstance(data, torch.Tensor) else data[-1].size(0)
        env.load_problems(batch_size, problems=data, aug_factor=1)
        reset_state, _, _ = env.reset()
        self.model.assign(preference)
        self.model.pre_forward(reset_state)
        probabilities = []
        state, reward, done = env.pre_step()
        while not done:
            selected, prob = self.model(state)
            state, reward, done = env.step(selected)
            probabilities.append(prob)
            if len(probabilities) > 4 * env.problem_size + 8:
                raise RuntimeError('Rollout exceeded the vehicle-routing step bound')

        objectives = -reward
        if objectives.ndim != 3 or objectives.size(-1) != 2:
            raise RuntimeError('Expected the two objective costs with shape [batch,pomo,2]')
        if not torch.isfinite(objectives).all():
            raise FloatingPointError('Non-finite route objectives')
        scales = self.model.objective_scales
        scalar_cost = (objectives / scales * preference).max(dim=2).values
        actor_reward = -scales[0] * scalar_cost
        advantage = actor_reward - actor_reward.mean(dim=1, keepdim=True)
        log_prob = torch.stack(probabilities, dim=2).log().sum(dim=2)
        rl_loss = (-advantage.detach() * log_prob).mean()
        aux_loss = self.model.aux_loss
        if not isinstance(aux_loss, torch.Tensor):
            aux_loss = rl_loss.new_tensor(aux_loss)
        loss = rl_loss + aux_loss
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')

        best_index = scalar_cost.argmin(dim=1)
        best_objectives = objectives[torch.arange(batch_size, device=objectives.device), best_index]
        metrics = torch.stack((loss.detach(), rl_loss.detach(), aux_loss.detach(),
                               scalar_cost.min(dim=1).values.mean().detach(),
                               best_objectives[:, 0].mean(), best_objectives[:, 1].mean(),
                               actor_reward.mean().detach()))
        return loss, metrics


def rng_state(device):
    state = {'python': random.getstate(), 'numpy': np.random.get_state(),
             'torch': torch.get_rng_state()}
    if device.type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if device.type == 'cuda' and 'cuda' in state:
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)


def json_value(value):
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


class Trainer:
    metric_names = ('loss', 'rl_loss', 'aux_loss', 'normalized_cost',
                    'total_distance', 'max_route_distance', 'actor_reward')

    def __init__(self, args, env_params, model_params, optimizer_params, trainer_params):
        self.args = args
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.device = args.device
        self.rank = args.rank
        self.world_size = args.world_size
        self.log_path = Path(args.log_path)
        self.envs = get_env(args.problem)
        self.model = get_model(args.model_type)(**model_params).to(self.device)
        self.rollout_model = RolloutModel(self.model)
        self.optimizer = Optimizer(self.model.parameters(), **optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **optimizer_params['scheduler'])
        self.global_step = 0
        self.total_instances = 0
        self.start_epoch = 1
        self.start_episode = 0
        self.start_epoch_metrics = np.zeros(len(self.metric_names), dtype=np.float64)
        self.scale_data = json.loads(Path(args.scales).read_text())

        if args.checkpoint:
            self._restore(args.checkpoint)
        if self.world_size > 1:
            self.rollout_model = DistributedDataParallel(
                self.rollout_model,
                device_ids=[args.local_rank] if self.device.type == 'cuda' else None,
                find_unused_parameters=True, broadcast_buffers=False)
        if self.rank == 0:
            self.log_path.mkdir(parents=True, exist_ok=True)
            config = {'args': vars(args), 'env_params': env_params, 'model_params': model_params,
                      'optimizer_params': optimizer_params, 'trainer_params': trainer_params,
                      'scales': self.scale_data,
                      'distributed_note': 'Global batch is split equally; MoE auxiliary loss is computed per rank.'}
            config_name = 'resume_config_step-{}.json'.format(self.global_step) if args.checkpoint else 'config.json'
            (self.log_path / config_name).write_text(json.dumps(config, indent=2, default=json_value) + '\n')
            self._record_sources()
            print('>> Trainable parameters: {:,}'.format(sum(p.numel() for p in self.model.parameters())), flush=True)
        if self.world_size > 1:
            dist.barrier()

    def _record_sources(self):
        root = Path(__file__).resolve().parent
        files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(root.rglob('*.py')) if '__pycache__' not in p.parts}
        try:
            revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        except subprocess.CalledProcessError:
            revision = None
        manifest = {'git_commit': revision, 'source_sha256': files}
        name = 'resume_source_manifest_step-{}.json'.format(self.global_step) if self.args.checkpoint else 'source_manifest.json'
        (self.log_path / name).write_text(json.dumps(manifest, indent=2) + '\n')

    def _restore(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        old_args = checkpoint.get('args', {})
        for key in ('problem', 'problem_size', 'pomo_size', 'train_batch_size', 'train_episodes',
                    'lr', 'weight_decay', 'milestones', 'gamma', 'seed'):
            if key in old_args and old_args[key] != getattr(self.args, key):
                raise ValueError('Resume configuration mismatch for {}: {} != {}'.format(
                    key, old_args[key], getattr(self.args, key)))
        for key, value in checkpoint['model_params'].items():
            if key not in ('device', 'objective_scales') and value != self.model_params.get(key):
                raise ValueError('Resume model configuration mismatch for {}'.format(key))
        scales = checkpoint['model_state_dict']['objective_scales'].detach().cpu()
        if not torch.equal(scales, self.model.objective_scales.detach().cpu()):
            raise ValueError('Resume objective scales differ from the checkpoint')
        if checkpoint.get('scales_data') != self.scale_data:
            raise ValueError('Resume calibration provenance differs from the checkpoint')
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.total_instances = checkpoint['total_instances']
        self.start_epoch = checkpoint.get('next_epoch', checkpoint['epoch'] + 1)
        self.start_episode = checkpoint.get('epoch_episode', 0)
        self.start_epoch_metrics = np.asarray(checkpoint.get('epoch_metric_sum', self.start_epoch_metrics), dtype=np.float64)
        states = checkpoint['rng_states']
        if len(states) != self.world_size:
            raise ValueError('Exact resume requires the original distributed world size')
        restore_rng(states[self.rank], self.device)
        if self.rank == 0:
            print('>> Resume {} at epoch {}, episode {}, step {}'.format(
                path, self.start_epoch, self.start_episode, self.global_step), flush=True)

    def _sample_control(self):
        control = torch.empty(2, device=self.device)
        if self.rank == 0:
            control[0] = random.randrange(len(self.envs))
            control[1] = torch.rand((), device=self.device)
        if self.world_size > 1:
            dist.broadcast(control, src=0)
        task_index = int(control[0].item())
        preference = torch.stack((control[1], 1 - control[1]))
        return task_index, preference

    def _write_log(self, record):
        if self.rank == 0:
            with (self.log_path / 'train.jsonl').open('a') as f:
                f.write(json.dumps(record, allow_nan=False) + '\n')

    def _save(self, epoch, episode, metric_sum, completed, periodic=False):
        state = rng_state(self.device)
        states = [state]
        if self.world_size > 1:
            states = [None] * self.world_size
            dist.all_gather_object(states, state)
        if self.rank == 0:
            checkpoint = {
                'epoch': epoch if completed else epoch - 1,
                'next_epoch': epoch + 1 if completed else epoch,
                'epoch_episode': 0 if completed else episode,
                'epoch_metric_sum': [0.0] * len(self.metric_names) if completed else metric_sum.tolist(),
                'global_step': self.global_step, 'total_instances': self.total_instances,
                'problem': self.args.problem, 'args': vars(self.args),
                'env_params': self.env_params, 'model_params': self.model_params,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scales_data': self.scale_data, 'rng_states': states,
                'protocol': 'PMOCO conditioned decoder experts; fixed-scale TCH; POMO; global batch DDP',
            }
            temporary = self.log_path / 'latest.pt.tmp'
            torch.save(checkpoint, temporary)
            temporary.replace(self.log_path / 'latest.pt')
            if periodic:
                shutil.copyfile(self.log_path / 'latest.pt', self.log_path / 'epoch-{}.pt'.format(epoch))
        if self.world_size > 1:
            dist.barrier()

    def run(self):
        if self.args.max_steps and self.global_step >= self.args.max_steps:
            if self.rank == 0:
                print('>> Debug step limit already reached: {}'.format(self.global_step), flush=True)
            return
        start_time = time.monotonic()
        starting_instances = self.total_instances
        self.model.train()
        self.model.set_eval_type(self.model_params['eval_type'])
        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            episode = self.start_episode if epoch == self.start_epoch else 0
            metric_sum = self.start_epoch_metrics.copy() if epoch == self.start_epoch else np.zeros(len(self.metric_names))
            epoch_start = time.monotonic()
            while episode < self.trainer_params['train_episodes']:
                global_batch = min(self.trainer_params['train_batch_size'],
                                   self.trainer_params['train_episodes'] - episode)
                if global_batch % self.world_size:
                    raise ValueError('Every global batch, including the last batch, must divide world_size')
                local_batch = global_batch // self.world_size
                task_index, preference = self._sample_control()
                env = MultiObjectiveEnv(self.envs[task_index](**self.env_params))
                data = env.get_random_problems(local_batch, self.env_params['problem_size'])
                self.optimizer.zero_grad(set_to_none=True)
                step_start = time.monotonic()
                loss, metrics = self.rollout_model(data, env, preference)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float('inf'), error_if_nonfinite=True, foreach=True)
                self.optimizer.step()
                if self.world_size > 1:
                    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                    metrics /= self.world_size
                values = metrics.detach().cpu().double().numpy()
                elapsed_step = time.monotonic() - step_start
                episode += global_batch
                self.global_step += 1
                self.total_instances += global_batch
                metric_sum += values * global_batch
                if self.global_step == 1 or self.global_step % self.args.step_log_interval == 0:
                    record = dict(zip(self.metric_names, values.tolist()))
                    memory = torch.zeros(2, device=self.device)
                    if self.device.type == 'cuda':
                        memory[0] = torch.cuda.max_memory_allocated(self.device)
                        memory[1] = torch.cuda.max_memory_reserved(self.device)
                    if self.world_size > 1:
                        dist.all_reduce(memory, op=dist.ReduceOp.MAX)
                    record.update(type='step', epoch=epoch, epoch_episode=episode,
                                  step=self.global_step, total_instances=self.total_instances,
                                  task=env.problem, preference=preference.detach().cpu().tolist(),
                                  lr=self.optimizer.param_groups[0]['lr'], grad_norm=float(grad_norm),
                                  peak_gpu_memory_allocated_bytes=int(memory[0].item()),
                                  peak_gpu_memory_reserved_bytes=int(memory[1].item()),
                                  seconds=elapsed_step, instances_per_second=global_batch / elapsed_step)
                    self._write_log(record)
                    if self.rank == 0:
                        print('epoch={} step={} instances={} task={} cost={:.5f} total={:.5f} max={:.5f} loss={:.5f} {:.2f} instances/s'.format(
                            epoch, self.global_step, self.total_instances, env.problem,
                            values[3], values[4], values[5], values[0], global_batch / elapsed_step), flush=True)
                if self.args.max_steps and self.global_step >= self.args.max_steps:
                    completed = episode == self.trainer_params['train_episodes']
                    if completed:
                        self.scheduler.step()
                    self._save(epoch, episode, metric_sum, completed, periodic=completed)
                    if self.rank == 0:
                        self._write_log({'type': 'debug_stop', 'epoch': epoch,
                                         'step': self.global_step, 'total_instances': self.total_instances})
                    return

            self.scheduler.step()
            means = metric_sum / self.trainer_params['train_episodes']
            record = dict(zip(self.metric_names, means.tolist()))
            record.update(type='epoch', epoch=epoch, step=self.global_step,
                          total_instances=self.total_instances, seconds=time.monotonic() - epoch_start,
                          lr=self.optimizer.param_groups[0]['lr'])
            self._write_log(record)
            periodic = epoch % self.args.save_interval == 0 or epoch == self.trainer_params['epochs']
            self._save(epoch, episode, metric_sum, completed=True, periodic=periodic)
            if self.rank == 0:
                throughput = (self.total_instances - starting_instances) / (time.monotonic() - start_time)
                remaining = (self.trainer_params['epochs'] - epoch) * self.trainer_params['train_episodes']
                print('Epoch {:4d}/{:4d} complete: total={:.5f}, max={:.5f}, loss={:.5f}; {:.2f} instances/s, ETA {:.2f}h'.format(
                    epoch, self.trainer_params['epochs'], means[4], means[5], means[0], throughput,
                    remaining / max(throughput, 1e-12) / 3600), flush=True)
        self._write_log({'type': 'training_complete', 'epoch': self.trainer_params['epochs'],
                         'step': self.global_step, 'total_instances': self.total_instances,
                         'seconds': time.monotonic() - start_time})
