"""Freeze shared objective units using only the six MVMoE training tasks."""

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from objectives import MultiObjectiveEnv
from utils import get_env


ROOT = Path(__file__).resolve().parents[1]


def load_reference_model(checkpoint_path, device, reference_code=None):
    """Load author code under an alias, independent of adapted MTMO models."""
    name = '_original_mvmoe_models'
    directory = Path(reference_code or ROOT / 'Routing-MVMoE') / 'models'
    package = types.ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(name + '.MOEModel', directory / 'MOEModel.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    model_params = {
        'embedding_dim': 128, 'sqrt_embedding_dim': 128 ** 0.5,
        'encoder_layer_num': 6, 'decoder_layer_num': 1,
        'head_num': 8, 'qkv_dim': 16, 'ff_hidden_dim': 512,
        'logit_clipping': 10, 'num_experts': 4, 'topk': 2,
        'eval_type': 'argmax', 'norm': 'instance', 'norm_loc': 'norm_last',
        'expert_loc': ['Enc0', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Enc5', 'Dec'],
        'problem': 'Train_ALL', 'routing_level': 'node',
        'routing_method': 'input_choice', 'device': device,
    }
    model = module.MOEModel(**model_params).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    model.set_eval_type('argmax')
    return model


@torch.no_grad()
def calibrate(args):
    device = torch.device('cpu' if args.no_cuda else 'cuda',
                          None if args.no_cuda else args.gpu_id)
    if not args.no_cuda:
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    checkpoint_path = Path(args.checkpoint).resolve()
    model = load_reference_model(checkpoint_path, device, args.reference_code)
    per_task = {}
    for env_class in get_env('Train_ALL'):
        chunks = []
        for offset in range(0, args.samples_per_task, args.batch_size):
            batch_size = min(args.batch_size, args.samples_per_task - offset)
            env = MultiObjectiveEnv(env_class(problem_size=args.problem_size,
                                              pomo_size=args.problem_size, device=device),
                                    check_feasibility=True)
            data = env.get_random_problems(batch_size, args.problem_size)
            env.load_problems(batch_size, problems=data, aug_factor=1)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)
            state, reward, done = env.pre_step()
            steps = 0
            while not done:
                selected, _ = model(state)
                state, reward, done = env.step(selected)
                steps += 1
                if steps > 2 * args.problem_size + 2:
                    raise RuntimeError('Reference rollout exceeded route step bound')
            objectives = -reward
            best = objectives[:, :, 0].argmin(dim=1)
            chosen = objectives[torch.arange(batch_size, device=device), best]
            chunks.append(chosen.cpu())
        costs = torch.cat(chunks, dim=0).double()
        per_task[env.problem] = {
            'instances': costs.size(0), 'mean': costs.mean(dim=0).tolist(),
            'min': costs.min(dim=0).values.tolist(), 'max': costs.max(dim=0).values.tolist(),
        }
        print('{}: {}'.format(env.problem, per_task[env.problem]['mean']), flush=True)

    scales = np.asarray([entry['mean'] for entry in per_task.values()]).mean(axis=0)
    if not np.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError('Calibration produced non-positive or non-finite scales')
    with checkpoint_path.open('rb') as f:
        checkpoint_hash = hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
    if checkpoint_hash is None:
        digest = hashlib.sha256()
        with checkpoint_path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(chunk)
        checkpoint_hash = digest.hexdigest()
    result = {
        'scales': scales.tolist(), 'objective_names': ['total_distance', 'max_route_distance'],
        'per_task': per_task, 'problem_size': args.problem_size,
        'samples_per_task': args.samples_per_task, 'batch_size': args.batch_size,
        'seed': args.seed, 'checkpoint': str(checkpoint_path), 'checkpoint_sha256': checkpoint_hash,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'source': 'RoyalSkye/Routing-MVMoE@af29e5af0595f94f3ecc3bc46d72df1089a62682',
        'protocol': 'six_train_tasks_equal_weight; original_MOE_4E; argmax; POMO; no_augmentation; best_total_candidate',
        'objective_definition': 'effective_total_and_longest_route_distance_v1',
        'unseen_task_calibration': False,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(output)
    print('Frozen objective scales: {} -> {}'.format(result['scales'], output), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--problem_size', type=int, default=100)
    parser.add_argument('--samples_per_task', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--checkpoint', default=str(ROOT / 'Routing-MVMoE/pretrained/mvmoe_4e_n100/epoch-5000.pt'))
    parser.add_argument('--reference_code', default=None, help='Optional frozen original MVMoE source directory')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--no_cuda', action='store_true')
    arguments = parser.parse_args()
    if arguments.samples_per_task <= 0 or arguments.batch_size <= 0:
        parser.error('Calibration counts must be positive')
    calibrate(arguments)
