import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from models import MOEModel
from objectives import MultiObjectiveEnv, route_objectives
from utils import get_env, seed_everything


def nondominated_mask(points):
    points = np.asarray(points, dtype=np.float64)
    dominated = (points[None] <= points[:, None]).all(2) & (points[None] < points[:, None]).any(2)
    return ~dominated.any(1)


def hypervolume(points, reference):
    points = np.asarray(points, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    points = points[np.isfinite(points).all(1) & (points < reference).all(1)]
    if not len(points):
        return 0.0
    points = points[nondominated_mask(points)]
    height, volume = reference[1], 0.0
    for x, y in points[np.argsort(points[:, 0], kind='stable')]:
        if y < height:
            volume += (reference[0] - x) * (height - y)
            height = y
    return float(volume)


def best_candidates(objectives, pref, scales):
    cost = (objectives / scales * pref).amax(-1)
    mask = cost == cost.amin(1, keepdim=True)
    first = objectives[..., 0].masked_fill(~mask, float('inf'))
    mask = mask & (first == first.amin(1, keepdim=True))
    return objectives[..., 1].masked_fill(~mask, float('inf')).argmin(1)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


class Tester:
    def __init__(self, args, **unused):
        self.args, self.device = args, args.device
        self.checkpoint = torch.load(args.checkpoint, map_location=self.device, weights_only=False)
        self.params = dict(self.checkpoint['model_params'])
        self.params.update(device=self.device, eval_type='argmax')
        self.scales = torch.as_tensor(self.checkpoint['scales_data']['scales'], device=self.device)
        self.params['objective_scales'] = self.scales.tolist()
        self.model = MOEModel(**self.params).to(self.device)
        self.model.load_state_dict(self.checkpoint['model_state_dict'], strict=True)
        self.model.eval()
        self.model.set_eval_type('argmax')
        self.preferences = torch.stack((torch.linspace(1, 0, args.preferences, device=self.device),
                                        torch.linspace(0, 1, args.preferences, device=self.device)), 1)
        self.output = Path(args.output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.checkpoint_hash = file_sha256(args.checkpoint)
        root = Path(__file__).resolve().parent
        self.source_hashes = {str(path.relative_to(root)): file_sha256(path)
                              for path in sorted(root.rglob('*.py'))
                              if '__pycache__' not in path.parts and 'tests' not in path.parts}

    def run(self):
        cases = [(n, cls) for n in self.args.problem_sizes for cls in get_env(self.args.problem)]
        results = []
        for index, (n, env_class) in enumerate(cases):
            if index % self.args.num_shards == self.args.shard_index:
                results.append(self._test_case(n, env_class))
        atomic_json(self.output / f'summary-shard-{self.args.shard_index}.json', results)
        return results

    @torch.no_grad()
    def _test_case(self, n, env_class):
        base = env_class(problem_size=n, pomo_size=n, device=self.device)
        name = base.problem
        data_path = Path(self.args.data_root) / name / f'{name.lower()}{n}_uniform.pkl'
        data_hash = file_sha256(data_path)
        result_path = self.output / f'{name}_n{n}.npz'
        metadata_path = self.output / f'{name}_n{n}.json'
        if result_path.exists() and metadata_path.exists() and not self.args.overwrite:
            previous = json.loads(metadata_path.read_text())
            expected = (self.checkpoint_hash, self.args.test_episodes, self.args.preferences,
                        self.args.aug_factor, self.args.seed, self.args.audit_routes)
            actual = (previous['checkpoint_sha256'], previous['instances'], previous['preferences'],
                      previous['augmentation'], previous['seed'], previous['audit_routes'])
            if expected != actual or file_sha256(result_path) != previous['result_sha256']:
                raise ValueError(f'Result configuration/checksum mismatch: {metadata_path}')
            if previous.get('data_sha256') != data_hash or previous.get('evaluation_sources') != self.source_hashes:
                raise ValueError(f'Result data/source provenance mismatch: {metadata_path}; use --overwrite intentionally')
            print(f'Already complete: {name} n={n}', flush=True)
            return previous
        seed_everything(self.args.seed + n * 100 + list(get_env('ALL')).index(env_class))
        data = base.load_dataset(str(data_path), num_samples=self.args.test_episodes)
        actual_count = data[0].size(0)
        if actual_count != self.args.test_episodes:
            raise ValueError(f'Expected {self.args.test_episodes} instances, got {actual_count}: {data_path}')
        for xy in data[:2]:
            if not torch.isfinite(xy).all() or (xy < 0).any() or (xy > 1).any():
                raise ValueError('Geometric HV reference requires unit-square coordinates: ' + str(data_path))
        k, max_steps = self.args.preferences, 2 * n + 2
        all_objectives = np.empty((actual_count, k, 2), dtype=np.float32)
        all_routes = np.zeros((actual_count, k, max_steps), dtype=np.uint16)
        all_lengths = np.zeros((actual_count, k), dtype=np.int32)
        started = time.time()
        for offset in range(0, actual_count, self.args.test_batch_size):
            count = min(self.args.test_batch_size, actual_count - offset)
            batch = tuple(x[offset:offset + count] for x in data)
            for pref_index, pref in enumerate(self.preferences):
                env = MultiObjectiveEnv(env_class(problem_size=n, pomo_size=n, device=self.device))
                env.load_problems(count, problems=batch, aug_factor=self.args.aug_factor)
                reset_state, _, _ = env.reset()
                self.model.assign(pref)
                self.model.pre_forward(reset_state)
                state, reward, done = env.pre_step()
                steps = 0
                while not done:
                    selected, _ = self.model(state)
                    if not torch.isfinite(state.ninf_mask.gather(2, selected[..., None])).all():
                        raise RuntimeError(f'Invalid masked action: {name} step={steps}')
                    state, reward, done = env.step(selected)
                    steps += 1
                    if steps > max_steps:
                        raise RuntimeError(f'Non-terminating rollout: {name}')
                group = env.pomo_size
                costs = (-reward).reshape(self.args.aug_factor, count, group, 2)
                costs = costs.permute(1, 0, 2, 3).reshape(count, -1, 2)
                selected_index = best_candidates(costs, pref, self.scales)
                seq = env.selected_node_list.reshape(self.args.aug_factor, count, group, steps)
                seq = seq.permute(1, 0, 2, 3).reshape(count, -1, steps)
                route = seq[torch.arange(count, device=self.device), selected_index]
                chosen = costs[torch.arange(count, device=self.device), selected_index]
                original_xy = torch.cat((batch[0], batch[1]), 1)
                canonical = route_objectives(original_xy, route[:, None], name.startswith('O'))[:, 0]
                torch.testing.assert_close(chosen, canonical, atol=5e-4, rtol=2e-5)
                all_objectives[offset:offset + count, pref_index] = canonical.cpu().numpy()
                all_routes[offset:offset + count, pref_index, :steps] = route.cpu().numpy().astype(np.uint16)
                all_lengths[offset:offset + count, pref_index] = steps
            print(f'{name} n={n}: {offset + count}/{actual_count}; {time.time()-started:.1f}s', flush=True)
        if self.args.audit_routes:
            self._audit(env_class, n, data, all_routes, all_objectives)
        nondominated = np.stack([nondominated_mask(x) for x in all_objectives])
        reference = np.array([2 * n * np.sqrt(2) + 1e-5, (n + 1) * np.sqrt(2) + 1e-5])
        hv = np.array([hypervolume(x, reference) for x in all_objectives])
        temp = result_path.with_suffix('.tmp.npz')
        np.savez_compressed(temp, objectives=all_objectives, preferences=self.preferences.cpu().numpy(),
                            routes=all_routes, route_lengths=all_lengths, nondominated=nondominated,
                            hypervolume=hv, hv_reference=reference)
        temp.replace(result_path)
        summary = dict(problem=name, problem_size=n, instances=actual_count, preferences=k,
                       augmentation=self.args.aug_factor, seed=self.args.seed,
                       checkpoint=str(Path(self.args.checkpoint).resolve()), checkpoint_sha256=self.checkpoint_hash,
                       checkpoint_epoch=self.checkpoint['epoch'], data_path=str(data_path.resolve()),
                       data_sha256=data_hash, evaluation_sources=self.source_hashes, scales=self.scales.tolist(),
                       train_tasks=['CVRP', 'OVRP', 'VRPB', 'VRPL', 'VRPTW', 'OVRPTW'],
                       training_size=self.checkpoint['env_params']['problem_size'],
                       hypervolume_mean=float(hv.mean()), hypervolume_ratio_mean=float((hv/reference.prod()).mean()),
                       hv_reference=reference.tolist(), hv_reference_policy='unit_square_geometric_upper_bound',
                       unique_nondominated_mean=float(np.mean([len(np.unique(x[m], axis=0))
                                                             for x, m in zip(all_objectives, nondominated)])),
                       endpoint_total=all_objectives[:, 0].mean(0).tolist(),
                       endpoint_longest=all_objectives[:, -1].mean(0).tolist(),
                       audit_routes=self.args.audit_routes, feasible_fraction=1.0,
                       elapsed_seconds=time.time()-started, result_sha256=file_sha256(result_path))
        atomic_json(metadata_path, summary)
        return summary

    @torch.no_grad()
    def _audit(self, env_class, n, data, routes, expected):
        flat_routes, flat_expected = routes.reshape(-1, routes.shape[-1]), expected.reshape(-1, 2)
        for offset in range(0, len(flat_routes), self.args.audit_batch_size):
            count = min(self.args.audit_batch_size, len(flat_routes) - offset)
            indices = torch.arange(offset, offset + count, device=self.device) // self.args.preferences
            batch = tuple(x[indices] for x in data)
            env = MultiObjectiveEnv(env_class(problem_size=n, pomo_size=1, device=self.device))
            env.load_problems(count, problems=batch, aug_factor=1)
            env.reset()
            state, reward, done = env.pre_step()
            route = torch.as_tensor(flat_routes[offset:offset + count].astype(np.int64), device=self.device)
            counts = torch.zeros(count, n + 1, device=self.device, dtype=torch.long)
            counts.scatter_add_(1, route, torch.ones_like(route))
            if not (counts[:, 1:] == 1).all():
                raise RuntimeError(f'Customer visit audit failed: {env.problem}')
            for step in range(route.size(1)):
                selected = route[:, step:step+1]
                if not torch.isfinite(state.ninf_mask.gather(2, selected[..., None])).all():
                    raise RuntimeError(f'Route replay mask violation: {env.problem} step={step}')
                state, reward, done = env.step(selected)
                if done:
                    break
            if not done:
                raise RuntimeError(f'Route replay did not finish: {env.problem}')
            target = torch.as_tensor(flat_expected[offset:offset + count], device=self.device)
            torch.testing.assert_close((-reward)[:, 0], target, atol=5e-4, rtol=2e-5)
