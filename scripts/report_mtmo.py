"""Build an auditable report only after the complete training/evaluation budget."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', required=True)
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    protocol = json.loads((run / 'protocol.json').read_text())
    checkpoint_path = run / 'training/latest.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    required = protocol['training']
    assert checkpoint['epoch'] == required['epochs']
    assert checkpoint['total_instances'] == required['total_instances']
    assert checkpoint['args']['max_steps'] == 0
    checkpoint_hash = sha256(checkpoint_path)
    expected = protocol['evaluation']
    cases = []
    for n in expected['problem_sizes']:
        for path in sorted((run / 'evaluation').glob(f'*_n{n}.json')):
            case = json.loads(path.read_text())
            assert case['checkpoint_sha256'] == checkpoint_hash, path
            assert case['instances'] == expected['instances_per_task_size'], path
            assert case['preferences'] == expected['preferences'], path
            assert case['augmentation'] == expected['augmentation'], path
            assert case['audit_routes'] and case['feasible_fraction'] == 1., path
            npz_path = path.with_suffix('.npz')
            assert sha256(npz_path) == case['result_sha256'], path
            with np.load(npz_path) as result:
                costs = result['objectives']
                assert costs.shape == (case['instances'], case['preferences'], 2), path
                assert np.isfinite(costs).all() and (costs >= 0).all(), path
            cases.append(case)
    assert len(cases) == expected['tasks'] * len(expected['problem_sizes']), len(cases)
    assert len({(c['problem'], c['problem_size']) for c in cases}) == len(cases)
    epochs = {}
    for line in (run / 'training/train.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['type'] == 'epoch':
            epochs[row['epoch']] = row
    assert set(epochs) == set(range(1, required['epochs'] + 1))
    output = run / 'report'
    output.mkdir(parents=True, exist_ok=True)
    columns = ['problem', 'problem_size', 'instances', 'preferences', 'augmentation',
               'hypervolume_mean', 'hypervolume_ratio_mean', 'unique_nondominated_mean',
               'feasible_fraction', 'elapsed_seconds']
    with (output / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: c[key] for key in columns} for c in cases)
    aggregate = {'checkpoint_epoch': checkpoint['epoch'], 'training_instances': checkpoint['total_instances'],
                 'checkpoint_sha256': checkpoint_hash, 'protocol': protocol, 'cases': cases}
    (output / 'results.json').write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = [epochs[i] for i in sorted(epochs)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, label in zip(axes, ['normalized_cost', 'total_distance', 'max_route_distance'],
                              ['Scalarized cost', 'Total distance', 'Longest route distance']):
        ax.plot([r['epoch'] for r in records], [r[key] for r in records], lw=.8)
        ax.set(xlabel='Epoch', ylabel=label)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / 'training.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(4, 4, figsize=(14, 12))
    primary = sorted([c for c in cases if c['problem_size'] == expected['primary_size']], key=lambda c: c['problem'])
    for ax, c in zip(axes.flat, primary):
        with np.load(run / 'evaluation' / f"{c['problem']}_n{c['problem_size']}.npz") as result:
            points = result['objectives'][0]
            ax.scatter(points[:, 0], points[:, 1], c=result['preferences'][:, 0], s=10, cmap='viridis')
        ax.set_title(c['problem'])
        ax.set(xlabel='Total distance', ylabel='Longest route')
        ax.grid(alpha=.2)
    fig.suptitle('Actual candidate solutions: instance 0 of each task (not the true Pareto front)')
    fig.tight_layout()
    fig.savefig(output / 'candidates_n100.png', dpi=180)
    plt.close(fig)
    lines = ['# MTMO 完整训练与评测结果', '',
             f"训练：N={required['problem_size']}，{checkpoint['epoch']} epochs，{checkpoint['total_instances']:,} 个在线实例。", '',
             f'最终 checkpoint SHA-256：`{checkpoint_hash}`。', '',
             '所有结果使用固定最终 checkpoint。尺度仅由六个训练任务校准，未见任务与规模不再校准。', '',
             'HV 越大越好；参考点使用单位方形几何上界，数值偏宽，仅作为已声明口径下的指标。未使用未知真实前沿计算 IGD。', '',
             '| 任务 | N | 平均HV | 平均HVR | 唯一非支配点数 | 可行比例 |',
             '| --- | --- | --- | --- | --- | --- |']
    for c in sorted(cases, key=lambda c: (c['problem_size'], c['problem'])):
        lines.append('| {} | {} | {:.6f} | {:.6f} | {:.2f} | {:.2%} |'.format(
            c['problem'], c['problem_size'], c['hypervolume_mean'], c['hypervolume_ratio_mean'],
            c['unique_nondominated_mean'], c['feasible_fraction']))
    lines += ['', 'N=100 是主评测；N=50/200 是同一个 N=100 模型的规模泛化评测，不能称为在这些规模训练的结果。', '',
              '每个点绑定一条真实路线；增强/POMO 候选按同一偏好选择；所有保存路线均通过原环境 mask 重放及客户覆盖检查。', '',
              '未进行与独立多目标基线的同预算对比，因此本报告不声称性能提升或方法优越性。', '',
              '![Training](training.png)', '', '![Candidates](candidates_n100.png)', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))
    print('Complete audited report:', output / 'REPORT.md', flush=True)


if __name__ == '__main__':
    main()
