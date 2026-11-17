import argparse
from pathlib import Path

import torch
from Tester import Tester


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTMO: per-instance multi-objective evaluation')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--problem', default='ALL')
    parser.add_argument('--problem_sizes', nargs='+', type=int, default=[100])
    parser.add_argument('--test_episodes', type=int, default=1000)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--preferences', type=int, default=101)
    parser.add_argument('--aug_factor', type=int, choices=[1, 8], default=8)
    parser.add_argument('--data_root', default=str(Path(__file__).resolve().parent / 'data'))
    parser.add_argument('--shard_index', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--num_threads', type=int, default=2)
    parser.add_argument('--audit_routes', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--audit_batch_size', type=int, default=128)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.preferences < 2 or not 0 <= args.shard_index < args.num_shards:
        parser.error('Need at least 2 preferences and a valid shard index.')
    if min(args.test_episodes, args.test_batch_size, args.audit_batch_size, args.num_threads) < 1:
        parser.error('Instance counts, batch sizes and thread count must be positive.')
    torch.set_num_threads(args.num_threads)
    if args.no_cuda:
        args.device = torch.device('cpu')
        torch.set_default_tensor_type(torch.FloatTensor)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable: run inside the GPU allocation.')
        torch.cuda.set_device(args.gpu_id)
        args.device = torch.device('cuda', args.gpu_id)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    Tester(args).run()
