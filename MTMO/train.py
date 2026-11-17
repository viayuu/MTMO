import argparse
import json
import os
import pprint as pp
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from Trainer import Trainer
from utils import seed_everything


PROBLEMS = ['Train_ALL', 'CVRP', 'OVRP', 'VRPB', 'VRPL', 'VRPTW', 'OVRPTW',
            'OVRPB', 'OVRPL', 'VRPBL', 'VRPBTW', 'VRPLTW', 'OVRPBL',
            'OVRPBTW', 'OVRPLTW', 'VRPBLTW', 'OVRPBLTW']


def args2dict(args):
    scales = json.loads(Path(args.scales).read_text())['scales']
    if len(scales) != 2 or not all(float(x) > 0 for x in scales):
        raise ValueError('The fixed calibration file must contain two positive scales')
    env_params = {'problem_size': args.problem_size, 'pomo_size': args.pomo_size, 'device': args.device}
    model_params = {'embedding_dim': args.embedding_dim, 'sqrt_embedding_dim': args.embedding_dim ** 0.5,
                    'encoder_layer_num': args.encoder_layer_num, 'decoder_layer_num': args.decoder_layer_num,
                    'qkv_dim': args.qkv_dim, 'head_num': args.head_num, 'logit_clipping': args.logit_clipping,
                    'ff_hidden_dim': args.ff_hidden_dim, 'num_experts': args.num_experts, 'eval_type': args.eval_type,
                    'norm': args.norm, 'norm_loc': args.norm_loc, 'expert_loc': args.expert_loc, 'problem': args.problem,
                    'topk': args.topk, 'routing_level': args.routing_level, 'routing_method': args.routing_method,
                    'device': args.device, 'objective_scales': scales}
    optimizer_params = {'optimizer': {'lr': args.lr, 'weight_decay': args.weight_decay},
                        'scheduler': {'milestones': args.milestones, 'gamma': args.gamma}}
    trainer_params = {'epochs': args.epochs, 'train_episodes': args.train_episodes,
                      'train_batch_size': args.train_batch_size, 'validation_interval': args.validation_interval,
                      'model_save_interval': args.save_interval, 'checkpoint': args.checkpoint}
    return env_params, model_params, optimizer_params, trainer_params


def get_parser():
    parser = argparse.ArgumentParser(description='MTMO: MVMoE backbone with PMOCO conditional decoder experts')
    parser.add_argument('--problem', default='Train_ALL', choices=PROBLEMS)
    parser.add_argument('--problem_size', type=int, default=100)
    parser.add_argument('--pomo_size', type=int, default=100)
    parser.add_argument('--model_type', default='MOE', choices=['MOE'])
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--sqrt_embedding_dim', type=float, default=128 ** 0.5)
    parser.add_argument('--encoder_layer_num', type=int, default=6)
    parser.add_argument('--decoder_layer_num', type=int, default=1)
    parser.add_argument('--qkv_dim', type=int, default=16)
    parser.add_argument('--head_num', type=int, default=8)
    parser.add_argument('--logit_clipping', type=float, default=10)
    parser.add_argument('--ff_hidden_dim', type=int, default=512)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--eval_type', default='argmax', choices=['argmax', 'softmax'])
    parser.add_argument('--norm', default='instance', choices=['batch', 'batch_no_track', 'instance', 'layer', 'rezero', 'none'])
    parser.add_argument('--norm_loc', default='norm_last', choices=['norm_first', 'norm_last'])
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--expert_loc', nargs='+', default=['Enc0', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Enc5', 'Dec'])
    parser.add_argument('--routing_level', default='node', choices=['node', 'instance', 'problem'])
    parser.add_argument('--routing_method', default='input_choice', choices=['input_choice', 'expert_choice', 'soft_moe', 'random'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--milestones', nargs='+', type=int, default=[4501])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--train_episodes', type=int, default=20000)
    parser.add_argument('--train_batch_size', type=int, default=128, help='Global batch, shared equally between DDP ranks')
    parser.add_argument('--validation_interval', type=int, default=50, help='Recorded source setting; full evaluation runs separately')
    parser.add_argument('--save_interval', '--model_save_interval', dest='save_interval', type=int, default=100)
    parser.add_argument('--step_log_interval', type=int, default=10)
    parser.add_argument('--checkpoint', default=None, help='Exact optimizer/model/RNG resume checkpoint')
    parser.add_argument('--scales', required=True, help='Frozen training-task calibration JSON')
    parser.add_argument('--run_dir', default=None)
    parser.add_argument('--log_dir', default='./results')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_threads', type=int, default=2)
    parser.add_argument('--max_steps', type=int, default=0, help='Debug-only absolute global step limit; zero runs the full budget')
    return parser


def main():
    args = get_parser().parse_args()
    args.rank = int(os.environ.get('RANK', '0'))
    args.local_rank = int(os.environ.get('LOCAL_RANK', str(args.gpu_id)))
    args.world_size = int(os.environ.get('WORLD_SIZE', '1'))
    torch.set_num_threads(args.num_threads)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if not args.no_cuda and not use_cuda:
        raise RuntimeError('CUDA unavailable. Run inside the GPU allocation, or use --no_cuda for a CPU check.')
    if use_cuda:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device('cuda', args.local_rank)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        args.device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')
    seed_everything(args.seed + args.rank)
    if args.train_batch_size % args.world_size or args.train_episodes % args.world_size:
        raise ValueError('Global batch and per-epoch episodes must divide the DDP world size')
    if min(args.epochs, args.train_episodes, args.train_batch_size, args.save_interval, args.step_log_interval) <= 0:
        raise ValueError('Training budgets and logging/save intervals must be positive')
    if args.pomo_size > args.problem_size:
        raise ValueError('POMO size cannot exceed problem size')
    if args.world_size > 1 and not args.run_dir and not args.checkpoint:
        raise ValueError('Distributed training requires --run_dir so all ranks use the same directory')
    if args.run_dir:
        args.log_path = str(Path(args.run_dir).resolve())
    elif args.checkpoint:
        args.log_path = str(Path(args.checkpoint).resolve().parent)
    else:
        args.log_path = str((Path(args.log_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')).resolve())
    args.scales = str(Path(args.scales).resolve())
    env_params, model_params, optimizer_params, trainer_params = args2dict(args)
    if args.world_size > 1:
        dist.init_process_group(backend='nccl' if use_cuda else 'gloo', timeout=timedelta(minutes=30))
    if args.rank == 0:
        pp.pprint(vars(args))
        print('>> Training {} with {} on {} ranks; global batch {}'.format(
            args.problem, args.model_type, args.world_size, args.train_batch_size), flush=True)
    try:
        Trainer(args, env_params, model_params, optimizer_params, trainer_params).run()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
