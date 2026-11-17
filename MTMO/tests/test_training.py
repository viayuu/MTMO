import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Trainer import Trainer
from train import args2dict, get_parser
from utils import seed_everything


class TrainingResumeTests(unittest.TestCase):
    def make_args(self, directory, scales, checkpoint=None, max_steps=0, topk=2):
        arguments = ['--no_cuda', '--problem', 'CVRP', '--problem_size', '20', '--pomo_size', '4',
                     '--embedding_dim', '16', '--head_num', '2', '--qkv_dim', '8',
                     '--encoder_layer_num', '1', '--ff_hidden_dim', '32', '--expert_loc', 'Enc0', 'Dec',
                     '--scales', str(scales), '--run_dir', str(directory), '--epochs', '2',
                     '--train_episodes', '8', '--train_batch_size', '4', '--step_log_interval', '1',
                     '--max_steps', str(max_steps), '--topk', str(topk)]
        if checkpoint:
            arguments += ['--checkpoint', str(checkpoint)]
        args = get_parser().parse_args(arguments)
        args.rank = args.local_rank = 0
        args.world_size = 1
        args.device = torch.device('cpu')
        args.log_path = str(directory)
        return args

    def run_training(self, args):
        seed_everything(args.seed)
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = Trainer(args, *args2dict(args))
            trainer.run()
        return trainer

    def test_mid_epoch_resume_and_semantic_configuration_lock(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                scales = root / 'scales.json'
                scales.write_text(json.dumps({'scales': [12., 3.], 'debug_only': True}))
                interrupted = root / 'interrupted'
                self.run_training(self.make_args(interrupted, scales, max_steps=1))
                partial = torch.load(interrupted / 'latest.pt', weights_only=False, map_location='cpu')
                self.assertEqual(partial['epoch'], 0)
                self.assertEqual(partial['epoch_episode'], 4)
                self.run_training(self.make_args(interrupted, scales, checkpoint=interrupted / 'latest.pt'))
                continuous = root / 'continuous'
                self.run_training(self.make_args(continuous, scales))
                resumed = torch.load(interrupted / 'latest.pt', weights_only=False, map_location='cpu')
                expected = torch.load(continuous / 'latest.pt', weights_only=False, map_location='cpu')
                self.assertEqual(resumed['global_step'], 4)
                self.assertEqual(resumed['total_instances'], 16)
                self.assertEqual(resumed['scheduler_state_dict'], expected['scheduler_state_dict'])
                for name, value in resumed['model_state_dict'].items():
                    self.assertTrue(torch.equal(value, expected['model_state_dict'][name]), name)
                self.assertTrue(torch.equal(resumed['rng_states'][0]['torch'], expected['rng_states'][0]['torch']))
                bad_args = self.make_args(interrupted, scales, checkpoint=interrupted / 'latest.pt', topk=1)
                with self.assertRaisesRegex(ValueError, 'topk'):
                    self.run_training(bad_args)
        finally:
            torch.set_num_threads(old_threads)


if __name__ == '__main__':
    unittest.main()
