import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from objectives import MultiObjectiveEnv, route_objectives, validate_routes
from envs import CVRPEnv, OVRPEnv
from utils import get_env


class RouteObjectivesTest(unittest.TestCase):
    def setUp(self):
        torch.set_default_tensor_type('torch.FloatTensor')
        self.xy = torch.tensor([[[0., 0.], [1., 0.], [0., 2.], [0., 3.]]])
        self.sequence = torch.tensor([[[0, 1, 0, 2, 3, 0]]])

    def test_closed_longest_route_boundary(self):
        torch.testing.assert_close(route_objectives(self.xy, self.sequence), torch.tensor([[[8., 6.]]]))

    def test_open_route(self):
        torch.testing.assert_close(route_objectives(self.xy, self.sequence, True), torch.tensor([[[4., 3.]]]))

    def test_route_reordering_and_padding(self):
        padded = torch.tensor([[[0, 0, 2, 3, 0, 1, 0, 0, 0]]])
        for opened in [False, True]:
            torch.testing.assert_close(route_objectives(self.xy, self.sequence, opened),
                                       route_objectives(self.xy, padded, opened))

    def test_rounding_is_per_edge(self):
        xy = torch.tensor([[[0., 0.], [.14, 0.], [.26, 0.]]])
        sequence = torch.tensor([[[0, 1, 2, 0]]])
        # Rounded edges are .1 + .1 + .3, not round(.52)=.5 by coincidence;
        # the open route also checks .1+.1 rather than round(.26)=.3.
        torch.testing.assert_close(route_objectives(xy, sequence, False, 10), torch.tensor([[[.5, .5]]]))
        torch.testing.assert_close(route_objectives(xy, sequence, True, 10), torch.tensor([[[.2, .2]]]))

    def test_customer_coverage(self):
        self.assertTrue(validate_routes(self.sequence, 3).all())
        self.assertFalse(validate_routes(torch.tensor([[[0, 1, 1, 2, 0]]]), 3).all())

    def test_online_matches_terminal_and_original(self):
        for env_class, expected in [(CVRPEnv, [8., 6.]), (OVRPEnv, [4., 3.])]:
            env = MultiObjectiveEnv(env_class(problem_size=3, pomo_size=1, device=torch.device('cpu')),
                                    check_feasibility=True)
            data = (self.xy[:, :1], self.xy[:, 1:], torch.tensor([[.2, .2, .2]]))
            env.load_problems(1, problems=data)
            env.reset()
            state, _, _ = env.pre_step()
            self.assertEqual(state.total_distance.item(), 0)
            for node in self.sequence[0, 0]:
                state, reward, done = env.step(node.reshape(1, 1))
            self.assertTrue(done)
            torch.testing.assert_close(-reward, torch.tensor([[expected]]))
            torch.testing.assert_close(state.total_distance, torch.tensor([[expected[0]]]))
            torch.testing.assert_close(state.max_route_distance, torch.tensor([[expected[1]]]))

    def test_all_sixteen_original_environments(self):
        torch.manual_seed(321)
        for env_class in get_env('ALL'):
            env = MultiObjectiveEnv(env_class(problem_size=20, pomo_size=20,
                                              device=torch.device('cpu')),
                                    check_feasibility=True)
            data = env.get_random_problems(2, 20)
            env.load_problems(2, problems=data)
            env.reset()
            state, _, done = env.pre_step()
            steps = 0
            while not done:
                if state.selected_count == 0:
                    selected = torch.zeros(2, env.pomo_size, dtype=torch.long)
                elif state.selected_count == 1:
                    selected = state.START_NODE
                else:
                    distance = (env.depot_node_xy[:, None] - state.current_coord[:, :, None]).norm(dim=-1)
                    selected = (-distance + state.ninf_mask).argmax(dim=2)
                state, reward, done = env.step(selected)
                steps += 1
                self.assertLessEqual(steps, 42, env.problem)
            self.assertEqual(tuple(reward.shape), (2, env.pomo_size, 2))
            self.assertTrue((-reward[:, :, 0] >= -reward[:, :, 1] - 1e-5).all(), env.problem)


if __name__ == '__main__':
    unittest.main()
