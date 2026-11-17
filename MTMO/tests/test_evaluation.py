import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Tester import best_candidates, hypervolume, nondominated_mask


class EvaluationTest(unittest.TestCase):
    def test_front_and_exact_rectangle_union(self):
        points = np.array([[1., 4.], [2., 2.], [4., 1.], [3., 3.], [2., 2.]])
        self.assertEqual(nondominated_mask(points).tolist(), [True, True, True, False, True])
        self.assertAlmostEqual(hypervolume(points, [5., 5.]), 11.)
        self.assertAlmostEqual(hypervolume(np.vstack((points, [10., .1])), [5., 5.]), 11.)

    def test_units_and_selection_preserve_actual_candidate(self):
        points = np.array([[1., 4.], [2., 2.], [4., 1.]])
        reference, scales = np.array([5., 5.]), np.array([2., 4.])
        hvr = hypervolume(points, reference) / reference.prod()
        self.assertAlmostEqual(hypervolume(points / scales, reference / scales) / (reference / scales).prod(), hvr)
        candidates = torch.tensor([[[1., 4.], [4., 1.]]])
        for pref, expected in [([1., 0.], 0), ([0., 1.], 1), ([.5, .5], 0)]:
            selected = best_candidates(candidates, torch.tensor(pref), torch.ones(2))
            self.assertEqual(selected.item(), expected)
            self.assertIn(tuple(candidates[0, selected].flatten().tolist()), {(1., 4.), (4., 1.)})


if __name__ == '__main__':
    unittest.main()
