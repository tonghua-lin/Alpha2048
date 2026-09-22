import unittest

import numpy as np
import torch

from alpha2048.core import slide
from alpha2048.gpu_search import GPUSearch, encode, heuristic, moves, reference_heuristic, reference_search


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GPUSearchTests(unittest.TestCase):
    def test_rules_and_heuristic(self):
        rng = np.random.default_rng(41)
        z = rng.integers(0, 12, (256, 4, 4))
        boards = np.where(z, 2 ** z, 0)
        boards[0] = [[2, 2, 2, 2], [4, 4, 8, 8], [0, 2, 0, 2], [2, 2, 4, 0]]
        b = torch.tensor(encode(boards), device="cuda")
        children, legal, rewards = moves(b)
        children, legal, rewards = children.cpu().numpy(), legal.cpu().numpy(), rewards.cpu().numpy()
        for i, board in enumerate(boards):
            for a in range(4):
                expected, reward, changed = slide(board, a)
                np.testing.assert_array_equal(children[i, a], encode(expected[None])[0])
                self.assertEqual(legal[i, a], changed)
                self.assertEqual(rewards[i, a], reward)
        np.testing.assert_allclose(heuristic(b).cpu(), [reference_heuristic(x) for x in boards], atol=1e-6)

    def test_search_depth_terminal_and_chunks(self):
        boards = np.array([
            [[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2, 4], [8, 16, 32, 32]],
            [[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2, 4], [8, 16, 32, 64]],
            [[2, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2], [4, 8, 0, 0]],
        ])
        for depth in (0, 1, 2):
            expected = np.stack([reference_search(b, depth) for b in boards])
            for cap in (32, 4096):
                actual, _ = GPUSearch(cap).search(boards, depth)
                np.testing.assert_allclose(actual, expected, atol=3e-6)

    def test_symmetry(self):
        board = np.array([[2, 2, 8, 0], [32, 16, 4, 0], [64, 4, 0, 0], [128, 8, 2, 0]])
        engine = GPUSearch()
        q, _ = engine.search(board[None], 1)
        rotated, _ = engine.search(np.rot90(board, -1)[None].copy(), 1)
        np.testing.assert_allclose(rotated[0], np.roll(q[0], 1), atol=1e-6)
        for k in range(4):
            self.assertAlmostEqual(reference_heuristic(board), reference_heuristic(np.rot90(board, k)[:, ::-1]))


if __name__ == "__main__":
    unittest.main()
