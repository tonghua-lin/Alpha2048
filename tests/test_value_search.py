import unittest
import numpy as np
import torch

from alpha2048.gpu_search import encode, reference_search
from alpha2048.test_value_games import score_actions
from alpha2048.value_search import ValueSearch, SearchStopped
from alpha2048.value_transformer import ValueTransformer


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class ValueSearchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = ValueTransformer(width=8, heads=2, feedforward=16, layers=1).cuda().eval()
        self.engine = ValueSearch(self.model, cap=64, inference_batch=7)
        self.board = np.array([[2, 2, 8, 16], [32, 64, 128, 256],
                               [512, 1024, 2, 4], [8, 16, 32, 0]], dtype=np.int64)

    def test_frontier_and_chance_backup_match_scalar_oracle(self):
        @torch.inference_mode()
        def evaluate(board):
            return self.model(torch.as_tensor(encode(board[None]), device='cuda')).item()
        for depth in (0, 1, 2):
            actual, stats = self.engine.search(self.board[None], depth)
            expected = reference_search(self.board, depth, evaluate)
            np.testing.assert_allclose(actual[0], expected, atol=2e-6)
            self.assertGreater(stats['leaf_evaluations'], 0)
        direct, _ = score_actions(self.model, self.board)
        actual, _ = self.engine.search(self.board[None], 0)
        np.testing.assert_allclose(actual[0], [v if v is not None else -np.inf for v in direct], atol=2e-6)

    def test_terminal_and_cancellation(self):
        board = np.array([[2, 4, 2, 4], [4, 2, 4, 2]]*2)
        actual, _ = self.engine.search(board[None], 1)
        self.assertTrue(np.isneginf(actual).all())
        self.engine.stop = lambda: True
        with self.assertRaises(SearchStopped):
            self.engine.search(self.board[None], 2)


if __name__ == '__main__':
    unittest.main()
