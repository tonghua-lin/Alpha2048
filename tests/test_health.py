import unittest
import itertools

import numpy as np
import torch

from alpha2048.gpu_search import GPUSearch, encode, reference_search
from alpha2048.health import features, health, reference_features, reference_health, VARIANTS


class HealthTests(unittest.TestCase):
    def test_compression_exhaustive_short_rows(self):
        # Includes zeros, odd/even runs and multiple separate merge groups.
        rows = np.array(list(itertools.product((0, 2, 4, 8), repeat=4)))
        boards = np.repeat(rows[:, None, :], 4, axis=1)
        actual = features(torch.tensor(encode(boards)), include_order=False)[2].numpy()
        np.testing.assert_array_equal(actual, [reference_features(b)[2] for b in boards])
        # A single row prevents vertical merges from masking a horizontal error.
        boards[:, 1:, :] = 0
        actual = features(torch.tensor(encode(boards)), include_order=False)[2].numpy()
        np.testing.assert_array_equal(actual, [reference_features(b)[2] for b in boards])

    def test_scalar_agreement_bounds_and_invariance(self):
        rng = np.random.default_rng(510)
        z = rng.integers(0, 10, (64, 4, 4))
        boards = np.where(z, 2 ** z, 0)
        actual = np.stack([x.numpy() for x in features(torch.tensor(encode(boards)))], axis=1)
        np.testing.assert_allclose(actual, [reference_features(b) for b in boards], atol=2e-7)
        self.assertTrue(((actual >= 0) & (actual <= 1)).all())
        for transformed in (boards * 8, np.rot90(boards, axes=(1, 2)).copy(), boards[:, :, ::-1].copy()):
            transformed_features = np.stack([x.numpy() for x in features(torch.tensor(encode(transformed)))], axis=1)
            np.testing.assert_allclose(actual, transformed_features, atol=2e-7)

    def test_constructed_health_cases(self):
        sparse = np.zeros((4, 4), dtype=np.int64)
        sparse[0, 0] = 2
        self.assertEqual(reference_features(sparse), reference_features(sparse * 1024))
        ordered = np.array([[256, 128, 64, 32], [128, 64, 32, 16],
                            [64, 32, 16, 8], [32, 16, 8, 4]])
        scrambled = ordered.copy()
        scrambled[0] = scrambled[0, [0, 2, 1, 3]]
        self.assertEqual(reference_features(ordered)[1], 0)
        self.assertGreater(reference_features(scrambled)[1], 0)
        self.assertGreater(reference_health(ordered, "space-order"), reference_health(scrambled, "space-order"))
        full_pairs = np.full((4, 4), 2)
        full_rigid = np.array([[2, 4, 2, 4], [4, 2, 4, 2]] * 2)
        self.assertEqual(reference_features(full_pairs)[2], 1)
        self.assertEqual(reference_features(full_rigid)[2], 0)
        self.assertGreater(reference_health(full_pairs), reference_health(full_rigid))
        ready = np.zeros((4, 4), dtype=np.int64)
        ready[0] = [16, 8, 4, 4]
        rigid = ready.copy()
        rigid[0] = [32, 16, 8, 4]
        self.assertEqual(reference_features(ready)[:2], reference_features(rigid)[:2])
        self.assertGreater(reference_features(ready)[2], reference_features(rigid)[2])
        self.assertGreater(reference_health(ready), reference_health(rigid))
        for variant in VARIANTS[1:]:
            for board in (sparse, ordered, scrambled, full_pairs, full_rigid):
                value = health(torch.tensor(encode(board[None])), variant).item()
                self.assertAlmostEqual(value, reference_health(board, variant), places=6)

    def test_invalid_configuration(self):
        for weights in ((1, 10, 1), (1, -1, 1), (1, float('nan'), 1), (1, 2)):
            with self.assertRaises(ValueError):
                GPUSearch(heuristic_name="health", weights=weights)
        with self.assertRaises(ValueError):
            GPUSearch(heuristic_name="unknown")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_gpu_search_against_independent_reference(self):
        board = np.array([[2, 4, 8, 16], [32, 64, 128, 256],
                          [512, 1024, 2, 4], [8, 16, 32, 32]])
        for variant in VARIANTS[1:]:
            for depth in (0, 1, 2):
                expected = reference_search(board, depth, lambda x: reference_health(x, variant))
                for cap in (32, 4096):
                    actual, _ = GPUSearch(cap, heuristic_name=variant).search(board[None], depth)
                    np.testing.assert_allclose(actual[0], expected, atol=3e-6)
        terminal = board.copy()
        terminal[-1, -1] = 64
        q, _ = GPUSearch(heuristic_name="health").search(terminal[None], 1)
        self.assertTrue(np.isneginf(q).all())


if __name__ == "__main__":
    unittest.main()
