import unittest

import numpy as np

from alpha2048 import Action, Game2048, GameState, is_terminal, legal_actions, slide, spawn_tile


class CoreTests(unittest.TestCase):
    def test_merge_once_and_skip_gaps(self):
        board = np.array([[2, 2, 2, 2], [2, 0, 2, 4], [4, 4, 8, 0], [0, 0, 0, 0]])
        before = board.copy()
        moved, reward, changed = slide(board, Action.LEFT)
        np.testing.assert_array_equal(moved, [[4, 4, 0, 0], [4, 4, 0, 0], [8, 8, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(reward, 20)
        self.assertTrue(changed)
        np.testing.assert_array_equal(board, before)

    def test_all_directions(self):
        board = np.array([[2, 0], [0, 2]])
        expected = {
            Action.UP: [[2, 2], [0, 0]], Action.RIGHT: [[0, 2], [0, 2]],
            Action.DOWN: [[0, 0], [2, 2]], Action.LEFT: [[2, 0], [2, 0]],
        }
        for action, target in expected.items():
            with self.subTest(action=action):
                result, reward, changed = slide(board, action)
                np.testing.assert_array_equal(result, target)
                self.assertEqual(reward, 0)
                self.assertTrue(changed)

    def test_terminal_and_2048_continuation(self):
        self.assertTrue(is_terminal(np.array([[2, 4], [8, 16]])))
        self.assertFalse(is_terminal(np.array([[2048, 2048], [8, 16]])))
        self.assertEqual(legal_actions(np.array([[2, 2], [4, 8]])), (Action.RIGHT, Action.LEFT))

    def test_spawn_and_full_board_rng(self):
        rng = np.random.default_rng(12)
        board = np.zeros((4, 4), dtype=np.int64)
        result = spawn_tile(board, rng)
        self.assertEqual(np.count_nonzero(result), 1)
        self.assertIn(result.max(), (2, 4))
        self.assertFalse(board.any())
        full = np.full((2, 2), 2)
        before = rng.bit_generator.state
        np.testing.assert_array_equal(spawn_tile(full, rng), full)
        self.assertEqual(rng.bit_generator.state, before)

    def test_invalid_inputs(self):
        for board in (np.zeros((2, 3), dtype=int), np.array([[1, 0], [0, 0]]),
                      np.array([[-2, 0], [0, 0]]), np.zeros((2, 2), dtype=float)):
            with self.assertRaises(ValueError):
                slide(board, Action.LEFT)
        with self.assertRaises(ValueError):
            slide(np.zeros((2, 2), dtype=int), 4)


class EnvironmentTests(unittest.TestCase):
    def set_board(self, env, board, score=0):
        env.set_state(GameState(np.array(board), score, env.get_state().rng_state))

    def test_seed_reset_and_detached_arrays(self):
        env = Game2048(seed=11)
        other = Game2048(seed=11)
        np.testing.assert_array_equal(env.board, other.board)
        initial = env.reset(seed=11)
        self.assertEqual(np.count_nonzero(initial), 2)
        initial[:] = 0
        self.assertEqual(np.count_nonzero(env.board), 2)
        snapshot = env.get_state()
        snapshot.board[:] = 0
        self.assertEqual(np.count_nonzero(env.board), 2)

    def test_invalid_move_does_not_spawn_or_consume_rng(self):
        env = Game2048(size=2, seed=7)
        self.set_board(env, [[2, 0], [4, 0]], score=8)
        before = env.get_state()
        result = env.step(Action.LEFT)
        self.assertFalse(result.changed)
        self.assertEqual(result.reward, 0)
        self.assertEqual(result.score, 8)
        np.testing.assert_array_equal(result.board, before.board)
        self.assertEqual(env.get_state().rng_state, before.rng_state)

    def test_merge_score_spawn_and_result_copy(self):
        env = Game2048(size=2, seed=7)
        self.set_board(env, [[2, 2], [0, 0]], score=8)
        result = env.step(Action.LEFT)
        self.assertEqual(result.reward, 4)
        self.assertEqual(result.score, 12)
        self.assertTrue(result.changed)
        self.assertEqual(result.board[0, 0], 4)
        self.assertEqual(np.count_nonzero(result.board), 2)
        result.board[:] = 0
        self.assertEqual(np.count_nonzero(env.board), 2)

    def test_snapshot_and_clone_exact_continuation(self):
        env = Game2048(seed=123)
        snapshot = env.get_state()
        clone = env.clone()
        actions = [Action.LEFT, Action.DOWN, Action.RIGHT, Action.UP] * 10
        expected = []
        for action in actions:
            result = env.step(action)
            other = clone.step(action)
            np.testing.assert_array_equal(result.board, other.board)
            self.assertEqual(result.score, other.score)
            expected.append(result)
        env.set_state(snapshot)
        for action, target in zip(actions, expected):
            result = env.step(action)
            np.testing.assert_array_equal(result.board, target.board)
            self.assertEqual(result.reward, target.reward)
        clone.reset()
        np.testing.assert_array_equal(env.board, expected[-1].board)

    def test_terminal_step_and_atomic_restore(self):
        env = Game2048(size=2)
        self.set_board(env, [[2, 4], [8, 16]])
        before = env.get_state()
        result = env.step(Action.UP)
        self.assertTrue(result.terminated)
        self.assertFalse(result.changed)
        self.assertEqual(result.reward, 0)
        with self.assertRaises(ValueError):
            env.set_state(GameState(np.zeros((4, 4), dtype=int), 0, before.rng_state))
        np.testing.assert_array_equal(env.board, before.board)

    def test_random_games_invariants(self):
        chooser = np.random.default_rng(20)
        for seed in range(5):
            env = Game2048(seed=seed)
            for _ in range(2000):
                actions = env.legal_actions()
                if not actions:
                    self.assertTrue(env.terminated)
                    break
                old = env.board
                score = env.score
                result = env.step(actions[int(chooser.integers(len(actions)))])
                self.assertTrue(result.changed)
                self.assertIn(int(result.board.sum() - old.sum()), (2, 4))
                self.assertEqual(result.score, score + result.reward)
            else:
                self.fail("random game did not terminate in 2000 steps")


if __name__ == "__main__":
    unittest.main()
