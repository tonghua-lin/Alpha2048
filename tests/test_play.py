import tkinter as tk
import unittest
from time import perf_counter, sleep
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from alpha2048.env import Game2048, GameState
from alpha2048.play import PlayWindow, tile_paths
from alpha2048.core import slide, Action


class PlayTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.game = Game2048(seed=7)
        self.app = PlayWindow(self.root, self.game)

    def tearDown(self):
        self.app.close()
        self.app = self.root = None
        import gc
        gc.collect()

    def board(self, values, score=0):
        self.app.cancel_animation()
        state = self.game.get_state()
        self.game.set_state(GameState(np.array(values), score, state.rng_state))

    def test_keyboard_merge_invalid_move_and_restart(self):
        self.board([[2, 2, 0, 0], [0]*4, [0]*4, [0]*4])
        before = self.game.get_state()
        self.app.on_key(SimpleNamespace(keysym='Up'))
        np.testing.assert_array_equal(self.game.board, before.board)
        self.assertEqual(self.game.get_state().rng_state, before.rng_state)
        self.assertEqual(self.app.moves, 0)
        self.app.on_key(SimpleNamespace(keysym='A'))
        self.assertEqual(self.game.score, 4)
        self.assertEqual(self.game.board[0, 0], 4)
        self.assertEqual(np.count_nonzero(self.game.board), 2)
        self.assertEqual(self.app.score_text.get(), '4')
        self.app.on_key(SimpleNamespace(keysym='r'))
        self.assertEqual(self.game.score, 0)
        self.assertEqual(self.app.moves, 0)
        self.assertEqual(self.app.best_text.get(), '4')
        self.assertEqual(np.count_nonzero(self.game.board), 2)

    def test_2048_continues_and_terminal_can_restart(self):
        self.board([[1024, 1024, 0, 0], [0]*4, [0]*4, [0]*4])
        self.app.on_key(SimpleNamespace(keysym='Left'))
        self.assertTrue(self.app.reached_2048)
        self.assertIn('2048 reached', self.app.status.get())
        self.assertFalse(self.game.terminated)
        self.board([[2, 4, 2, 4], [4, 2, 4, 2]]*2, 2048)
        self.app.on_key(SimpleNamespace(keysym='Down'))
        self.assertIn('No moves remain', self.app.status.get())
        self.assertTrue(self.game.terminated)
        self.app.new_button.invoke()
        self.assertFalse(self.game.terminated)
        self.assertFalse(self.app.reached_2048)

    def finish_animation(self):
        deadline = perf_counter() + 2
        while self.app.animation is not None and perf_counter() < deadline:
            self.root.update()
            sleep(.01)
        self.assertIsNone(self.app.animation)

    def test_animation_buffer_and_restart(self):
        self.board([[2, 2, 0, 0], [0]*4, [0]*4, [0]*4])
        expected = self.game.clone()
        expected.step(Action.LEFT)
        expected.step(Action.DOWN)
        self.app.move(Action.LEFT)
        self.assertIsNotNone(self.app.animation)
        self.app.move(Action.DOWN)
        self.finish_animation()
        np.testing.assert_array_equal(self.game.board, expected.board)
        self.assertEqual(self.game.score, expected.score)
        self.app.move(self.game.legal_actions()[0])
        self.app.restart()
        self.assertIsNone(self.app.animation_job)
        self.assertIsNone(self.app.pending_action)
        self.assertEqual(self.game.score, 0)
        self.assertEqual(np.count_nonzero(self.game.board), 2)

    def test_language_and_help_preserve_board(self):
        before = self.game.get_state()
        self.assertEqual(self.app.score_label.cget('text'), 'SCORE')
        self.app.toggle_language()
        self.assertEqual(self.app.score_label.cget('text'), '得分')
        self.app.show_help()
        self.assertIn('方向键', self.app.help_body.cget('text'))
        self.app.toggle_language()
        self.assertIn('arrow keys', self.app.help_body.cget('text'))
        self.app.close_help()
        np.testing.assert_array_equal(self.game.board, before.board)
        self.assertEqual(self.game.get_state().rng_state, before.rng_state)

    def test_invalid_move_shakes_without_changing_state(self):
        self.board([[2, 4, 0, 0], [0]*4, [0]*4, [0]*4])
        before = self.game.get_state()
        self.app.draw()
        resting = self.app.canvas.coords(self.app.canvas.find_all()[0])
        self.app.move(Action.LEFT)
        self.assertIsNotNone(self.app.shake_job)
        with patch('alpha2048.play.perf_counter', return_value=self.app.shake['start'] + .02):
            self.app.draw()
        shifted = self.app.canvas.coords(self.app.canvas.find_all()[0])
        self.assertNotEqual(resting[0], shifted[0])
        self.assertEqual(resting[1], shifted[1])
        np.testing.assert_array_equal(self.game.board, before.board)
        self.assertEqual(self.game.get_state().rng_state, before.rng_state)
        self.assertEqual(self.game.score, before.score)
        self.assertEqual(self.app.moves, 0)
        # A valid input immediately interrupts the shake.
        self.app.move(Action.DOWN)
        self.assertIsNone(self.app.shake_job)
        self.assertIsNotNone(self.app.animation)
        self.app.restart()
        self.app.move(Action.UP)
        self.app.restart()
        self.assertIsNone(self.app.shake_job)
        self.assertIsNone(self.app.animation_job)

    def test_confetti_is_cosmetic_and_game_over_is_below_board(self):
        self.root.deiconify()
        self.root.update()
        before = self.game.get_state()
        self.app.start_confetti()
        with patch('alpha2048.play.perf_counter', return_value=self.app.confetti['start'] + 1):
            self.app.draw()
        self.assertTrue(self.app.canvas.find_withtag('confetti'))
        np.testing.assert_array_equal(self.game.board, before.board)
        self.assertEqual(self.game.get_state().rng_state, before.rng_state)
        start = self.app.confetti['start']
        self.root.after_cancel(self.app.confetti_job)
        with patch('alpha2048.play.perf_counter', return_value=start + 3):
            self.app.confetti_tick()
        self.assertIsNone(self.app.confetti)
        self.assertIsNone(self.app.confetti_job)
        self.board([[2, 4, 2, 4], [4, 2, 4, 2]]*2)
        self.app.draw()
        footer = self.app.canvas.find_withtag('game-over')[0]
        board_rect = self.app.canvas.coords(self.app.canvas.find_all()[0])
        self.assertGreater(self.app.canvas.bbox(footer)[1], board_rect[3])
        self.assertEqual(self.app.canvas.itemcget(footer, 'text'), 'Game over')
        self.app.toggle_language()
        footer = self.app.canvas.find_withtag('game-over')[0]
        self.assertEqual(self.app.canvas.itemcget(footer, 'text'), '游戏结束')
        self.app.restart()
        self.assertFalse(self.app.canvas.find_withtag('game-over'))


class AnimationPathsTests(unittest.TestCase):
    def test_tile_destinations_match_rules_in_all_directions(self):
        rng = np.random.default_rng(2048)
        boards = [np.array([[2, 2, 2, 2], [4, 0, 4, 4], [8, 8, 16, 0], [0]*4])]
        boards += [np.where(z > 0, 2**z, 0) for z in rng.integers(0, 6, (32, 4, 4))]
        for board in boards:
            for action in Action:
                paths, merged = tile_paths(board, action)
                actual = np.zeros((4, 4), dtype=np.int64)
                sources = set()
                for value, source, destination in paths:
                    self.assertNotIn(source, sources)
                    sources.add(source)
                    self.assertEqual(value, board[source])
                    actual[destination] += value
                expected, reward, _ = slide(board, action)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(len(sources), np.count_nonzero(board))
                self.assertEqual(sum(int(actual[pos]) for pos in merged), reward)


if __name__ == '__main__':
    unittest.main()
