import time
import tkinter as tk
import unittest
from threading import Event, get_ident
from unittest.mock import patch

import numpy as np
import torch

from alpha2048.advisor import AdvisorWindow, Inference, recommendations, best_action, AIError, SPEEDS
from alpha2048.core import legal_actions
from alpha2048.env import Game2048, GameState
from alpha2048.gpu_search import encode, reference_search
from alpha2048.play import PlayWindow
from alpha2048.value_search import ValueSearch
from alpha2048.value_transformer import ValueTransformer


class RecommendationTests(unittest.TestCase):
    def test_softmax_legal_mask_ties_and_extreme_scores(self):
        q = np.array([-np.inf, 10001., 10001., 9999.])
        p = recommendations(q)
        self.assertEqual(p[0], 0)
        self.assertAlmostEqual(p.sum(), 1)
        self.assertEqual(p[1], p[2])
        self.assertEqual(best_action(q), 1)
        np.testing.assert_allclose(p, recommendations(q - 10000))
        self.assertGreater(p[1], recommendations(q, 1)[1])
        self.assertEqual(recommendations([-np.inf]*4).sum(), 0)
        self.assertIsNone(best_action([-np.inf]*4))
        with self.assertRaises(ValueError):
            recommendations([0, 1, np.nan, 2])

    def test_cpu_search_matches_scalar_reference(self):
        torch.set_num_threads(1)
        torch.manual_seed(2)
        model = ValueTransformer(width=8, layers=1, heads=2, feedforward=16).eval()
        engine = ValueSearch(model, cap=64, inference_batch=7)
        board = np.array([[2, 2, 8, 16], [32, 64, 128, 256],
                          [512, 1024, 2, 4], [8, 16, 32, 0]])
        @torch.inference_mode()
        def evaluate(b):
            return model(torch.as_tensor(encode(b[None]))).item()
        for depth in (0, 1, 2):
            actual, stats = engine.search(board[None], depth)
            np.testing.assert_allclose(actual[0], reference_search(board, depth, evaluate), atol=2e-6)
            self.assertGreater(stats['leaf_evaluations'], 0)

    def test_gpu_unavailable_and_memory_errors(self):
        board = Game2048(seed=1).board
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaises(AIError) as error:
                Inference()(board, 'cuda', 1)
            self.assertEqual(error.exception.code, 'gpu_unavailable')
        with patch('torch.cuda.is_available', return_value=True), patch(
                'alpha2048.value_models.load_value_model', side_effect=torch.cuda.OutOfMemoryError('test')):
            with self.assertRaises(AIError) as error:
                Inference()(board, 'cuda', 2)
            self.assertEqual(error.exception.code, 'gpu_memory')
        with patch('alpha2048.value_models.load_value_model', side_effect=RuntimeError('DefaultCPUAllocator: not enough memory')):
            with self.assertRaises(AIError) as error:
                Inference()(board, 'cpu', 1)
            self.assertEqual(error.exception.code, 'cpu_memory')

    def test_cpu_thread_configuration_runs_once_per_runtime(self):
        provider = Inference()
        board = Game2048(seed=1).board
        with patch('torch.set_num_threads', wraps=torch.set_num_threads) as configure:
            provider(board, 'cpu', 0)
            provider(board, 'cpu', 0)
        configure.assert_called_once_with(4)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_real_depth_two_cpu_gpu_match(self):
        board = np.array([[2, 2, 8, 16], [32, 64, 128, 256],
                          [512, 1024, 2, 4], [8, 16, 32, 0]])
        provider = Inference()
        cpu, _ = provider(board, 'cpu', 2)
        gpu, _ = provider(board, 'cuda', 2)
        np.testing.assert_allclose(cpu, gpu, atol=2e-5)


def fake_advice(board, model, depth):
    scores = np.full(4, -np.inf)
    for action in legal_actions(board):
        scores[action] = float(action)
    return scores, .001


class AdvisorTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = PlayWindow(self.root, Game2048(seed=42))
        self.release = None

    def tearDown(self):
        if self.release is not None:
            self.release.set()
        self.app.close()
        if self.app.ai_executor is not None:
            self.app.ai_executor.shutdown(wait=True, cancel_futures=True)
        self.app = self.root = None
        # Reclaim Tk cycles on the owning thread before the next worker starts.
        import gc
        gc.collect()

    def panel(self, provider=fake_advice):
        panel = AdvisorWindow(self.app, provider)
        self.app.advisor = panel
        return panel

    def until(self, predicate, seconds=4):
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            self.root.update()
            if predicate():
                return
            time.sleep(.01)
        self.fail('AI callback timed out')

    def test_advice_single_step_auto_stop_and_manual_takeover(self):
        panel = self.panel()
        self.until(lambda: panel.scores is not None)
        self.assertEqual(self.app.moves, 0)
        panel.single_step()
        self.until(lambda: self.app.moves == 1 and self.app.animation is None and panel.scores is not None)
        self.assertEqual(self.app.moves, 1)
        self.assertFalse(panel.want_step)
        panel.choose('speed', 'fast')
        panel.start_auto()
        self.until(lambda: self.app.moves >= 3)
        panel.pause()
        count = self.app.moves
        self.until(lambda: self.app.animation is None and panel.scores is not None)
        self.assertEqual(self.app.moves, count)
        panel.start_auto()
        self.app.move(self.app.game.legal_actions()[0])
        self.assertFalse(panel.auto)
        self.app.toggle_language()
        self.assertEqual(panel.window.title(), 'AI 助手')
        panel.close()
        self.assertIsNone(self.app.advisor)

    def test_stale_worker_result_cannot_execute_after_restart(self):
        self.release = Event()
        started = Event()
        def blocked(board, model, depth):
            started.set()
            self.release.wait(3)
            return fake_advice(board, model, depth)
        panel = self.panel(blocked)
        self.until(started.is_set)
        panel.single_step()
        old_key = panel.key()
        self.app.restart()
        self.assertNotEqual(old_key, panel.key())
        self.release.set()
        self.until(lambda: panel.cached_key == panel.key())
        self.assertEqual(self.app.moves, 0)
        self.assertFalse(panel.want_step)

    def test_stop_while_computing_prevents_pending_step(self):
        self.release = Event()
        def blocked(board, model, depth):
            self.release.wait(3)
            return fake_advice(board, model, depth)
        panel = self.panel(blocked)
        panel.single_step()
        panel.pause()
        self.release.set()
        self.until(lambda: panel.scores is not None)
        self.assertEqual(self.app.moves, 0)

    def test_real_latest_model_single_step_on_cpu(self):
        panel = self.panel(Inference())
        panel.single_step()
        self.until(lambda: self.app.moves == 1 and self.app.animation is None and panel.scores is not None, seconds=8)
        self.assertEqual(self.app.moves, 1)
        self.assertGreater(panel.seconds, 0)
        self.assertAlmostEqual(recommendations(panel.scores).sum(), 1)

    def test_error_does_not_loop_and_can_retry(self):
        calls = []
        def fails_once(board, model, depth):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError('test failure')
            return fake_advice(board, model, depth)
        panel = self.panel(fails_once)
        panel.start_auto()
        self.until(lambda: panel.failed_key is not None)
        self.assertFalse(panel.auto)
        self.assertIn('test failure', panel.status.get())
        panel.request_advice()
        self.until(lambda: panel.scores is not None)
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.app.moves, 0)
        weights = [card[2].cget('text') for card in panel.cards]
        self.assertAlmostEqual(sum(float(w[:-1]) for w in weights if w.endswith('%')), 100)

    def test_terminal_stops_auto_and_close_ignores_worker(self):
        panel = self.panel()
        state = self.app.game.get_state()
        board = np.array([[2, 4, 2, 4], [4, 2, 4, 2]]*2)
        self.app.game.set_state(GameState(board, 0, state.rng_state))
        self.app.revision += 1
        panel.start_auto()
        self.until(lambda: not panel.auto)
        self.assertIn('Game over', panel.status.get())
        panel.close()
        self.app.restart()
        self.release = Event()
        def blocked(board, model, depth):
            self.release.wait(3)
            return fake_advice(board, model, depth)
        panel = self.panel(blocked)
        panel.single_step()
        panel.close()
        self.release.set()
        self.until(lambda: not panel.results.empty())
        self.assertEqual(self.app.moves, 0)
        self.assertIsNone(self.app.advisor)

    def test_speed_delay_starts_after_computation_and_animation_matches(self):
        def delayed(board, device, depth):
            time.sleep(.06)
            return fake_advice(board, device, depth)
        panel = self.panel(delayed)
        for mode, (delay, scale) in SPEEDS.items():
            panel.choose('speed', mode)
            panel.request_advice()
            count = self.app.moves
            observed = []
            original = self.app.move
            def record(action, **kwargs):
                observed.append(time.perf_counter() - panel.ready_at)
                return original(action, **kwargs)
            with patch.object(self.app, 'move', side_effect=record):
                panel.single_step()
                self.until(lambda: self.app.moves == count + 1)
            self.assertGreaterEqual(observed[0], delay - .005)
            self.assertLess(observed[0], delay + .25)
            if scale == 0:
                self.assertIsNone(self.app.animation)
            else:
                self.assertEqual(self.app.animation['scale'], scale)
            self.until(lambda: self.app.animation is None)

    def test_device_error_and_cpu_recovery_with_checked_choices(self):
        def provider(board, device, depth):
            if device == 'cuda':
                raise AIError('gpu_unavailable')
            return fake_advice(board, device, depth)
        panel = self.panel(provider)
        panel.choose('depth', 2)
        panel.choose('device', 'cuda')
        panel.start_auto()
        self.until(lambda: panel.failed_key is not None)
        self.assertIn('GPU unavailable', panel.status.get())
        self.assertFalse(panel.auto)
        self.app.toggle_language()
        self.assertIn('GPU 不可用', panel.status.get())
        panel.choose('device', 'cpu')
        self.until(lambda: panel.cached_key == panel.key())
        self.assertIsNone(panel.error)
        self.assertEqual(panel.depth.get(), 2)
        for name, choices in panel.choices.items():
            self.assertEqual(sum(button.cget('text').startswith('✓') for button, _ in choices), 1)

    def test_first_2048_pauses_auto_once_and_allows_resume(self):
        state = self.app.game.get_state()
        board = np.array([[1024, 1024, 0, 0], [0]*4, [0]*4, [0]*4])
        self.app.game.set_state(GameState(board, 0, state.rng_state))
        panel = self.panel()
        panel.choose('speed', 'instant')
        panel.start_auto()
        self.until(lambda: self.app.reached_2048)
        self.assertFalse(panel.auto)
        self.assertFalse(panel.want_step)
        self.assertEqual(self.app.moves, 1)
        confetti_start = self.app.confetti['start']
        panel.single_step()
        self.until(lambda: self.app.moves == 2)
        self.assertEqual(self.app.confetti['start'], confetti_start)
        panel.start_auto()
        self.until(lambda: self.app.moves >= 3)
        self.assertTrue(panel.auto)
        panel.pause()
        self.app.restart()
        self.assertIsNone(self.app.confetti)
        self.assertIsNone(self.app.confetti_job)
        self.assertFalse(self.app.reached_2048)

    def test_recommendations_form_a_cross(self):
        panel = self.panel()
        positions = [(int(card.grid_info()['row']), int(card.grid_info()['column']))
                     for card, _, _ in panel.cards]
        self.assertEqual(positions, [(0, 1), (1, 2), (2, 1), (1, 0)])

    def test_one_worker_is_reused_for_steps_and_panel_reopen(self):
        threads = []
        def record(board, device, depth):
            threads.append(get_ident())
            return fake_advice(board, device, depth)
        panel = self.panel(record)
        executor = self.app.ai_executor
        panel.choose('speed', 'instant')
        panel.start_auto()
        self.until(lambda: self.app.moves >= 12)
        panel.close()
        self.app.restart()
        panel = self.panel(record)
        self.until(lambda: panel.scores is not None)
        self.assertIs(self.app.ai_executor, executor)
        self.assertGreaterEqual(len(threads), 13)
        self.assertEqual(len(set(threads)), 1)


if __name__ == '__main__':
    unittest.main()
