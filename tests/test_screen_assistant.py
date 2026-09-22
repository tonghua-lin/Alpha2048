import time
import tkinter as tk
import unittest
from pathlib import Path
from threading import Event, get_ident
from unittest.mock import patch
import numpy as np
from PIL import Image

from alpha2048.screen_ocr import BoardReader, RecognitionError
from alpha2048.screen_assistant import ScreenEngine, ScreenAssistant, overlaps, absolute_geometry

FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'play2048-board.png'
EXPECTED = np.array([[0,0,0,0],[0,0,0,2],[2,8,16,4],[4,2,4,64]])


class Provider:
    def __init__(self):
        self.cancelled = Event()
        self.calls = []

    def __call__(self,board,device,depth):
        self.calls.append((get_ident(),device,depth))
        return np.array([1.,2.,3.,-np.inf]),.01


class Reader:
    def read(self,image):
        return EXPECTED.copy()


class ScreenTests(unittest.TestCase):
    def test_compact_window_keeps_status_and_recommendations_visible(self):
        root = tk.Tk()
        with patch.object(root,'winfo_screenheight',return_value=768):
            app = ScreenAssistant(root)
        try:
            for language in ('en','zh'):
                app.language = language
                app.render()
                root.update()
                self.assertLessEqual(root.winfo_height(),668)
                self.assertGreaterEqual(app.status.winfo_height(),app.status.winfo_reqheight())
                for _,_,value in app.cards:
                    self.assertTrue(value.winfo_ismapped())
                    self.assertGreaterEqual(value.winfo_height(),value.winfo_reqheight())
        finally:
            app.close()
            del app,root
            import gc
            gc.collect()

    def test_invalid_scores_do_not_enter_cache(self):
        class InvalidProvider(Provider):
            def __call__(self,*args):
                return np.array([np.nan,0,0,0]),0
        provider = InvalidProvider()
        engine = ScreenEngine(lambda box:Image.new('RGB',(200,200),'white'),Reader(),provider)
        with self.assertRaises(ValueError):
            engine.analyze((0,0,200,200),'cpu',0)
        self.assertIsNone(engine.cache)

    def test_closing_during_work_does_not_leave_tk_owned_by_worker(self):
        import gc
        import weakref
        class SlowEngine:
            def __init__(self):
                self.cancelled,self.started,self.release = Event(),Event(),Event()
            def analyze(self,*args):
                self.started.set()
                self.release.wait(2)
                return 'cancelled',None,None
        root = tk.Tk()
        root.withdraw()
        engine = SlowEngine()
        app = ScreenAssistant(root,engine)
        app.box = (10000,0,10200,200)
        executor = app.executor
        ref = weakref.ref(app)
        try:
            app.submit()
            self.assertTrue(engine.started.wait(1))
            app.close()
            del app
            gc.collect()
            self.assertIsNone(ref())
        finally:
            engine.release.set()
            executor.shutdown(wait=True,cancel_futures=True)
            try:
                root.destroy()
            except tk.TclError:
                pass
            del root
            gc.collect()

    def test_selection_starts_polling_recovers_and_stop_halts(self):
        class Engine:
            cancelled = None
            def __init__(self):
                self.cancelled = Event()
                self.calls = 0
            def analyze(self,*args):
                self.calls += 1
                if self.calls == 1:
                    return 'waiting',None,None
                if self.calls == 2:
                    raise RecognitionError('animation')
                board = EXPECTED.copy()
                board[0,0] = 2 ** (self.calls-2)
                return 'done',board,np.array([1.,2.,3.,-np.inf])
        root = tk.Tk()
        root.withdraw()
        engine = Engine()
        app = ScreenAssistant(root,engine)
        try:
            app.selected((10000,0,10200,200))
            root.withdraw()
            self.assertTrue(app.live)
            deadline = time.perf_counter()+3
            while (app.board is None or app.board[0,0] != 4) and time.perf_counter()<deadline:
                root.update()
                time.sleep(.01)
            self.assertIsNotNone(app.board)
            self.assertEqual(app.board[0,0],4)
            self.assertTrue(app.live)
            app.stop()
            calls = engine.calls
            deadline = time.perf_counter()+.65
            while time.perf_counter()<deadline:
                root.update()
                time.sleep(.01)
            self.assertEqual(engine.calls,calls)
            self.assertIsNone(app.scores)
        finally:
            app.close()
            app.executor.shutdown(wait=True,cancel_futures=True)
            del app,root
            import gc
            gc.collect()

    def test_negative_monitor_origin_is_absolute(self):
        root = tk.Tk()
        root.withdraw()
        window = tk.Toplevel(root)
        window.overrideredirect(True)
        try:
            for x,y in ((0,-267),(-320,0),(-320,-267),(10,20)):
                window.geometry(absolute_geometry(x,y,200,200))
                root.update()
                self.assertEqual((window.winfo_rootx(),window.winfo_rooty()),(x,y))
        finally:
            root.destroy()
            del window,root
            import gc
            gc.collect()

    def test_real_ocr_at_multiple_sizes_and_invalid_input(self):
        reader = BoardReader()
        image = Image.open(FIXTURE)
        for size in (320,616,900):
            np.testing.assert_array_equal(reader.read(image.resize((size,size))),EXPECTED)
        for image in (Image.new('RGB',(400,400),'white'),Image.new('RGB',(400,200),'white')):
            with self.assertRaises(RecognitionError):
                reader.read(image)

    def test_changed_during_search_is_rejected_and_cache_keys_include_settings(self):
        first = Image.new('RGB',(200,200),'white')
        second = Image.new('RGB',(200,200),'black')
        captures = iter([first,first,second])
        provider = Provider()
        engine = ScreenEngine(lambda box:next(captures),Reader(),provider)
        self.assertEqual(engine.analyze((0,0,200,200),'cpu',1)[0],'waiting')
        engine.capture = lambda box:first
        self.assertEqual(engine.analyze((0,0,200,200),'cpu',1)[0],'done')
        self.assertEqual(len(provider.calls),1)
        engine.analyze((0,0,200,200),'cpu',2)
        self.assertEqual(len(provider.calls),2)

    def test_animation_and_cancel_skip_inference(self):
        captures = iter([Image.new('RGB',(200,200),'white'),Image.new('RGB',(200,200),'black')])
        provider = Provider()
        engine = ScreenEngine(lambda box:next(captures),Reader(),provider)
        self.assertEqual(engine.analyze((0,0,200,200),'cpu',1)[0],'waiting')
        self.assertFalse(provider.calls)
        engine.capture = lambda box:Image.new('RGB',(200,200))
        provider.cancelled.set()
        self.assertEqual(engine.analyze((0,0,200,200),'cpu',1)[0],'cancelled')
        self.assertFalse(provider.calls)

    def test_ui_stale_result_stop_language_and_worker_reuse(self):
        root = tk.Tk()
        root.withdraw()
        provider = Provider()
        engine = ScreenEngine(lambda box:Image.new('RGB',(200,200),'white'),Reader(),provider)
        app = ScreenAssistant(root,engine)
        app.box = (10000,0,10200,200)
        try:
            app.refresh()
            app.submit()
            app.stop()
            deadline = time.perf_counter()+2
            while app.inflight and time.perf_counter()<deadline:
                root.update()
                time.sleep(.01)
            self.assertIsNone(app.scores)
            self.assertFalse(app.live)
            for depth in (0,2):
                app.depth.set(depth)
                app.refresh()
                deadline = time.perf_counter()+2
                while app.scores is None and time.perf_counter()<deadline:
                    root.update()
                    time.sleep(.01)
                self.assertIsNotNone(app.scores)
            self.assertEqual(len(set(call[0] for call in provider.calls)),1)
            app.toggle_language()
            self.assertEqual(app.select_button.cget('text'),'框选棋盘')
            self.assertTrue(overlaps((0,0,200,200),(100,100,300,300)))
            self.assertFalse(overlaps((0,0,100,100),(100,100,200,200)))
        finally:
            app.close()
            app.executor.shutdown(wait=True,cancel_futures=True)
            del app,root
            import gc
            gc.collect()


if __name__ == '__main__':
    unittest.main()
