"""Bounded playback check using a hidden worker; training checks are archived."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import tkinter as tk
import unittest
from uuid import uuid4
import numpy as np
import torch
from alpha2048.desktop import Workspace, PROJECT
from alpha2048.run_io import read_json
from alpha2048.value_models import build_value_model

class DesktopTests(unittest.TestCase):

    def setUp(self):
        self.directory = PROJECT / '.tmp' / ('desktop-test-' + uuid4().hex)
        self.directory.mkdir(parents=True)
        self.root = tk.Tk()
        self.app = None

    def tearDown(self):
        if self.app and self.app.alive():
            self.app.process.terminate()
            self.app.process.wait(timeout=10)
        if self.app and self.app.log_handle:
            self.app.log_handle.close()
        try:
            if self.app:
                self.root.after_cancel(self.app.poll_id)
            self.root.destroy()
        except tk.TclError:
            pass
        assert self.directory.resolve().parent == (PROJECT / '.tmp').resolve()
        shutil.rmtree(self.directory)

    def until(self, predicate, seconds=18):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.root.update()
            if predicate():
                return
            time.sleep(0.03)
        self.fail('Desktop worker timeout: ' + self.app.status.get())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_search_single_step_pause_stop(self):
        self.app = Workspace(self.root, 'search', self.directory)
        model = build_value_model('transformer')
        checkpoint = self.directory / 'transformer.pt'
        torch.save(dict(model_type='transformer', model_config=model.config, model=model.state_dict()), checkpoint)
        self.app.checkpoint.set(str(checkpoint))
        self.app.single_step()
        self.until(lambda: read_json(self.app.output / 'status.json', {}).get('status') == 'paused' and read_json(self.app.output / 'status.json', {}).get('step') == 1)
        first = read_json(self.app.output / 'status.json')
        self.assertEqual(sum((q is not None for q in first['q'])), 4)
        self.app.single_step()
        self.until(lambda: read_json(self.app.output / 'status.json', {}).get('step') == 2)
        self.app.stop()
        self.until(lambda: self.app.process is None)
        self.assertEqual(read_json(self.app.output / 'status.json')['status'], 'stopped')
        result = read_json(self.app.output / 'results.json')[0]
        self.assertEqual(result['steps'], 2)
        self.assertFalse(result['terminated'])
        self.assertEqual(read_json(self.app.output / 'run.json')['model_type'], 'transformer')
if __name__ == '__main__':
    unittest.main()
