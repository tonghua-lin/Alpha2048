import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from alpha2048.runtime import resource_path, configure_logging
from alpha2048.inference import Inference, AIError, displayed_percentages
from alpha2048.screen_ocr import BoardReader


class RuntimeTests(unittest.TestCase):
    def test_assets_resolve_from_source_or_bundle_not_working_directory(self):
        self.assertTrue(resource_path('models/latest/best.pt').is_file())
        with patch.object(sys,'_MEIPASS',str(Path('bundle-root').resolve()),create=True):
            self.assertEqual(resource_path('assets/ocr/dict.txt'),Path('bundle-root').resolve()/'assets/ocr/dict.txt')

    def test_unwritable_logs_do_not_prevent_startup(self):
        with patch('pathlib.Path.mkdir',side_effect=PermissionError('read-only')):
            self.assertIsNone(configure_logging('test'))

    def test_missing_and_corrupt_ocr_are_distinct(self):
        with patch('alpha2048.screen_ocr.ASSETS',Path('absent-ocr-test')):
            with self.assertRaises(AIError) as error:
                BoardReader()
            self.assertEqual(error.exception.code,'ocr_missing')
        with patch('onnxruntime.InferenceSession',side_effect=ValueError('broken protobuf')):
            with self.assertRaises(AIError) as error:
                BoardReader()
            self.assertEqual(error.exception.code,'ocr_invalid')

    def test_missing_and_corrupt_game_models_are_distinct(self):
        board = np.zeros((4,4),dtype=np.int64)
        for failure,code in ((FileNotFoundError(),'model_missing'),(ValueError('bad checkpoint'),'model_invalid')):
            with patch('alpha2048.value_models.load_value_model',side_effect=failure):
                with self.assertRaises(AIError) as error:
                    Inference()(board,'cpu',0)
                self.assertEqual(error.exception.code,code)

    def test_percentage_rounding_including_terminal(self):
        values = displayed_percentages([0,0,0,-np.inf])
        self.assertEqual(values.sum(),1000)
        self.assertEqual(values[3],0)
        self.assertEqual(displayed_percentages([-np.inf]*4).sum(),0)


if __name__ == '__main__':
    unittest.main()
