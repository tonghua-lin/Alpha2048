"""Shared executable entry point; filename selects the product window."""
import json
from pathlib import Path
import sys
import traceback


def self_test(output):
    result = {'ok': False}
    try:
        import numpy as np
        import torch
        from alpha2048.inference import Inference, AIError
        from alpha2048.screen_ocr import BoardReader
        from alpha2048.runtime import resource_path
        assert torch.version.cuda is None, 'Release must use CPU-only Torch'
        board = np.array([[2,2,4,8],[0,4,8,16],[0,0,16,32],[0,0,0,64]],dtype=np.int64)
        scores,_ = Inference()(board,'cpu',1)
        assert np.isfinite(scores).any()
        try:
            Inference()(board,'cuda',0)
        except AIError as exc:
            assert exc.code == 'gpu_unavailable'
        else:
            raise AssertionError('GPU incorrectly available in CPU release')
        reader = BoardReader()
        prediction = reader.session.run(None,{reader.session.get_inputs()[0].name:np.zeros((1,3,48,96),dtype=np.float32)})[0]
        assert prediction.shape[-1] == len(reader.chars)
        import tkinter as tk
        from alpha2048.play import PlayWindow
        from alpha2048.screen_assistant import ScreenAssistant
        import gc
        for factory in (PlayWindow, ScreenAssistant):
            root = tk.Tk()
            root.withdraw()
            app = factory(root)
            root.update_idletasks()
            app.close()
            del app,root
            gc.collect()
        result.update(ok=True,torch=torch.__version__,scores=scores.tolist(),
                      resources=str(resource_path('')),ocr_output_shape=list(prediction.shape))
    except Exception:
        result['error'] = traceback.format_exc()
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return 0 if result['ok'] else 1


def main():
    if len(sys.argv)==3 and sys.argv[1]=='--self-test':
        return self_test(Path(sys.argv[2]).resolve())
    try:
        executable = Path(sys.executable).stem.lower().replace(' ', '').replace('-', '')
        if 'screenassistant' in executable or '--screen' in sys.argv:
            from alpha2048.screen_assistant import main as start
        else:
            from alpha2048.play import main as start
        start()
        return 0
    except Exception:
        import logging
        from alpha2048.runtime import configure_logging
        log = configure_logging('startup')
        logging.exception('Application startup failed')
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror('2048',f'Unable to start. Please extract the complete ZIP and try again.\n\nLog: {log}',parent=root)
        root.destroy()
        return 1


if __name__=='__main__':
    raise SystemExit(main())
