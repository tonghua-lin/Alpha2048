"""Local, recognition-only OCR for a tightly selected 4x4 board."""
import numpy as np
from PIL import Image, ImageOps

from .runtime import resource_path
from .inference import AIError

ASSETS = resource_path('assets/ocr')


class RecognitionError(ValueError):
    pass


def fingerprint(image):
    return np.asarray(image.convert('RGB').resize((128, 128)), dtype=np.int16)


def same_frame(first, second):
    if first is None or second is None:
        return False
    delta = np.abs(first - second)
    return float(delta.mean()) < 1.0 and float(np.mean(delta > 15)) < .005


class BoardReader:
    def __init__(self):
        try:
            import onnxruntime as ort
        except (ImportError, OSError) as exc:
            raise AIError('runtime_missing') from exc
        if not (ASSETS / 'en_PP-OCRv4_rec.onnx').is_file() or not (ASSETS / 'dict.txt').is_file():
            raise AIError('ocr_missing')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        try:
            self.session = ort.InferenceSession(str(ASSETS / 'en_PP-OCRv4_rec.onnx'), options,
                                                providers=['CPUExecutionProvider'])
            self.chars = ['blank'] + (ASSETS / 'dict.txt').read_text(encoding='utf-8').splitlines() + [' ']
            if self.session.get_outputs()[0].shape[-1] != len(self.chars):
                raise ValueError('OCR dictionary mismatch')
        except Exception as exc:
            raise AIError('ocr_invalid') from exc

    def read(self, image):
        width, height = image.size
        if min(width, height) < 160 or abs(width / height - 1) > .12:
            raise RecognitionError('Select the full square board')
        board = np.zeros((4, 4), dtype=np.int64)
        batch, positions = [], []
        for row in range(4):
            for col in range(4):
                # Inset avoids rounded corners, grid gutters and tile shadows.
                box = ((col+.12)*width/4, (row+.12)*height/4,
                       (col+.88)*width/4, (row+.88)*height/4)
                tile = image.crop(tuple(round(v) for v in box)).convert('L')
                gray = np.asarray(tile, dtype=np.float32)
                border = max(2, round(tile.height*.07))
                background = np.median(np.concatenate((gray[:border].ravel(), gray[-border:].ravel())))
                mask = np.abs(gray-background) > 30
                if np.percentile(gray,99)-np.percentile(gray,1) < 18 and mask.sum() < 3:
                    continue
                # Small fragments or low contrast are unknown, never empty.
                if mask.sum() < max(12, gray.size*.002):
                    raise RecognitionError(f'Unclear cell {row+1},{col+1}')
                ys, xs = np.where(mask)
                if xs.min() < 2 or ys.min() < 2 or xs.max() >= tile.width-2 or ys.max() >= tile.height-2:
                    raise RecognitionError(f'Clipped cell {row+1},{col+1}')
                pad = max(3, round(tile.height*.07))
                tile = tile.crop((max(0,int(xs.min())-pad), max(0,int(ys.min())-pad),
                                  min(tile.width,int(xs.max())+pad+1), min(tile.height,int(ys.max())+pad+1)))
                if np.median(gray[mask]) > background:
                    tile = ImageOps.invert(tile)
                target_width = min(320, int(np.ceil(48*tile.width/tile.height)))
                arr = np.asarray(tile.resize((target_width,48), Image.Resampling.BILINEAR), dtype=np.float32)
                batch.append(np.repeat(arr[None],3,axis=0)/127.5-1)
                positions.append((row,col))
        if not batch:
            raise RecognitionError('No tiles found')
        padded = np.zeros((len(batch),3,48,max(48,max(a.shape[2] for a in batch))), dtype=np.float32)
        for i, arr in enumerate(batch):
            padded[i,:,:,:arr.shape[2]] = arr
        predictions = self.session.run(None,{self.session.get_inputs()[0].name:padded})[0]
        if predictions.shape[-1] != len(self.chars):
            raise RecognitionError('OCR dictionary mismatch')
        for (row,col), prediction in zip(positions,predictions):
            ids = prediction.argmax(axis=1)
            kept = [i for i,t in enumerate(ids) if t and (i==0 or t!=ids[i-1])]
            text = ''.join(self.chars[ids[i]] for i in kept)
            confidence = min((float(prediction[i,ids[i]]) for i in kept), default=0)
            value = int(text) if text.isascii() and text.isdigit() else 0
            if confidence < .90 or value < 2 or value > 131072 or value & (value-1):
                raise RecognitionError(f'Uncertain cell {row+1},{col+1}: {text!r}')
            board[row,col] = value
        return board
