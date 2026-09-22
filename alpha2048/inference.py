"""Shared model inference and recommendation math; independent of either UI."""
import numpy as np
from .runtime import resource_path


class AIError(RuntimeError):
    def __init__(self, code, detail=''):
        super().__init__(detail or code)
        self.code = code


def recommendations(scores, temperature=.25):
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (4,) or np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError('Invalid action scores')
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be positive')
    legal = np.isfinite(values)
    weights = np.zeros(4)
    if legal.any():
        weights[legal] = np.exp((values[legal] - values[legal].max()) / temperature)
        weights /= weights.sum()
    return weights


def best_action(scores):
    values = np.asarray(scores)
    if not np.isfinite(values).any():
        return None
    return int(np.flatnonzero(values >= values.max() - 1e-6)[0])


def displayed_percentages(scores):
    """Tenths of a percent, summing to 1000 for a nonterminal board."""
    scaled = recommendations(scores) * 1000
    displayed = np.floor(scaled).astype(int)
    if scaled.sum() > 0:
        remainder = 1000 - int(displayed.sum())
        order = np.argsort(-(scaled-displayed), kind='stable')
        displayed[order[:remainder]] += 1
    return displayed


class Inference:
    def __init__(self):
        from threading import Event
        self.engines = {}
        self.cancelled = Event()
        self.configured = False

    def __call__(self, board, device, depth):
        try:
            import torch
        except (ImportError, OSError) as exc:
            raise AIError('runtime_missing') from exc
        from .value_models import load_value_model
        from .value_search import ValueSearch, SearchStopped
        if device not in ('cpu', 'cuda') or depth not in (0, 1, 2):
            raise ValueError('Invalid search configuration')
        if device == 'cuda' and not torch.cuda.is_available():
            raise AIError('gpu_unavailable')
        if not self.configured:
            # Configure once on the persistent worker. Reconfiguring from a new
            # thread per move can accumulate native OpenMP pools on Windows.
            torch.set_num_threads(4)
            torch.backends.mha.set_fastpath_enabled(False)
            self.configured = True
        try:
            if device not in self.engines:
                path = resource_path('models/latest/best.pt')
                # Load/validate on CPU so corrupt files are not reported as GPU failures.
                try:
                    model = load_value_model(path, 'cpu')
                except FileNotFoundError:
                    raise
                except (MemoryError, torch.OutOfMemoryError):
                    raise
                except Exception as exc:
                    if 'DefaultCPUAllocator' in str(exc):
                        raise AIError('cpu_memory') from exc
                    raise AIError('model_invalid') from exc
                self.engines[device] = ValueSearch(model.to(device), stop=self.cancelled.is_set)
            scores, stats = self.engines[device].search(board[None], depth)
            recommendations(scores[0])
            return scores[0], stats['seconds']
        except AIError:
            raise
        except SearchStopped:
            raise
        except FileNotFoundError as exc:
            raise AIError('model_missing') from exc
        except torch.cuda.OutOfMemoryError as exc:
            self.engines.pop(device, None)
            raise AIError('gpu_memory') from exc
        except Exception as exc:
            if isinstance(exc, MemoryError) or 'DefaultCPUAllocator' in str(exc):
                raise AIError('cpu_memory') from exc
            if device == 'cuda':
                self.engines.pop(device, None)
                raise AIError('gpu_error', str(exc)) from exc
            raise
