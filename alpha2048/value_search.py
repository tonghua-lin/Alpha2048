"""Expectimax with a learned afterstate value at the search frontier."""
import torch

from .gpu_search import GPUSearch


class SearchStopped(Exception):
    pass


class ValueSearch(GPUSearch):
    def __init__(self, model, cap=4096, inference_batch=512, stop=None):
        super().__init__(cap=cap, device=next(model.parameters()).device)
        if inference_batch < 1:
            raise ValueError('inference_batch must be positive')
        self.model = model.eval()
        self.inference_batch = inference_batch
        self.stop = stop or (lambda: False)

    def afterstates(self, boards, depth):
        if self.stop():
            raise SearchStopped()
        if depth:
            return super().afterstates(boards, depth)
        output = torch.empty(len(boards), device=self.device)
        for start in range(0, len(boards), self.inference_batch):
            if self.stop():
                raise SearchStopped()
            batch = boards[start:start + self.inference_batch]
            values = self.model(batch)
            if not torch.isfinite(values).all():
                raise ValueError('Model produced nonfinite leaf values')
            output[start:start + len(batch)] = values
        self.stats['leaf_evaluations'] += len(boards)
        return output
