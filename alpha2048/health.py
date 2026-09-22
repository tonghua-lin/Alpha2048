"""Scale-invariant afterstate health features; no learned values or rewards."""
import math

import numpy as np
import torch

from .core import slide

HEALTH_VERSION = "space-order-compressibility-v1"
VARIANTS = ("legacy", "space", "space-order", "space-merge", "health")
DEFAULT_WEIGHTS = (math.log(17), 2.0, 2.0)


def features(boards, include_order=True, include_merge=True):
    # Local import avoids a cycle when GPUSearch selects this evaluator.
    from .gpu_search import compact

    empty = (boards == 0).sum(dim=1)
    space = torch.log1p(empty.float()) / math.log(17)
    disorder = torch.zeros_like(space)
    compression = torch.zeros_like(space)
    if include_order or include_merge:
        grid = boards.reshape(-1, 4, 4)
        axes = (compact(grid), compact(grid.transpose(1, 2)))
    if include_order:
        for axis in axes:
            lines = axis.float()
            diff = lines[:, :, 1:] - lines[:, :, :-1]
            valid = (lines[:, :, 1:] > 0) & (lines[:, :, :-1] > 0)
            positive = (diff.clamp_min(0) * valid).sum(dim=(1, 2))
            negative = ((-diff).clamp_min(0) * valid).sum(dim=(1, 2))
            total = positive + negative
            # Each axis contributes half of 2*min(P,N)/(P+N).
            disorder += torch.minimum(positive, negative) / total.clamp_min(1)
    if include_merge:
        # In each equal-valued run, either direction merges floor(run_length/2)
        # pairs. Count disjoint pairs in compact rows/columns without building
        # four afterstates or computing score rewards. Invalid moves contribute 0.
        for lines in axes:
            pair01 = (lines[:, :, 0] > 0) & (lines[:, :, 0] == lines[:, :, 1])
            pair12 = (~pair01) & (lines[:, :, 1] > 0) & (lines[:, :, 1] == lines[:, :, 2])
            pair23 = (~pair12) & (lines[:, :, 2] > 0) & (lines[:, :, 2] == lines[:, :, 3])
            counts = (pair01.long() + pair12.long() + pair23.long()).sum(dim=1)
            compression = torch.maximum(compression, counts.float() / 8)
    return space, disorder, compression


def health(boards, variant="health", weights=DEFAULT_WEIGHTS):
    if variant not in VARIANTS[1:]:
        raise ValueError("unknown health variant")
    space, disorder, compression = features(
        boards, include_order=variant in ("space-order", "health"),
        include_merge=variant in ("space-merge", "health"))
    return weights[0] * space - weights[1] * disorder + weights[2] * compression


def reference_features(board):
    """Independent scalar implementation on literal tile values."""
    board = np.asarray(board)
    empty = int((board == 0).sum())
    exponents = np.log2(np.maximum(board, 1))
    disorder = 0.0
    for lines in (exponents, exponents.T):
        positive = negative = 0.0
        for line in lines:
            diff = np.diff(line[line > 0])
            positive += np.maximum(diff, 0).sum()
            negative += np.maximum(-diff, 0).sum()
        total = positive + negative
        disorder += min(positive, negative) / total if total else 0.0
    compression = 0
    for action in range(4):
        after, _, legal = slide(board, action)
        if legal:
            compression = max(compression, int((after == 0).sum()) - empty)
    return math.log1p(empty) / math.log(17), disorder, compression / 8


def reference_health(board, variant="health", weights=DEFAULT_WEIGHTS):
    if variant not in VARIANTS[1:]:
        raise ValueError("unknown health variant")
    space, disorder, compression = reference_features(board)
    return (weights[0] * space
            - (weights[1] * disorder if variant in ("space-order", "health") else 0)
            + (weights[2] * compression if variant in ("space-merge", "health") else 0))
