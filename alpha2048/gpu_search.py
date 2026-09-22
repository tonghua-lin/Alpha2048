"""Exact finite-depth chance expansion, alpha=0, bounded batch intermediates.

Depth counts spawn+decision pairs AFTER the candidate root action. No CNN.
The batch cap bounds each expansion's children, not total allocator memory:
ancestor frames and temporary tensors also occupy memory (O(depth * cap)).
"""
from contextlib import contextmanager
from time import perf_counter

import numpy as np
import torch

from .core import slide

TERMINAL = -10.0
HEURISTIC_VERSION = "log-empty-minus-normalized-corner-inversions-v1"


def encode(boards):
    a = np.asarray(boards)
    if a.ndim != 3 or a.shape[1:] != (4, 4) or a.dtype.kind not in "iu":
        raise ValueError("expected integer [N,4,4] literal boards")
    if np.any(a < 0) or np.any((a != 0) & ((a < 2) | ((a & (a - 1)) != 0))):
        raise ValueError("tiles must be zero or powers of two >= 2")
    return np.where(a > 0, np.log2(np.maximum(a, 1)), 0).astype(np.uint8).reshape(-1, 16)


def compact(rows):
    positions = torch.arange(4, device=rows.device)
    order = ((rows == 0).to(torch.int64) * 4 + positions).argsort(dim=-1)
    return rows.gather(-1, order)


def moves(boards):
    """Return [N,4,16] boards, [N,4] legal mask and merge rewards."""
    device = boards.device
    grid = np.arange(16).reshape(4, 4)
    idx = torch.tensor(np.stack([np.rot90(grid, k).reshape(-1) for k in (1, 2, 3, 0)]).copy(), device=device)
    rows = compact(boards[:, idx].reshape(-1, 4))
    result = rows.clone()
    consumed = torch.zeros(len(rows), dtype=torch.bool, device=device)
    reward = torch.zeros(len(rows), dtype=torch.int64, device=device)
    for i in range(3):
        merge = (~consumed) & (rows[:, i] != 0) & (rows[:, i] == rows[:, i + 1])
        result[:, i] = torch.where(merge, rows[:, i] + 1, result[:, i])
        result[:, i + 1] = torch.where(merge, 0, result[:, i + 1])
        reward += torch.where(merge, torch.bitwise_left_shift(torch.ones_like(reward), rows[:, i].long() + 1), 0)
        consumed = merge
    result = compact(result).reshape(-1, 4, 16)
    result = result.gather(2, idx.argsort(dim=1)[None].expand(len(boards), -1, -1))
    legal = (result != boards[:, None]).any(dim=2)
    return result, legal, reward.reshape(-1, 4, 4).sum(dim=2)


def heuristic(boards):
    """log(1+empties) minus best-corner exponent inversions / exponent mass.

    Empty cells are removed before comparing neighbors. Penalty is <= 2,
    so TERMINAL=-10 is strictly below every nonterminal leaf score.
    """
    b = boards.reshape(-1, 4, 4)
    row = compact(b).float()
    col = compact(b.transpose(1, 2)).float()
    penalty = torch.zeros(len(b), device=b.device)
    for lines in (row, col):
        diff = lines[:, :, 1:] - lines[:, :, :-1]
        valid = (lines[:, :, 1:] > 0) & (lines[:, :, :-1] > 0)
        up = (diff.clamp_min(0) * valid).sum(dim=(1, 2))
        down = ((-diff).clamp_min(0) * valid).sum(dim=(1, 2))
        penalty += torch.minimum(up, down)
    return torch.log1p((boards == 0).sum(dim=1).float()) - penalty / boards.float().sum(dim=1).clamp_min(1)


class GPUSearch:
    def __init__(self, cap=65536, device="cuda", profile=False,
                 heuristic_name="legacy", weights=None):
        from .health import DEFAULT_WEIGHTS, VARIANTS
        if heuristic_name not in VARIANTS:
            raise ValueError("unknown heuristic")
        weights = DEFAULT_WEIGHTS if weights is None else tuple(weights)
        if len(weights) != 3 or not all(np.isfinite(w) and w >= 0 for w in weights):
            raise ValueError("health weights must be three finite nonnegative numbers")
        if heuristic_name != "legacy" and weights[1] >= -TERMINAL:
            raise ValueError("order weight must keep nonterminal health above TERMINAL")
        self.heuristic_name, self.weights = heuristic_name, weights
        if cap < 32:
            raise ValueError("cap must be >= 32")
        self.cap, self.device, self.profile = cap, torch.device(device), profile
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Search requires CPU or CUDA")
        self.stats = {}

    @contextmanager
    def stage(self, name):
        if self.profile:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            start = perf_counter()
        yield
        if self.profile:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.stats[name + "_seconds"] = self.stats.get(name + "_seconds", 0) + perf_counter() - start

    def action_values(self, boards, remaining):
        with self.stage("move"):
            children, legal, _ = moves(boards)
            pairs = legal.nonzero()
            active = children[pairs[:, 0], pairs[:, 1]]
        self.stats["move_inputs"] += len(boards)
        self.stats["action_children"] += len(active)
        del children
        values = self.afterstates(active, remaining)
        with self.stage("max_reduce"):
            q = torch.full((len(boards), 4), -torch.inf, device=self.device)
            q[pairs[:, 0], pairs[:, 1]] = values
        return q

    def decisions(self, boards, remaining):
        output = torch.empty(len(boards), device=self.device)
        for start in range(0, len(boards), max(1, self.cap // 4)):
            chunk = boards[start:start + self.cap // 4]
            q = self.action_values(chunk, remaining)
            with self.stage("max_reduce"):
                value = q.max(dim=1).values
                output[start:start + len(chunk)] = torch.where(torch.isfinite(value), value, TERMINAL)
        return output

    def afterstates(self, boards, depth):
        if depth == 0:
            with self.stage("heuristic"):
                if self.heuristic_name == "legacy":
                    values = heuristic(boards)
                else:
                    from .health import health
                    values = health(boards, self.heuristic_name, self.weights)
            self.stats["leaf_evaluations"] += len(boards)
            return values
        output = torch.empty(len(boards), device=self.device)
        for start in range(0, len(boards), self.cap // 32):
            chunk = boards[start:start + self.cap // 32]
            with self.stage("chance"):
                empty = chunk == 0
                pairs = empty.nonzero().repeat_interleave(2, dim=0)
                parent, cell = pairs[:, 0], pairs[:, 1]
                tile = torch.arange(len(pairs), device=self.device) % 2 + 1
                children = chunk[parent].clone()
                children[torch.arange(len(pairs), device=self.device), cell] = tile.to(torch.uint8)
                probabilities = torch.where(tile == 1, 0.9, 0.1) / empty.sum(dim=1)[parent]
            self.stats["chance_children"] += len(children)
            self.stats["chance_batches"] += 1
            values = self.decisions(children, depth - 1)
            with self.stage("chance_reduce"):
                total = torch.zeros(len(chunk), device=self.device)
                total.scatter_add_(0, parent, probabilities * values)
                output[start:start + len(chunk)] = total
        return output

    @torch.inference_mode()
    def search(self, literal_boards, depth=2):
        if depth < 0:
            raise ValueError("depth must be >= 0")
        self.stats = dict(move_inputs=0, action_children=0, chance_children=0, leaf_evaluations=0, chance_batches=0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        start = perf_counter()
        boards = torch.as_tensor(encode(literal_boards), device=self.device)
        result = []
        for i in range(0, len(boards), self.cap // 4):
            result.append(self.action_values(boards[i:i + self.cap // 4], depth))
        q = torch.cat(result).cpu().numpy() if result else np.empty((0, 4), dtype=np.float32)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.stats.update(seconds=perf_counter() - start,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0,
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else 0)
        return q, dict(self.stats)


def reference_heuristic(board):
    z = encode(np.asarray(board)[None])[0].reshape(4, 4).astype(float)
    penalty = 0.0
    for lines in (z, z.T):
        up = down = 0.0
        for line in lines:
            diff = np.diff(line[line > 0])
            up += np.maximum(diff, 0).sum()
            down += np.maximum(-diff, 0).sum()
        penalty += min(up, down)
    return np.log1p((z == 0).sum()) - penalty / max(1, z.sum())


def reference_search(board, depth, evaluator=None):
    """Independent scalar CPU oracle; intended for small correctness cases."""
    def after(x, remaining):
        if remaining == 0:
            return (reference_heuristic if evaluator is None else evaluator)(x)
        cells = np.argwhere(x == 0)
        value = 0.0
        for row, col in cells:
            for tile, prob in ((2, 0.9), (4, 0.1)):
                child = x.copy()
                child[row, col] = tile
                q = decision(child, remaining - 1)
                best = max(q)
                value += prob / len(cells) * (best if np.isfinite(best) else TERMINAL)
        return value

    def decision(s, remaining):
        q = np.full(4, -np.inf)
        for action in range(4):
            child, _, legal = slide(s, action)
            if legal:
                q[action] = after(child, remaining)
        return q
    return decision(np.asarray(board), depth)
