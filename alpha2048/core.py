"""Pure board operations; tile values are stored literally (0, 2, 4, ...)."""

from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

Board = NDArray[np.int64]


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


def copy_board(board: Board) -> Board:
    """Validate and copy a square board, without silently truncating values."""
    array = np.asarray(board)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] < 2:
        raise ValueError("board must be a square array of size >= 2")
    if array.dtype.kind not in "iu" or np.any(array < 0):
        raise ValueError("board must contain nonnegative integers")
    if np.any(array > np.iinfo(np.int64).max):
        raise ValueError("tile exceeds int64 storage")
    result = array.astype(np.int64, copy=True)
    if np.any((result != 0) & ((result < 2) | ((result & (result - 1)) != 0))):
        raise ValueError("tiles must be zero or powers of two >= 2")
    return result


def slide(board: Board, action: Action | int) -> tuple[Board, int, bool]:
    """Move without spawning. Each original tile can merge at most once.

    Returns (new_board, merge_score, changed), never mutating the input.
    Actions use the stable encoding UP=0, RIGHT=1, DOWN=2, LEFT=3.
    """
    action = Action(action)
    original = copy_board(board)
    rotations = {Action.LEFT: 0, Action.UP: 1, Action.RIGHT: 2, Action.DOWN: 3}
    k = rotations[action]
    rows = np.rot90(original, k)
    output = np.zeros_like(rows)
    reward = 0
    for row_index, row in enumerate(rows):
        values = [int(value) for value in row if value]
        merged = []
        index = 0
        while index < len(values):
            value = values[index]
            if index + 1 < len(values) and value == values[index + 1]:
                value *= 2
                if value > np.iinfo(np.int64).max:
                    raise OverflowError("merged tile exceeds int64 storage")
                reward += value
                index += 1
            merged.append(value)
            index += 1
        output[row_index, :len(merged)] = merged
    result = np.rot90(output, -k).copy()
    return result, reward, not np.array_equal(result, original)


def spawn_tile(board: Board, rng: np.random.Generator) -> Board:
    """Return a copy with one tile: uniform empty cell, 90% 2 and 10% 4.

    A full board is returned as a copy and consumes no randomness.
    """
    result = copy_board(board)
    empty = np.argwhere(result == 0)
    if len(empty):
        row, column = empty[int(rng.integers(len(empty)))]
        result[row, column] = 2 if rng.random() < 0.9 else 4
    return result


def legal_actions(board: Board) -> tuple[Action, ...]:
    """Actions that change the board, in enum order."""
    return tuple(action for action in Action if slide(board, action)[2])


def is_terminal(board: Board) -> bool:
    """True when no movement or merge is possible; 2048 is not a terminal tile."""
    return not legal_actions(board)
