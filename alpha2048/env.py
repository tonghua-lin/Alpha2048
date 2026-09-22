"""Stateful environment with isolated RNG and reproducible snapshots."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .core import Action, Board, copy_board, is_terminal, legal_actions, slide, spawn_tile


@dataclass(frozen=True)
class StepResult:
    board: Board
    reward: int
    terminated: bool
    changed: bool
    score: int


@dataclass(frozen=True)
class GameState:
    """Detached snapshot. RNG state is included for exact continuation."""

    board: Board
    score: int
    rng_state: dict


class Game2048:
    """Standard 2048. Construction starts a game; reset starts another.

    Invalid moves and steps after termination are no-ops with reward zero.
    Returned arrays are copies and can safely be used by callers.
    """

    def __init__(self, size: int = 4, seed: int | None = None):
        if isinstance(size, bool) or not isinstance(size, int) or size < 2:
            raise ValueError("size must be an integer >= 2")
        self._size = size
        self._rng = np.random.default_rng(seed)
        self.reset()

    @property
    def size(self) -> int:
        return self._size

    @property
    def board(self) -> Board:
        return self._board.copy()

    @property
    def score(self) -> int:
        return self._score

    @property
    def terminated(self) -> bool:
        return is_terminal(self._board)

    def reset(self, seed: int | None = None) -> Board:
        """Start with two tiles. A seed restarts RNG; None continues its stream."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._score = 0
        self._board = np.zeros((self.size, self.size), dtype=np.int64)
        for _ in range(2):
            self._board = spawn_tile(self._board, self._rng)
        return self.board

    def legal_actions(self) -> tuple[Action, ...]:
        return legal_actions(self._board)

    def step(self, action: Action | int) -> StepResult:
        board, reward, changed = slide(self._board, action)
        if changed:
            self._board = spawn_tile(board, self._rng)
            self._score += reward
        return StepResult(self.board, reward, self.terminated, changed, self.score)

    def get_state(self) -> GameState:
        return GameState(self.board, self.score, deepcopy(self._rng.bit_generator.state))

    def set_state(self, state: GameState) -> None:
        """Restore a snapshot of matching board size; validate before changing state."""
        board = copy_board(state.board)
        if board.shape != (self.size, self.size):
            raise ValueError("snapshot size does not match environment")
        if isinstance(state.score, bool) or not isinstance(state.score, (int, np.integer)) or state.score < 0:
            raise ValueError("score must be a nonnegative integer")
        rng = np.random.default_rng()
        rng.bit_generator.state = deepcopy(state.rng_state)
        self._board = board
        self._score = int(state.score)
        self._rng = rng

    def clone(self) -> "Game2048":
        """Independent environment with identical board, score, and future RNG."""
        result = Game2048(size=self.size)
        result.set_state(self.get_state())
        return result
