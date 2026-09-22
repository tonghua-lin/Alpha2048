"""Public game API. Importing this package does not load the GUI."""

from .core import Action, is_terminal, legal_actions, slide, spawn_tile
from .env import Game2048, GameState, StepResult

__all__ = [
    "Action", "Game2048", "GameState", "StepResult", "is_terminal",
    "legal_actions", "slide", "spawn_tile",
]
