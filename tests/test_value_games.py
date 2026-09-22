import unittest
import numpy as np
import torch
from alpha2048.test_value_games import score_actions
from alpha2048.core import legal_actions


class FixedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.zeros(len(x), device=x.device)


class ValueGameTests(unittest.TestCase):
    def test_only_legal_actions_and_stable_ties(self):
        board = np.array([[2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        values, chosen = score_actions(FixedModel(), board)
        legal = legal_actions(board)
        self.assertEqual(chosen, min(legal))
        for action in range(4):
            self.assertEqual(values[action] is not None, action in legal)

    def test_terminal_board_has_no_action(self):
        board = np.array([[2, 4, 2, 4], [4, 2, 4, 2]]*2)
        self.assertEqual(score_actions(FixedModel(), board), ([None]*4, None))


if __name__ == '__main__':
    unittest.main()
