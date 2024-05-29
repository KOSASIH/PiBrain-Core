# AI Optimizer Tests

import unittest
import torch
from optimizer import Optimizer

class TestOptimizer(unittest.TestCase):
    """
    Test Optimizer Class.

    Args:
        unittest.TestCase: Base class for test cases.
    """
    def test_optimizer_initialization(self):
        """
        Tests the initialization of the optimizer.
        """
        optimizer = Optimizer()
        self.assertIsInstance(optimizer, torch.optim.Adam)

    def test_optimizer_step(self):
        """
        Tests the step function of the optimizer.
        """
        optimizer = Optimizer()
        params = [torch.tensor([1.0], requires_grad=True)]
        optimizer.zero_grad()
        loss = torch.tensor([1.0], requires_grad=False)
        loss.backward()
        optimizer.step()
        self.assertEqual(params[0].grad, torch.tensor([1.0]))

if __name__ == '__main__':
    unittest.main()
