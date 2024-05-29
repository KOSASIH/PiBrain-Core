# Decision-Making Tests

import unittest
from decision import Decision

class TestDecision(unittest.TestCase):
    """
    Test Decision Class.

    Args:
        unittest.TestCase: Base class for test cases.
    """
    def test_decision_initialization(self):
        """
        Tests the initialization of the decision.
        """
        decision = Decision()
        self.assertIsInstance(decision, Decision)

    def test_decision_make(self):
        """
        Tests the make function of the decision.
        """
        decision = Decision()
        result = decision.make(1, 2, 3)
        self.assertIsInstance(result, bool)

if __name__ == '__main__':
    unittest.main()
