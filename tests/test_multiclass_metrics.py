import unittest
import numpy as np
from src.multiclass_metrics import macro_binary_auroc


class MacroAurocTests(unittest.TestCase):
    def test_perfect_ranking_is_one(self):
        targets = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
        probabilities = np.array([[0.1, 0.2], [0.9, 0.8], [0.2, 0.7], [0.8, 0.3]])
        macro, scores = macro_binary_auroc(targets, probabilities)
        self.assertAlmostEqual(macro, 1.0)
        self.assertEqual(scores, [1.0, 1.0])

    def test_single_class_column_is_excluded(self):
        targets = np.array([[0, 1], [0, 0], [0, 1]])
        probabilities = np.array([[0.2, 0.9], [0.1, 0.2], [0.3, 0.8]])
        macro, scores = macro_binary_auroc(targets, probabilities)
        self.assertIsNone(scores[0])
        self.assertAlmostEqual(macro, 1.0)


if __name__ == "__main__":
    unittest.main()
