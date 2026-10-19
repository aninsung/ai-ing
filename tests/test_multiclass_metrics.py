import unittest
import numpy as np
from sklearn.metrics import f1_score
from src.multiclass_metrics import macro_binary_auroc, primary_decision_metrics


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


class PrimaryDecisionTests(unittest.TestCase):
    def test_macro_f1_and_rare_recall(self):
        names = ["No Finding", "Nodule", "Pneumonia", "Edema", "Fibrosis"]
        targets = np.array([0, 0, 1, 1, 2, 3, 4])
        predictions = np.array([0, 0, 1, 0, 0, 3, 4])
        metrics = primary_decision_metrics(targets, predictions, names)
        expected_f1 = f1_score(targets, predictions, average="macro", labels=[0, 1, 2, 3, 4], zero_division=0)
        self.assertAlmostEqual(metrics["macro_f1"], expected_f1)
        self.assertAlmostEqual(metrics["accuracy"], 5 / 7)
        self.assertAlmostEqual(metrics["class_recall"]["Nodule"], 0.5)
        self.assertAlmostEqual(metrics["class_recall"]["Pneumonia"], 0.0)
        self.assertAlmostEqual(metrics["class_recall"]["Edema"], 1.0)
        self.assertAlmostEqual(metrics["class_recall"]["Fibrosis"], 1.0)
        self.assertAlmostEqual(metrics["rare_recall"], np.mean([0.5, 0.0, 1.0, 1.0]))
        self.assertAlmostEqual(metrics["balanced_accuracy"], np.mean([1.0, 0.5, 0.0, 1.0, 1.0]))

    def test_missing_rare_class_is_skipped(self):
        metrics = primary_decision_metrics([0, 0], [0, 1], ["No Finding", "Nodule"])
        self.assertIsNone(metrics["class_recall"]["Nodule"])
        self.assertIsNone(metrics["rare_recall"])
        self.assertIsNone(primary_decision_metrics([0], [0], ["No Finding"])["rare_recall"])


if __name__ == "__main__":
    unittest.main()
