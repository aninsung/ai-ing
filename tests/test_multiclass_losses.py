import unittest
import torch
import torch.nn.functional as F
from src.multiclass_dataset import primary_sample_weights
from src.multiclass_losses import class_balanced_focal_loss, effective_number_weights, multitask_loss


class ClassBalancedFocalTests(unittest.TestCase):
    def test_gamma_zero_matches_cross_entropy(self):
        logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 0.5, 0.5]])
        targets = torch.tensor([0, 1])
        weights = torch.ones(3)
        focal = class_balanced_focal_loss(logits, targets, weights, gamma=0.0)
        self.assertAlmostEqual(focal.item(), F.cross_entropy(logits, targets).item(), places=5)

    def test_rare_class_receives_higher_weight(self):
        weights = effective_number_weights(torch.tensor([50000.0, 80.0]), beta=0.9999)
        self.assertGreater(weights[1].item(), weights[0].item())

    def test_rare_rows_draw_more_often(self):
        rows = [{"primary_class_id": "0"}] * 20 + [{"primary_class_id": "1"}]
        weights = primary_sample_weights(rows, power=1.0)
        self.assertGreater(weights[-1], weights[0])

    def test_multitask_cb_focal_is_finite(self):
        primary = torch.randn(4, 3)
        auxiliary = torch.randn(4, 2)
        labels = torch.tensor([0, 1, 2, -1])
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        weights = torch.tensor([0.5, 1.0, 4.0])
        loss, _, _ = multitask_loss(
            primary, auxiliary, labels, targets, weights, 0.5,
            primary_loss_name="cb_focal", focal_gamma=2.0,
        )
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
