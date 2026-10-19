import unittest
import torch
from src.multiclass_model import MultiTaskResNet50, build_multitask_model


class BackboneTests(unittest.TestCase):
    def test_resnet_and_densenet_emit_both_heads(self):
        images = torch.rand(2, 1, 224, 224)
        for name in ("resnet50", "densenet121"):
            model = build_multitask_model(name, num_classes=15, auxiliary_classes=14, pretrained=False)
            model.eval()
            with torch.no_grad():
                primary, auxiliary = model(images)
            self.assertEqual(tuple(primary.shape), (2, 15))
            self.assertEqual(tuple(auxiliary.shape), (2, 14))

    def test_resnet_alias_matches_factory_keys(self):
        alias = MultiTaskResNet50(15, 14, pretrained=False)
        built = build_multitask_model("resnet50", 15, 14, pretrained=False)
        self.assertEqual(set(alias.state_dict()), set(built.state_dict()))


if __name__ == "__main__":
    unittest.main()
