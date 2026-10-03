import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from prepare_seg_data import prepare, patient_split
from src.seg_manifest import load_manifest, supervised_rows
from src.seg_metrics import binary_scores, summarize


class SegmentationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def manifest(self, rows):
        fields = ['patient_id','image_path','mask_path','annotation_type','review_status','split']
        path = self.root / 'manifest.csv'
        with path.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def row(self, name='a.png', **updates):
        Image.new('L', (32, 16)).save(self.root / name)
        row = dict(patient_id='1', image_path=name, mask_path='', annotation_type='unannotated',
                   review_status='pending', split='train')
        row.update(updates)
        return row

    def test_unannotated_is_not_negative(self):
        rows = load_manifest(self.manifest([self.row()]))
        with self.assertRaisesRegex(ValueError, 'No approved'):
            supervised_rows(rows, 'train')

    def test_patient_leakage_rejected_even_if_unannotated(self):
        rows = [self.row(), self.row('b.png', split='test')]
        with self.assertRaisesRegex(ValueError, 'Patient leakage'):
            load_manifest(self.manifest(rows))

    def test_pseudo_cannot_enter_test(self):
        row = self.row(split='test', annotation_type='pseudo', mask_path='a.png')
        with self.assertRaisesRegex(ValueError, 'only in train'):
            load_manifest(self.manifest([row]))

    def test_verified_negative_allowed_without_mask(self):
        row = self.row(annotation_type='negative', review_status='approved')
        self.assertEqual(len(supervised_rows(load_manifest(self.manifest([row])), 'train')), 1)

    def test_positive_metrics_and_empty_policy(self):
        positive = binary_scores([[1,0],[0,0]], [[1,0],[0,0]])
        negative = binary_scores([[0,1],[0,0]], [[0,0],[0,0]])
        report = summarize([positive, negative])
        self.assertEqual(report['positive_dice'], 1)
        self.assertEqual(report['negative_image_fp_rate'], 1)
        self.assertEqual(report['negative_pixel_fp_rate'], .25)
        self.assertIsNone(binary_scores([0], [0])['dice'])
        self.assertEqual(binary_scores([0], [1])['dice'], 0)

    def test_prepare_reports_missing_and_keeps_patients_together(self):
        Image.new('L', (8,8)).save(self.root / 'a.png')
        Image.new('L', (8,8)).save(self.root / 'b.png')
        csv_path = self.root / 'nih.csv'
        csv_path.write_text('Patient ID,Image Index,Finding Labels\n1,a.png,Nodule\n1,b.png,No Finding\n2,missing.png,Nodule\n')
        report = prepare(csv_path, self.root, self.root/'manifest.csv')
        self.assertEqual(report['missing_images'], ['missing.png'])
        rows = load_manifest(self.root/'manifest.csv')
        self.assertEqual(rows[0]['split'], rows[1]['split'])
        self.assertTrue(all(r['annotation_type']=='unannotated' for r in rows))
        self.assertEqual(patient_split('1'), patient_split('1'))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is not installed')
    def test_mask_alignment_and_restore(self):
        from src.seg_transforms import letterbox, restore_probability
        mask = np.zeros((16,32), dtype=np.uint8)
        mask[4:12,8:24] = 255
        image, target, geometry = letterbox(Image.fromarray(mask), Image.fromarray(mask), 64)
        self.assertTrue(np.array_equal(image.numpy() > .5, target.numpy() > .5))
        restored = restore_probability(target[0], geometry).numpy() > .5
        self.assertTrue(np.array_equal(restored, mask > 0))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is not installed')
    def test_invalid_and_empty_masks_rejected(self):
        from src.seg_dataset import SegDataset
        row = self.row(annotation_type='mask', review_status='approved', mask_path='mask.png')
        Image.new('L', (32,16), 2).save(self.root/'mask.png')
        dataset = SegDataset(load_manifest(self.manifest([row])), 64)
        with self.assertRaisesRegex(ValueError, 'binary'):
            dataset.validate_all()
        Image.new('L', (32,16), 0).save(self.root/'mask.png')
        with self.assertRaisesRegex(ValueError, 'Empty mask'):
            dataset.validate_all()

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('monai'),
                         'PyTorch/MONAI are not installed')
    def test_forward_backward_smoke(self):
        import torch
        from monai.losses import DiceCELoss
        from src.seg_network import build_model
        torch.set_num_threads(1)
        model = build_model()
        images = torch.rand(2,1,64,64)
        target = torch.zeros_like(images)
        target[:,:,20:30,20:30] = 1
        logits = model(images)
        self.assertEqual(logits.shape, target.shape)
        loss = DiceCELoss(sigmoid=True)(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
