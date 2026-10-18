"""Tests for BBox → weak pseudo-mask preparation."""
import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from prepare_bbox_pseudomasks import prepare
from prepare_seg_data import patient_split
from src.seg_manifest import load_manifest


class BBoxPseudoMaskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / 'images'
        self.images.mkdir()

    def _write_image(self, name, size=(128, 128)):
        path = self.images / name
        Image.new('L', size, 40).save(path)
        return path

    def test_manifest_is_train_pseudo_only_and_loadable(self):
        # Force one train and one non-train patient via known split buckets.
        train_name = val_name = None
        for patient in range(1, 500):
            name = f'{patient:08d}_000.png'
            split = patient_split(str(patient))
            if split == 'train' and train_name is None:
                train_name = name
            if split != 'train' and val_name is None:
                val_name = name
            if train_name and val_name:
                break
        self.assertIsNotNone(train_name)
        self.assertIsNotNone(val_name)

        self._write_image(train_name)
        self._write_image(val_name)
        bbox = self.root / 'bbox.csv'
        bbox.write_text(
            'Image Index,Finding Label,Bbox [x],Bbox [y],Bbox [w],Bbox [h]\n'
            f'{train_name},Nodule,10,10,40,40\n'
            f'{val_name},Nodule,20,20,30,30\n',
            encoding='utf-8',
        )
        report = prepare(
            bbox,
            self.images,
            self.root / 'masks',
            self.root / 'manifest.csv',
        )
        self.assertEqual(report['images'], 1)
        self.assertEqual(report['masks_written'], 2)
        self.assertEqual(report['skipped_non_train'], 1)

        with (self.root / 'manifest.csv').open(encoding='utf-8', newline='') as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['annotation_type'], 'pseudo')
        self.assertEqual(rows[0]['review_status'], 'pending')
        self.assertEqual(rows[0]['split'], 'train')

        # load_manifest accepts pending pseudo in train (paths must exist).
        loaded = load_manifest(self.root / 'manifest.csv')
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]['split'], 'train')


if __name__ == '__main__':
    unittest.main()
