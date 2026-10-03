import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from .seg_transforms import letterbox


class SegDataset(Dataset):
    def __init__(self, rows, image_size=512, augment=False):
        self.rows, self.image_size, self.augment = rows, image_size, augment

    def __len__(self):
        return len(self.rows)

    def original(self, index):
        row = self.rows[index]
        with Image.open(row['image_path']) as source:
            image = source.convert('L')
        if row['mask_path']:
            with Image.open(row['mask_path']) as source:
                mask = source.copy()
            if mask.mode not in {'L', '1', 'I', 'I;16'}:
                raise ValueError('Masks must be single-channel, not RGB/palette')
            if mask.size != image.size:
                raise ValueError(f'Image/mask size mismatch: {row["image_path"]}')
            values = set(np.unique(np.asarray(mask)).tolist())
            if not (values <= {0, 1} or values <= {0, 255}):
                raise ValueError(f'Mask must encode binary 0/1 or 0/255: {row["mask_path"]}')
            mask = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255)
        else:
            if row['annotation_type'] != 'negative':
                raise ValueError('Only verified negatives may have no mask')
            mask = Image.new('L', image.size, 0)
        positive = bool(np.any(np.asarray(mask)))
        if row['annotation_type'] == 'negative' and positive:
            raise ValueError('Verified negative has a non-empty mask')
        if row['annotation_type'] in {'mask', 'pseudo'} and not positive:
            raise ValueError('Empty mask: mark reviewed negatives as negative explicitly')
        return image, mask

    def validate_all(self):
        for i in range(len(self)):
            self.original(i)

    def __getitem__(self, index):
        image, mask = self.original(index)
        image, mask, geometry = letterbox(image, mask, self.image_size, self.augment)
        return {'image': image, 'mask': mask, 'geometry': geometry, 'index': index}
