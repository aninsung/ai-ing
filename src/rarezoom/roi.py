"""Overlapping 512 crops taken from the 1024 original."""
from PIL import Image
import numpy as np
import torch

RESAMPLE = getattr(Image, "Resampling", Image).BILINEAR
RAW_SIZE = 1024
CROP = 512
STRIDE = 256
VIEW = 224
LOCAL_SIZE = 384


def roi_boxes(raw_size=RAW_SIZE, crop=CROP, stride=STRIDE):
    if (raw_size - crop) % stride:
        raise ValueError("ROI grid does not cover the raw image")
    steps = (raw_size - crop) // stride + 1
    boxes = []
    for row in range(steps):
        for col in range(steps):
            y, x = row * stride, col * stride
            boxes.append((y, x, y + crop, x + crop))
    return boxes


BOXES = roi_boxes()


def load_raw(path, raw_size=RAW_SIZE):
    with Image.open(path) as source:
        image = source.convert("L")
        width, height = image.size
        if (width, height) != (raw_size, raw_size):
            image = image.resize((raw_size, raw_size), RESAMPLE)
        array = np.asarray(image, dtype=np.uint8)
    return array, width, height


def view_tensor(array, size=VIEW):
    image = Image.fromarray(array).resize((size, size), RESAMPLE)
    values = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(values).unsqueeze(0)


def crop_tensor(array, roi, size=LOCAL_SIZE):
    top, left, bottom, right = BOXES[roi]
    return view_tensor(array[top:bottom, left:right], size)
