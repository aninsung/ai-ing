import monai.transforms as mt

def get_transforms(image_size=224, phase='train'):
    """
    MONAI transforms 파이프라인 (Phase 2)
    """
    base_transforms = [
        mt.LoadImaged(keys=["image"], image_only=True),
        mt.EnsureChannelFirstd(keys=["image"]),
        mt.Lambdad(keys=["image"], func=lambda x: x[0:1, ...]), # RGBA/RGB 등 다중 채널 방지 (1채널 고정)
        mt.Resized(keys=["image"], spatial_size=(image_size, image_size)),
        mt.ScaleIntensityd(keys=["image"]),
        mt.EnsureTyped(keys=["image"], dtype='float32'),
    ]
    return mt.Compose(base_transforms)
