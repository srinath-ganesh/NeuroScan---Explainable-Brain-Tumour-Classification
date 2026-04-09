"""
Train vs val/test transforms for brain MRI.
- Train: heavy augmentation (Resize 256, RandomRotation, RandomAffine, ColorJitter, RandomCrop 224) for OOD robustness.
- Val/Test: clean pipeline (Resize 256, CenterCrop 224, ToTensor, Normalize).
"""
from torchvision import transforms as T


class ToThreeChannels:
    """Replicate grayscale (1,H,W) to 3 channels for ImageNet-pretrained backbones."""

    def __call__(self, x):
        if x.shape[0] == 1:
            return x.repeat(3, 1, 1)
        return x


# ImageNet normalization (used by both train and val)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms():
    """
    Heavy augmentation for training (OOD robustness):
    Resize 256, RandomRotation(15), RandomAffine(translate), ColorJitter, RandomCrop(224), ToTensor, Normalize.
    Applied strictly to the training split only.
    """
    return T.Compose([
        T.Resize((256, 256)),
        T.RandomRotation(degrees=15),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.RandomCrop(224),
        T.ToTensor(),
        ToThreeChannels(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_test_transforms():
    """
    Clean pipeline for validation and test:
    Resize 256, CenterCrop(224), ToTensor, Normalize.
    Applied strictly to validation and test splits.
    """
    return T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor(),
        ToThreeChannels(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
