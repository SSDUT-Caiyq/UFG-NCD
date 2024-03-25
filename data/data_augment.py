from torchvision import transforms
import random
from PIL import ImageFilter


def get_transform(image_size, crop_pct, interpolation, trans_type="imagenet", task="ncd", mode="train"):
    mean, std = {
        "cifar10": [(0.491, 0.482, 0.447), (0.202, 0.199, 0.201)],
        "cifar100": [(0.507, 0.487, 0.441), (0.267, 0.256, 0.276)],
        "imagenet": [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
    }[trans_type]

    transform = {
        # NCD transform followed UNO
        "ncd": {
            "imagenet": {
                "train": transforms.Compose(
                    [
                        transforms.RandomResizedCrop(image_size, scale=(0.67, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
                        transforms.RandomHorizontalFlip(),
                        # Add
                        transforms.RandomVerticalFlip(),
                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                ),
                "test": transforms.Compose(
                    [
                        transforms.Resize(int(image_size / crop_pct), interpolation),
                        transforms.CenterCrop(image_size),
                        # transforms.Resize(image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                ),
            }
        },
        # GCD transform followed GCD
        "gcd": {
            "imagenet": {
                "train": transforms.Compose(
                    [
                        transforms.Resize(int(image_size / crop_pct), interpolation),
                        transforms.RandomCrop(image_size),
                        transforms.RandomHorizontalFlip(),
                        # Add
                        transforms.RandomVerticalFlip(),
                        transforms.ColorJitter(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                ),
                "test": transforms.Compose(
                    [
                        transforms.Resize(int(image_size / crop_pct), interpolation),
                        transforms.CenterCrop(image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                ),
            }
        },
    }[task][trans_type][mode]

    return transform


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
