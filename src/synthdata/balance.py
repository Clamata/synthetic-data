"""Meeting 1: the two classical ways to add Egyptian Mau training photos.

Both functions take the list of 15 real Egyptian Mau training photos and a
budget (55), and return a list of exactly `budget` Photo objects to append
to the training set.
"""

from collections.abc import Sequence
import random
from torchvision import transforms
from PIL import Image

from synthdata.dataset import Photo


def oversample(
    minority_photos: Sequence[Photo], budget: int, rng: random.Random
) -> list[Photo]:
    """Return `budget` photos by sampling from `minority_photos` with replacement.

    Repeats the photos as evenly as possible. Uses `rng` for any random choices.
    """
    if not minority_photos or budget <= 0:
        return []

    photos = list(minority_photos)
    q, r = divmod(budget, len(photos))
    result = photos * q

    if r > 0:
        result.extend(rng.sample(photos, r))

    return result


def augment(
    minority_photos: Sequence[Photo], budget: int, rng: random.Random
) -> list[Photo]:
    """Return `budget` newly augmented photos derived from `minority_photos`.

    Each returned Photo must have a transform set that is applied at load
    time. Uses `rng` for any random choices.
    """
    if not minority_photos or budget <= 0:
        return []

    sampled = oversample(minority_photos, budget, rng)

    augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    ])

    augmented_photos = []
    for photo in sampled:
        augmented_photos.append(Photo(
            path=photo.path,
            label=photo.label,
            transform=augment_transform
        ))

    return augmented_photos
