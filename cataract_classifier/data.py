"""
Contains all functions related to data loading
"""

import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class CataractDataset(Dataset):
    """
    A PyTorch Dataset class for handling images.

    This class extends PyTorch's Dataset and is designed to work with image data.
    It supports loading images, and applying transformations.

    Attributes:
        img_paths (list): List of image file paths.
        transform (callable): Transformations to be applied to the images required by the model.
        augmentations (callable): Augmentations to be applied to the images.
    """

    def __init__(
        self,
        img_filepaths: list[Path],
        transform: Callable,
        augmentation: Optional[Callable] = None,
    ):
        """
        Args:
            img_paths (list): List of image file paths.
            class_to_idx (dict): Dictionary mapping class names to class indices.
            transforms (callable, optional): Transformations to be applied to the images.
        """
        super().__init__()
        self._img_filepaths = img_filepaths
        self._transform = transform
        self._augmentation = augmentation

    def __len__(self):
        """Returns the number of items in the dataset.

        Returns:
            int: Number of items in the dataset.
        """
        return len(self._img_filepaths)

    def __getitem__(self, index):
        """Retrieves an item from the dataset at the specified index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image and its corresponding label.
        """
        img_filepath = self._img_filepaths[index]
        img = Image.open(img_filepath).convert("RGB")

        label = img_filepath.parent.stem.lower()
        label = torch.tensor(1.0) if label == "cataract" else torch.tensor(0.0)

        if self._augmentation is not None:
            # Have to convert to numpy array because we're using
            # albumentations Compose() method and converted back to
            # PIL Image to use torchvision.transform.Compose()
            img = Image.fromarray(
                self._augmentation(image=np.array(img))["image"]
            )

        # apply model transforms and convert to tensor
        img = self._transform(img)

        return img, label


def get_dataset_paths(
    dataset_path: str | Path,
    valid_test_split: float = 0.5,
    random_seed: int = 0,
) -> dict[str, list[str]]:
    """Get train, validation and holdout image paths.

    Args:
        dataset_path: /path/to/dataset
        valid_test_split: Proportion of images in the validation dataset, rest goes to the holdout dataset.
            Note that given training data directory is used as it is (no split).
        random_seed: Any number for reproducibility

    Returns:
        a python dictionary having keys 'train', 'valid' and 'test',
        and values as corresponding list of image filepaths.
    """
    # Get all image filepaths
    train_img_files = [
        filepath
        for filepath in (Path(dataset_path) / "train/").rglob("*")
        if filepath.is_file()
    ]
    test_img_files = [
        filepath
        for filepath in (Path(dataset_path) / "test/").rglob("*")
        if filepath.is_file()
    ]

    # Split the test images into validation and testing dataset
    random.seed(random_seed)
    random.shuffle(test_img_files)

    k = int(len(test_img_files) * valid_test_split)
    assert (
        k > 0
    ), f"No samples in validation dataset; valid_test_ratio={valid_test_split}"

    valid_img_files = test_img_files[:k]
    test_img_files = test_img_files[k:]

    return {
        "train": train_img_files,
        "valid": valid_img_files,
        "test": test_img_files,
    }
