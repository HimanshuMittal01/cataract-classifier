"""Contains all functions related to data loading and handling image datasets.

This module includes:
1. A custom PyTorch Dataset class (`CataractDataset`) for handling image loading and transformations.
2. A function (`get_dataset_paths`) for obtaining the paths to the training, validation, and testing datasets.
"""

import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class CataractDataset(Dataset):
    """A PyTorch Dataset class for handling images of cataract and normal eyes.

    Attributes:
        img_filepaths : List of image file paths to be loaded.
        transform : Transformations specific to the model to be applied to the images.
        augmentation : Augmentations (e.g., rotations, flips) to be applied to the images.
    """

    def __init__(
        self,
        img_filepaths: list[Path],
        transform: Callable,
        augmentation: Optional[Callable] = None,
    ):
        """Initialize the dataset.

        Args:
            img_filepaths : List of image file paths to load.
            transform : Transformations to apply to each image.
            augmentation : Augmentation techniques to apply to images (default is None).
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
        """Retrieve an image and its label from the dataset at the specified index.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            tuple: A tuple containing the image tensor and its corresponding label (0 for "Normal", 1 for "Cataract").
        """
        img_filepath = self._img_filepaths[index]
        img = Image.open(img_filepath).convert("RGB")

        # Determine the label based on the parent folder name ('cataract' or 'normal')
        label = img_filepath.parent.stem.lower()
        label = torch.tensor(1.0) if label == "cataract" else torch.tensor(0.0)

        # Apply augmentations if specified
        if self._augmentation is not None:
            # Convert image to numpy array for albumentations, then back to PIL for torchvision transforms
            img = Image.fromarray(
                self._augmentation(image=np.array(img))["image"]
            )

        # Apply model transforms (assuming it converts to tensor)
        img = self._transform(img)

        return img, label


def get_dataset_paths(
    dataset_path: str | Path,
    valid_test_split: float = 0.5,
    random_seed: int = 0,
) -> dict[str, list[str]]:
    """
    Get file paths for train, validation, and test datasets, and split test dataset into validation and testing sets.

    Args:
        dataset_path : Path to the dataset directory.
        valid_test_split : Proportion of images to include in the validation dataset (default is 0.5).
        random_seed : Random seed for reproducibility of dataset splits (default is 0).

    Returns:
        dict: A dictionary with keys 'train', 'valid', and 'test', each containing a list of file paths for the respective dataset split.
    """
    # Get all image file paths in the 'train' directory
    train_img_files = [
        filepath
        for filepath in (Path(dataset_path) / "train/").rglob("*")
        if filepath.is_file()
    ]

    # Get all image file paths in the 'test/cataract' and 'test/normal' directories
    test_cataract_img_files = [
        filepath
        for filepath in (Path(dataset_path) / "test/cataract/").rglob("*")
        if filepath.is_file()
    ]
    test_normal_img_files = [
        filepath
        for filepath in (Path(dataset_path) / "test/normal/").rglob("*")
        if filepath.is_file()
    ]

    # Shuffle and split the test images into validation and test sets
    random.seed(random_seed)
    random.shuffle(test_cataract_img_files)
    random.shuffle(test_normal_img_files)

    # Determine the number of images to assign to the validation set
    n_valid_cataract = int(len(test_cataract_img_files) * valid_test_split)
    n_valid_normal = int(len(test_normal_img_files) * valid_test_split)
    assert (
        n_valid_cataract > 0 and n_valid_normal > 0
    ), f"No samples in validation dataset; valid_test_ratio={valid_test_split}"

    # Split the images into validation and test sets
    valid_img_files = (
        test_cataract_img_files[:n_valid_cataract]
        + test_normal_img_files[:n_valid_normal]
    )
    test_img_files = (
        test_cataract_img_files[n_valid_cataract:]
        + test_normal_img_files[n_valid_normal:]
    )

    # Shuffle the validation and test datasets
    random.shuffle(valid_img_files)
    random.shuffle(test_img_files)

    return {
        "train": train_img_files,
        "valid": valid_img_files,
        "test": test_img_files,
    }
