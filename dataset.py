"""
    This module creates a custom PyTorch Dataset for loading prepared and pre-processed image data and corresponding labels from a CSV DataFrame.

    Each row in the DataFrame contains case_id which is the directory for the image file
    and its associated tumor type.
    
"""
import os
from typing import Any, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CSVDataset(Dataset):
    """
    Attributes:
        df (pd.DataFrame): DataFrame containing dataset metadata.
        args: Object containing configuration parameters such as dataset directories and image settings.
        transform (transforms.Compose): Composed transformation pipeline applied to each image.
    """

    def __init__(self, args, df: pd.DataFrame) -> None:
        """
        Initialize the dataset with arguments and data.
        Initialize the image transformation pipeline for pre-processing.

        Args:
            args: Configuration ArgumentParser.
            df (pd.DataFrame): DataFrame containing 'case_id' and 'tumor' columns.
        """
        self.df: pd.DataFrame = df.reset_index(drop=True)
        self.args = args
        self.transform: transforms.Compose = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a single image and label pair from the dataset.

        Args:
            i (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Image tensor of shape (C, H, W) after transformations.
                - Label tensor (long), typically 0 or 1 for binary classification.
        """
        row = self.df.iloc[i]
        image_path = os.path.join(
            self.args.dataset_dir,
            self.args.image_dir,
            row["case_id"],
            self.args.image_file,
        )
        # image preparation
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img)
        
        # label preparation
        label_idx = int(row["tumor"])
        y = torch.zeros(self.args.num_classes, dtype=torch.int32)
        y[label_idx] = 1
        
        return x, y