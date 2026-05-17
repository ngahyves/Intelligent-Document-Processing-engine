# src/vision/preprocess.py

import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from src.config.logging_config import get_logger

from src.config.settings import (
    TRAIN_DIR,
    TEST_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED
)

logger = get_logger(__name__)


class DataProcess:
    """
    A modular data processing class responsible for:
    - validating dataset paths
    - building train/test transforms
    - loading ImageFolder datasets
    - creating DataLoaders
    - exposing class names

    All parameters are loaded from settings.py.
    """

    def __init__(
        self,
        train_path: str = TRAIN_DIR,
        test_path: str = TEST_DIR,
        batch_size: int = BATCH_SIZE,
        img_size: int = IMG_SIZE,
        num_workers: int = NUM_WORKERS,
        seed: int = SEED
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.seed = seed

        self.train_dataset = None
        self.test_dataset = None
        self.classes = None

        logger.info("DataProcess initialized with:")
        logger.info(f"Train path: {train_path}")
        logger.info(f"Test path: {test_path}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Image size: {img_size}")
        logger.info(f"Num workers: {num_workers}")

    # ---------------------------------------------------------
    # Path validation
    # ---------------------------------------------------------
    def _validate_paths(self):
        """Ensure train and test directories exist."""
        if not os.path.isdir(self.train_path):
            logger.error(f"Train path not found: {self.train_path}")
            raise FileNotFoundError(f"Train path not found: {self.train_path}")

        if not os.path.isdir(self.test_path):
            logger.error(f"Test path not found: {self.test_path}")
            raise FileNotFoundError(f"Test path not found: {self.test_path}")

        logger.info("Dataset paths validated successfully.")

    # ---------------------------------------------------------
    # Transforms
    # ---------------------------------------------------------
    def _build_transforms(self):
        """Create train and test transforms."""
        train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return train_transform, test_transform

    # ---------------------------------------------------------
    # Dataset loading
    # ---------------------------------------------------------
    def load_datasets(self):
        """Load ImageFolder datasets and store class names."""
        self._validate_paths()
        train_transform, test_transform = self._build_transforms()

        try:
            logger.info("Loading datasets...")
            self.train_dataset = datasets.ImageFolder(
                root=self.train_path, transform=train_transform
            )
            self.test_dataset = datasets.ImageFolder(
                root=self.test_path, transform=test_transform
            )
        except Exception as e:
            logger.exception("Error while loading datasets.")
            raise RuntimeError(f"Error while loading datasets: {e}")

        if len(self.train_dataset) == 0:
            logger.error("Train dataset is empty.")
            raise ValueError("Train dataset is empty.")

        if len(self.test_dataset) == 0:
            logger.error("Test dataset is empty.")
            raise ValueError("Test dataset is empty.")

        self.classes = self.train_dataset.classes

        logger.info(f"Train dataset size: {len(self.train_dataset)} images")
        logger.info(f"Test dataset size: {len(self.test_dataset)} images")
        logger.info(f"Detected classes: {self.classes}")

    # ---------------------------------------------------------
    # DataLoaders
    # ---------------------------------------------------------
    def get_loaders(self):
        """Return train and test DataLoaders."""
        if self.train_dataset is None or self.test_dataset is None:
            logger.warning("Datasets not loaded yet. Calling load_datasets()...")
            self.load_datasets()

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        logger.info("DataLoaders created successfully.")
        return train_loader, test_loader, self.classes
    
#Calling the different functions
if __name__ == "__main__":
    logger.info("Running DataProcess")
    dp = DataProcess()
    train_loader, test_loader, classes = dp.get_loaders()
    logger.info(f"Classes detected: {classes}")

