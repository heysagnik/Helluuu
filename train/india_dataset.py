"""
Satellite Segmentation Datasets
===============================

Support for publicly available satellite/aerial segmentation datasets:
- Dubai Aerial Imagery (Kaggle - verified available)
- LandCover.ai (Poland - verified available)
- Generic format for any satellite dataset

Classes (standard mapping):
    0: Background/Unlabeled
    1: Roads
    2: Water Bodies
    3: Vegetation/Trees
    4: Buildings/Urban
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Callable, Dict, List
from PIL import Image

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# Standard class definitions for satellite segmentation
SATELLITE_CLASSES = {
    0: 'background',  # Unlabeled/Background
    1: 'roads',       # Roads and transportation
    2: 'water',       # Rivers, lakes, ocean
    3: 'vegetation',  # Trees, forests, vegetation
    4: 'buildings'    # Buildings, urban areas
}

# Optimized class weights for urban satellite imagery
SATELLITE_CLASS_WEIGHTS = [0.5, 1.5, 1.2, 0.7, 1.3]


class DubaiAerialDataset(Dataset):
    """
    Dubai Aerial Imagery Semantic Segmentation Dataset.
    
    Available on Kaggle: humansintheloop/semantic-segmentation-of-aerial-imagery
    
    Original classes (6):
        Building, Land, Road, Vegetation, Water, Unlabeled
    
    Mapped to our 5-class scheme:
        0: Unlabeled + Land -> background
        1: Road -> roads
        2: Water -> water
        3: Vegetation -> vegetation  
        4: Building -> buildings
    
    Download:
        kaggle datasets download -d humansintheloop/semantic-segmentation-of-aerial-imagery
    
    Args:
        root: Root directory after extraction
        split: 'train' or 'val' (dataset will be split 80/20)
        transform: Optional transforms
        image_size: Target image size
    """
    
    # Dubai dataset RGB colors -> class mapping
    DUBAI_COLORS = {
        (155, 155, 155): 0,  # Unlabeled -> background
        (226, 169, 41): 0,   # Land -> background  
        (132, 41, 246): 1,   # Road -> roads
        (110, 193, 228): 2,  # Water -> water
        (60, 16, 152): 3,    # Vegetation -> vegetation
        (254, 221, 58): 4,   # Building -> buildings
    }
    
    CLASSES = ['background', 'roads', 'water', 'vegetation', 'buildings']
    NUM_CLASSES = 5
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (512, 512),
        val_ratio: float = 0.2
    ):
        super().__init__()
        
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        # Find all tiles
        all_samples = self._load_all_samples()
        
        # Split train/val
        split_idx = int(len(all_samples) * (1 - val_ratio))
        if split == 'train':
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]
        
        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {root}")
        
        print(f"DubaiAerialDataset: {len(self.samples)} samples for {split}")
    
    def _load_all_samples(self) -> List[Tuple[Path, Path]]:
        """Find all image-mask pairs."""
        samples = []
        
        # Check common structures
        possible_structures = [
            # Structure 1: Tile folders
            (self.root / 'Tile *' / 'images', self.root / 'Tile *' / 'masks'),
            # Structure 2: Flat
            (self.root / 'images', self.root / 'masks'),
            # Structure 3: Semantic segmentation folder
            (self.root / 'semantic_drone_dataset' / 'images', 
             self.root / 'semantic_drone_dataset' / 'masks'),
        ]
        
        # Check for Tile folders
        tile_folders = list(self.root.glob('Tile*')) + list(self.root.glob('tile*'))
        
        if tile_folders:
            for tile_dir in sorted(tile_folders):
                img_dir = tile_dir / 'images'
                mask_dir = tile_dir / 'masks'
                
                if img_dir.exists() and mask_dir.exists():
                    for img_file in sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png')):
                        mask_file = mask_dir / img_file.name.replace('.jpg', '.png')
                        if not mask_file.exists():
                            mask_file = mask_dir / img_file.name
                        if mask_file.exists():
                            samples.append((img_file, mask_file))
        else:
            # Flat structure
            img_dir = self.root / 'images'
            mask_dir = self.root / 'masks'
            
            if img_dir.exists() and mask_dir.exists():
                for img_file in sorted(img_dir.glob('*')):
                    if img_file.suffix.lower() in ['.jpg', '.png', '.tif']:
                        stem = img_file.stem
                        for ext in ['.png', '.jpg', '.tif']:
                            mask_file = mask_dir / (stem + ext)
                            if mask_file.exists():
                                samples.append((img_file, mask_file))
                                break
        
        return samples
    
    def _rgb_to_class(self, rgb_mask: np.ndarray) -> np.ndarray:
        """Convert RGB mask to class indices."""
        class_mask = np.zeros(rgb_mask.shape[:2], dtype=np.int64)
        
        for color, class_idx in self.DUBAI_COLORS.items():
            matches = np.all(rgb_mask == color, axis=-1)
            class_mask[matches] = class_idx
        
        return class_mask
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]
        
        # Load
        image = np.array(Image.open(img_path).convert('RGB'))
        mask_rgb = np.array(Image.open(mask_path).convert('RGB'))
        
        # Convert RGB mask to class indices
        mask = self._rgb_to_class(mask_rgb)
        
        # Resize
        if image.shape[:2] != self.image_size:
            image = np.array(Image.fromarray(image).resize(
                (self.image_size[1], self.image_size[0]), Image.BILINEAR
            ))
            mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize(
                (self.image_size[1], self.image_size[0]), Image.NEAREST
            )).astype(np.int64)
        
        # Apply transforms
        if self.transform is not None:
            result = self.transform(image=image, mask=mask)
            if isinstance(result, dict):
                image, mask = result['image'], result['mask']
            else:
                image, mask = result
        
        # Convert to tensors
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            if image.max() > 1.0:
                image = image / 255.0
        
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        
        return image, mask


class LandCoverAIDataset(Dataset):
    """
    LandCover.ai Dataset (Poland aerial imagery).
    
    Available at: https://landcover.ai.linuxpolska.com/
    
    Classes (4):
        0: Background
        1: Buildings -> buildings
        2: Woodlands -> vegetation
        3: Water -> water
        4: Roads -> roads (if available)
    
    Args:
        root: Root directory
        split: 'train' or 'val'
        transform: Optional transforms
        image_size: Target image size
    """
    
    # Mapping LandCover.ai classes to our standard
    LANDCOVER_MAPPING = {
        0: 0,  # Background
        1: 4,  # Buildings -> buildings
        2: 3,  # Woodlands -> vegetation
        3: 2,  # Water -> water
    }
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (512, 512)
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        self.samples = self._load_samples()
        print(f"LandCoverAIDataset: {len(self.samples)} samples")
    
    def _load_samples(self) -> List[Tuple[Path, Path]]:
        samples = []
        
        img_dir = self.root / 'images'
        mask_dir = self.root / 'masks'
        
        if not img_dir.exists():
            img_dir = self.root / self.split / 'images'
            mask_dir = self.root / self.split / 'masks'
        
        if img_dir.exists() and mask_dir.exists():
            for img_file in sorted(img_dir.glob('*.tif')) + sorted(img_dir.glob('*.png')):
                mask_file = mask_dir / (img_file.stem + '.tif')
                if not mask_file.exists():
                    mask_file = mask_dir / (img_file.stem + '.png')
                if mask_file.exists():
                    samples.append((img_file, mask_file))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]
        
        # Load image
        if img_path.suffix.lower() == '.tif' and HAS_RASTERIO:
            with rasterio.open(img_path) as src:
                image = src.read()[:3].transpose(1, 2, 0)
        else:
            image = np.array(Image.open(img_path).convert('RGB'))
        
        # Load mask
        if mask_path.suffix.lower() == '.tif' and HAS_RASTERIO:
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
        else:
            mask = np.array(Image.open(mask_path))
        
        # Remap classes
        new_mask = np.zeros_like(mask)
        for src_class, dst_class in self.LANDCOVER_MAPPING.items():
            new_mask[mask == src_class] = dst_class
        mask = new_mask
        
        # Resize
        if image.shape[:2] != self.image_size:
            image = np.array(Image.fromarray(image).resize(
                (self.image_size[1], self.image_size[0]), Image.BILINEAR
            ))
            mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize(
                (self.image_size[1], self.image_size[0]), Image.NEAREST
            ))
        
        if self.transform:
            result = self.transform(image=image, mask=mask)
            image, mask = result['image'], result['mask']
        
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        
        return image, mask


def print_dataset_instructions():
    """Print download instructions for available datasets."""
    print("=" * 70)
    print("Available Satellite Segmentation Datasets")
    print("=" * 70)
    
    print("\n1. DUBAI AERIAL IMAGERY (Recommended - easiest to use)")
    print("-" * 50)
    print("   Kaggle: humansintheloop/semantic-segmentation-of-aerial-imagery")
    print("   Classes: Building, Land, Road, Vegetation, Water, Unlabeled")
    print("   Size: ~72 images in 8 tiles")
    print()
    print("   Download:")
    print("     pip install kaggle")
    print("     kaggle datasets download -d humansintheloop/semantic-segmentation-of-aerial-imagery")
    print("     # Extract to ./data/dubai/")
    
    print("\n2. LANDCOVER.AI (Poland - high resolution)")
    print("-" * 50)
    print("   Website: https://landcover.ai.linuxpolska.com/")
    print("   Classes: Background, Buildings, Woodlands, Water")
    print("   Size: 41 large images (~216 km²)")
    print()
    print("   Download: Visit website and request access")
    
    print("\n" + "=" * 70)
    print("Usage:")
    print("=" * 70)
    print("""
from train import DubaiAerialDataset, create_satellite_dataloader

# Load Dubai dataset
train_loader = create_satellite_dataloader(
    root='./data/dubai',
    dataset_type='dubai',
    split='train',
    batch_size=8
)

for images, masks in train_loader:
    print(images.shape, masks.shape)
    break
""")
    print("=" * 70)


def create_satellite_dataloader(
    root: str,
    split: str = 'train',
    dataset_type: str = 'dubai',
    batch_size: int = 8,
    image_size: Tuple[int, int] = (512, 512),
    num_workers: int = 4,
    transform: Optional[Callable] = None
) -> DataLoader:
    """
    Create DataLoader for satellite datasets.
    
    Args:
        root: Dataset root directory
        split: 'train' or 'val'
        dataset_type: 'dubai' or 'landcover'
        batch_size: Batch size
        image_size: Image size (H, W)
        num_workers: Number of workers
        transform: Optional transform
    
    Returns:
        DataLoader instance
    """
    if dataset_type.lower() == 'dubai':
        dataset = DubaiAerialDataset(
            root=root, split=split, transform=transform, image_size=image_size
        )
    elif dataset_type.lower() in ['landcover', 'landcoverai']:
        dataset = LandCoverAIDataset(
            root=root, split=split, transform=transform, image_size=image_size
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_type}. Use 'dubai' or 'landcover'")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )


if __name__ == "__main__":
    print_dataset_instructions()
