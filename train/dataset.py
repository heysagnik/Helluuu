"""
Satellite Image Dataset
=======================

Dataset class for loading and preprocessing satellite imagery
for semantic segmentation.

Supports:
- Common image formats (TIFF, PNG, JPG)
- Automatic tile extraction from large images
- Class mapping and remapping
"""

import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, Callable, List, Dict
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from .augmentations import get_train_transforms, get_val_transforms


# Default class mapping
DEFAULT_CLASS_MAPPING = {
    'background': 0,
    'roads': 1,
    'water': 2,
    'trees': 3,
    'buildings': 4
}


class SatelliteDataset(Dataset):
    """
    Dataset for satellite image segmentation.
    
    Expects data organized as:
        root/
            images/
                img_001.png
                img_002.png
                ...
            masks/
                img_001.png
                img_002.png
                ...
    
    Args:
        root: Root directory containing images/ and masks/
        split: Dataset split ('train', 'val', 'test')
        transform: Transform to apply
        image_size: Target image size
        class_mapping: Dictionary mapping class names to indices
        tile_size: If set, extract tiles from large images
        tile_overlap: Overlap between tiles
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (512, 512),
        class_mapping: Optional[Dict[str, int]] = None,
        tile_size: Optional[int] = None,
        tile_overlap: int = 0
    ):
        super().__init__()
        
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.class_mapping = class_mapping or DEFAULT_CLASS_MAPPING
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        
        # Set transform
        if transform is not None:
            self.transform = transform
        elif split == 'train':
            self.transform = get_train_transforms(size=image_size)
        else:
            self.transform = get_val_transforms(size=image_size)
        
        # Find images and masks
        self.samples = self._get_samples()
        
        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {root}")
    
    def _get_samples(self) -> List[Tuple[Path, Path]]:
        """Find all image-mask pairs."""
        samples = []
        
        # Supported extensions
        extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        
        # Check for Tile subfolders (Dubai dataset structure)
        tile_folders = list(self.root.glob("Tile*")) + list(self.root.glob("tile*"))
        
        if tile_folders:
            print(f"Found {len(tile_folders)} tile folders")
            for tile_dir in sorted(tile_folders):
                img_dir = tile_dir / 'images'
                mask_dir = tile_dir / 'masks'
                
                if img_dir.exists() and mask_dir.exists():
                    for img_path in sorted(img_dir.iterdir()):
                        if img_path.suffix.lower() in extensions:
                            # Find corresponding mask
                            for ext in extensions:
                                mask_path = mask_dir / f"{img_path.stem}{ext}"
                                if mask_path.exists():
                                    samples.append((img_path, mask_path))
                                    break
            if samples:
                return samples
        
        # Look for standard directory structure
        image_dir = self.root / 'images'
        mask_dir = self.root / 'masks'
        
        if not image_dir.exists():
            # Try alternative: root/train/images, root/val/images etc.
            image_dir = self.root / self.split / 'images'
            mask_dir = self.root / self.split / 'masks'
        
        if not image_dir.exists():
            # Flat structure: assume images are in root directly
            image_dir = self.root
            mask_dir = self.root / 'labels'
        
        if not image_dir.exists():
            return samples
        
        for img_path in sorted(image_dir.iterdir()):
            if img_path.suffix.lower() in extensions:
                # Find corresponding mask
                mask_name = img_path.stem
                for ext in extensions:
                    mask_path = mask_dir / f"{mask_name}{ext}"
                    if mask_path.exists():
                        samples.append((img_path, mask_path))
                        break
        
        return samples
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load image from file."""
        if path.suffix.lower() in {'.tif', '.tiff'} and HAS_RASTERIO:
            with rasterio.open(path) as src:
                image = src.read()
                # CHW -> HWC
                image = np.transpose(image, (1, 2, 0))
                # Handle multi-band images (use first 3 or 4 bands)
                if image.shape[2] > 3:
                    image = image[:, :, :3]
        elif HAS_PIL:
            image = np.array(Image.open(path).convert('RGB'))
        else:
            raise ImportError("Either PIL or rasterio is required to load images")
        
        return image.astype(np.uint8)
    
    def _load_mask(self, path: Path) -> np.ndarray:
        """Load segmentation mask from file."""
        if path.suffix.lower() in {'.tif', '.tiff'} and HAS_RASTERIO:
            with rasterio.open(path) as src:
                mask = src.read(1)
        elif HAS_PIL:
            mask = np.array(Image.open(path))
        else:
            raise ImportError("Either PIL or rasterio is required to load masks")
        
        # Handle RGB masks (Dubai dataset uses color-coded masks)
        if mask.ndim == 3 and mask.shape[2] >= 3:
            mask = self._rgb_to_class(mask)
        elif mask.ndim == 3:
            mask = mask[:, :, 0]
        
        return mask.astype(np.int64)
    
    def _rgb_to_class(self, rgb_mask: np.ndarray) -> np.ndarray:
        """Convert RGB mask to class indices for Dubai dataset."""
        # Dubai dataset color mapping (R, G, B) -> class
        COLOR_TO_CLASS = {
            (155, 155, 155): 0,  # Unlabeled -> background
            (226, 169, 41): 0,   # Land -> background
            (132, 41, 246): 1,   # Road -> roads
            (110, 193, 228): 2,  # Water -> water
            (60, 16, 152): 3,    # Vegetation -> trees
            (254, 221, 58): 4,   # Building -> buildings
        }
        
        h, w = rgb_mask.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.int64)
        
        for color, class_idx in COLOR_TO_CLASS.items():
            matches = np.all(rgb_mask[:, :, :3] == color, axis=-1)
            class_mask[matches] = class_idx
        
        return class_mask
    
    def _extract_tiles(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Extract tiles from large image."""
        h, w = image.shape[:2]
        size = self.tile_size
        stride = size - self.tile_overlap
        
        tiles = []
        for y in range(0, h - size + 1, stride):
            for x in range(0, w - size + 1, stride):
                tile_img = image[y:y + size, x:x + size]
                tile_mask = mask[y:y + size, x:x + size]
                tiles.append((tile_img, tile_mask))
        
        return tiles
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]
        
        # Load image and mask
        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)
        
        # Apply transforms
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        # Ensure mask values are in valid range [0, num_classes-1]
        if isinstance(mask, torch.Tensor):
            mask = mask.clamp(0, len(self.class_mapping) - 1)
        else:
            mask = np.clip(mask, 0, len(self.class_mapping) - 1)
        
        return image, mask
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights based on frequency.
        
        Less frequent classes get higher weights.
        """
        print("Calculating class weights...")
        
        class_counts = np.zeros(len(self.class_mapping))
        
        for img_path, mask_path in self.samples[:100]:  # Sample subset for speed
            mask = self._load_mask(mask_path)
            for c in range(len(self.class_mapping)):
                class_counts[c] += (mask == c).sum()
        
        # Inverse frequency
        total = class_counts.sum()
        weights = total / (class_counts + 1e-6)
        
        # Normalize
        weights = weights / weights.sum() * len(self.class_mapping)
        
        return torch.tensor(weights, dtype=torch.float32)


class TileDataset(Dataset):
    """
    Dataset that extracts tiles from large satellite images.
    
    Useful for very high resolution images that don't fit in memory.
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        tile_size: int = 512,
        tile_overlap: int = 128,
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (512, 512)
    ):
        super().__init__()
        
        self.root = Path(root)
        self.split = split
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.image_size = image_size
        
        # Set transform
        if transform is not None:
            self.transform = transform
        elif split == 'train':
            self.transform = get_train_transforms(size=image_size)
        else:
            self.transform = get_val_transforms(size=image_size)
        
        # Build tile index
        self.tiles = self._build_tile_index()
    
    def _build_tile_index(self) -> List[Tuple[Path, Path, int, int]]:
        """Build index of all tiles."""
        tiles = []
        
        # Find image-mask pairs
        image_dir = self.root / 'images'
        mask_dir = self.root / 'masks'
        
        if not image_dir.exists():
            return tiles
        
        extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        
        for img_path in sorted(image_dir.iterdir()):
            if img_path.suffix.lower() not in extensions:
                continue
            
            # Find mask
            mask_path = None
            for ext in extensions:
                mp = mask_dir / f"{img_path.stem}{ext}"
                if mp.exists():
                    mask_path = mp
                    break
            
            if mask_path is None:
                continue
            
            # Get image dimensions
            if HAS_PIL:
                with Image.open(img_path) as img:
                    w, h = img.size
            else:
                continue
            
            # Generate tile coordinates
            stride = self.tile_size - self.tile_overlap
            for y in range(0, max(1, h - self.tile_size + 1), stride):
                for x in range(0, max(1, w - self.tile_size + 1), stride):
                    tiles.append((img_path, mask_path, x, y))
        
        return tiles
    
    def __len__(self) -> int:
        return len(self.tiles)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path, x, y = self.tiles[idx]
        
        # Load tile
        if HAS_PIL:
            with Image.open(img_path) as img:
                tile_img = img.crop((x, y, x + self.tile_size, y + self.tile_size))
                tile_img = np.array(tile_img.convert('RGB'))
            
            with Image.open(mask_path) as mask:
                tile_mask = mask.crop((x, y, x + self.tile_size, y + self.tile_size))
                tile_mask = np.array(tile_mask)
                if tile_mask.ndim == 3:
                    tile_mask = tile_mask[:, :, 0]
        else:
            raise ImportError("PIL is required for TileDataset")
        
        tile_mask = tile_mask.astype(np.int64)
        
        # Apply transforms
        if self.transform is not None:
            tile_img, tile_mask = self.transform(tile_img, tile_mask)
        
        return tile_img, tile_mask


def create_dataloader(
    root: str,
    split: str = 'train',
    batch_size: int = 8,
    image_size: Tuple[int, int] = (512, 512),
    num_workers: int = 4,
    pin_memory: bool = True
) -> torch.utils.data.DataLoader:
    """
    Create a dataloader for satellite imagery.
    
    Args:
        root: Dataset root directory
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        image_size: Target image size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        DataLoader instance
    """
    dataset = SatelliteDataset(
        root=root,
        split=split,
        image_size=image_size
    )
    
    shuffle = (split == 'train')
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == 'train')
    )


if __name__ == "__main__":
    print("SatelliteDataset - Dataset utilities for satellite imagery")
    print("=" * 60)
    
    print("\nExpected directory structure:")
    print("  root/")
    print("    images/")
    print("      img_001.png")
    print("    masks/")
    print("      img_001.png")
    
    print("\nDefault class mapping:")
    for name, idx in DEFAULT_CLASS_MAPPING.items():
        print(f"  {idx}: {name}")
    
    print("\nUsage:")
    print("  from train.dataset import SatelliteDataset, create_dataloader")
    print("  dataset = SatelliteDataset('path/to/data', split='train')")
    print("  loader = create_dataloader('path/to/data', batch_size=8)")
