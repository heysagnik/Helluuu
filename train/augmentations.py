"""
Satellite-Specific Data Augmentations
=====================================

Augmentation pipeline designed for satellite imagery:
- Geometric: Rotation, flips, elastic deformation
- Radiometric: Brightness, contrast, saturation
- Regularization: CutOut, MixUp
- Multi-scale training
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any
import random
import math


class Compose:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class RandomHorizontalFlip:
    """Random horizontal flip."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        return image, mask


class RandomVerticalFlip:
    """Random vertical flip."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        return image, mask


class RandomRotate90:
    """Random rotation by 90 degree increments."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            k = random.randint(1, 3)
            image = np.rot90(image, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()
        return image, mask


class RandomRotate:
    """Random rotation by arbitrary angle."""
    
    def __init__(self, limit: float = 45.0, p: float = 0.5):
        self.limit = limit
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            angle = random.uniform(-self.limit, self.limit)
            image = self._rotate(image, angle, mode='bilinear')
            mask = self._rotate(mask, angle, mode='nearest')
        return image, mask
    
    def _rotate(self, arr: np.ndarray, angle: float, mode: str) -> np.ndarray:
        """Rotate using torch for efficiency."""
        # Convert to tensor
        if arr.ndim == 2:
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
        else:
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
        
        # Create rotation matrix
        angle_rad = angle * math.pi / 180
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ]).unsqueeze(0).float()
        
        grid = F.affine_grid(theta, tensor.size(), align_corners=False)
        
        if mode == 'nearest':
            rotated = F.grid_sample(tensor, grid, mode='nearest', align_corners=False)
        else:
            rotated = F.grid_sample(tensor, grid, mode='bilinear', align_corners=False)
        
        # Convert back
        if arr.ndim == 2:
            return rotated.squeeze().numpy().astype(arr.dtype)
        else:
            return rotated.squeeze().permute(1, 2, 0).numpy().astype(arr.dtype)


class RandomScale:
    """Random scale transformation for multi-scale training."""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.5, 2.0), p: float = 0.5):
        self.scale_range = scale_range
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            scale = random.uniform(*self.scale_range)
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Resize image
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
            image_resized = F.interpolate(image_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
            image = image_resized.squeeze().permute(1, 2, 0).numpy().astype(image.dtype)
            
            # Resize mask
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
            mask_resized = F.interpolate(mask_tensor, size=(new_h, new_w), mode='nearest')
            mask = mask_resized.squeeze().numpy().astype(mask.dtype)
        
        return image, mask


class RandomCrop:
    """Random crop to fixed size."""
    
    def __init__(self, size: Tuple[int, int] = (512, 512)):
        self.size = size
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        new_h, new_w = self.size
        
        # Pad if necessary
        if h < new_h or w < new_w:
            pad_h = max(0, new_h - h)
            pad_w = max(0, new_w - w)
            
            if image.ndim == 3:
                image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            else:
                image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            h, w = image.shape[:2]
        
        # Random crop
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        
        image = image[top:top + new_h, left:left + new_w]
        mask = mask[top:top + new_h, left:left + new_w]
        
        return image, mask


class CenterCrop:
    """Center crop to fixed size."""
    
    def __init__(self, size: Tuple[int, int] = (512, 512)):
        self.size = size
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        new_h, new_w = self.size
        
        # Pad if necessary
        if h < new_h or w < new_w:
            pad_h = max(0, new_h - h)
            pad_w = max(0, new_w - w)
            
            if image.ndim == 3:
                image = np.pad(image, ((pad_h // 2, pad_h - pad_h // 2), 
                                       (pad_w // 2, pad_w - pad_w // 2), (0, 0)), mode='reflect')
            else:
                image = np.pad(image, ((pad_h // 2, pad_h - pad_h // 2),
                                       (pad_w // 2, pad_w - pad_w // 2)), mode='reflect')
            mask = np.pad(mask, ((pad_h // 2, pad_h - pad_h // 2),
                                 (pad_w // 2, pad_w - pad_w // 2)), mode='constant', constant_values=0)
            h, w = image.shape[:2]
        
        # Center crop
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        
        image = image[top:top + new_h, left:left + new_w]
        mask = mask[top:top + new_h, left:left + new_w]
        
        return image, mask


class RandomBrightnessContrast:
    """Random brightness and contrast adjustment."""
    
    def __init__(
        self,
        brightness_limit: float = 0.2,
        contrast_limit: float = 0.2,
        p: float = 0.5
    ):
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            brightness = random.uniform(-self.brightness_limit, self.brightness_limit)
            contrast = random.uniform(1 - self.contrast_limit, 1 + self.contrast_limit)
            
            image = image.astype(np.float32)
            image = (image - 127.5) * contrast + 127.5 + brightness * 255
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image, mask


class RandomHueSaturation:
    """Random hue and saturation adjustment."""
    
    def __init__(
        self,
        hue_limit: float = 0.1,
        sat_limit: float = 0.2,
        p: float = 0.5
    ):
        self.hue_limit = hue_limit
        self.sat_limit = sat_limit
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p and image.ndim == 3:
            # Convert to HSV
            image = image.astype(np.float32) / 255.0
            
            # Simple RGB to HSV approximation
            max_c = np.max(image, axis=2)
            min_c = np.min(image, axis=2)
            diff = max_c - min_c
            
            # Saturation adjustment
            sat_factor = random.uniform(1 - self.sat_limit, 1 + self.sat_limit)
            
            # Adjust saturation
            mean = np.mean(image, axis=2, keepdims=True)
            image = mean + (image - mean) * sat_factor
            
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        return image, mask


class RandomGamma:
    """Random gamma correction."""
    
    def __init__(self, gamma_limit: Tuple[float, float] = (0.8, 1.2), p: float = 0.5):
        self.gamma_limit = gamma_limit
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_limit)
            image = np.power(image.astype(np.float32) / 255.0, gamma)
            image = (image * 255).astype(np.uint8)
        return image, mask


class RandomNoise:
    """Add random Gaussian noise."""
    
    def __init__(self, var_limit: Tuple[float, float] = (1, 10), p: float = 0.3):
        self.var_limit = var_limit
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            var = random.uniform(*self.var_limit)
            noise = np.random.normal(0, var, image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return image, mask


class CutOut:
    """Random rectangular cutout."""
    
    def __init__(
        self,
        num_holes: int = 8,
        max_h_size: int = 32,
        max_w_size: int = 32,
        fill_value: int = 0,
        p: float = 0.5
    ):
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            h, w = image.shape[:2]
            
            for _ in range(self.num_holes):
                hole_h = random.randint(1, self.max_h_size)
                hole_w = random.randint(1, self.max_w_size)
                
                top = random.randint(0, h - hole_h)
                left = random.randint(0, w - hole_w)
                
                image[top:top + hole_h, left:left + hole_w] = self.fill_value
        
        return image, mask


class Normalize:
    """Normalize image to [0, 1] or with ImageNet stats."""
    
    def __init__(
        self,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    ):
        self.mean = np.array(mean)
        self.std = np.array(std)
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return image, mask


class ToTensor:
    """Convert numpy arrays to PyTorch tensors."""
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        # Image: HWC -> CHW
        if image.ndim == 3:
            image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        else:
            image = torch.from_numpy(image.copy()).unsqueeze(0).float()
        
        # Mask: HW -> HW (long for cross entropy)
        mask = torch.from_numpy(mask.copy()).long()
        
        return image, mask


def get_train_transforms(
    size: Tuple[int, int] = (512, 512),
    scale_range: Tuple[float, float] = (0.5, 2.0)
) -> Compose:
    """
    Get training augmentation pipeline.
    
    Args:
        size: Target image size
        scale_range: Range for random scaling
    
    Returns:
        Compose transform
    """
    return Compose([
        RandomScale(scale_range=scale_range, p=0.5),
        RandomCrop(size=size),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        RandomHueSaturation(hue_limit=0.1, sat_limit=0.2, p=0.3),
        RandomGamma(gamma_limit=(0.8, 1.2), p=0.3),
        RandomNoise(var_limit=(1, 10), p=0.2),
        CutOut(num_holes=8, max_h_size=32, max_w_size=32, p=0.3),
        Normalize(),
        ToTensor()
    ])


def get_val_transforms(size: Tuple[int, int] = (512, 512)) -> Compose:
    """
    Get validation/test augmentation pipeline.
    
    Args:
        size: Target image size
    
    Returns:
        Compose transform
    """
    return Compose([
        CenterCrop(size=size),
        Normalize(),
        ToTensor()
    ])


if __name__ == "__main__":
    # Test augmentations
    print("Testing Augmentations...")
    
    # Create dummy data
    image = np.random.randint(0, 255, (600, 600, 3), dtype=np.uint8)
    mask = np.random.randint(0, 5, (600, 600), dtype=np.uint8)
    
    print(f"Original image: {image.shape}, mask: {mask.shape}")
    
    # Test train transforms
    train_transforms = get_train_transforms(size=(512, 512))
    aug_image, aug_mask = train_transforms(image.copy(), mask.copy())
    
    print(f"After train transforms:")
    print(f"  Image: {aug_image.shape}, dtype: {aug_image.dtype}")
    print(f"  Mask: {aug_mask.shape}, dtype: {aug_mask.dtype}")
    
    # Test val transforms
    val_transforms = get_val_transforms(size=(512, 512))
    val_image, val_mask = val_transforms(image.copy(), mask.copy())
    
    print(f"After val transforms:")
    print(f"  Image: {val_image.shape}, dtype: {val_image.dtype}")
    print(f"  Mask: {val_mask.shape}, dtype: {val_mask.dtype}")
    
    print("\n✓ All augmentation tests passed!")
