import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from typing import Tuple, Any, List
from ..distribution_shift import DistributionShift
import random
import numpy as np

class TransformedDataset(Dataset):
    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        x, y = self.dataset[idx]
        return self.transform(x), y

    def __len__(self) -> int:
        return len(self.dataset)

class GaussianNoise(DistributionShift):
    """Adds Gaussian noise with increasing standard deviation"""
    SEVERITY_SCALE = [0.05, 0.1, 0.15, 0.2, 0.25]  # Reasonable noise levels
    
    def __init__(self, severity_scale: List[float] = None):
        super().__init__(name="gaussian_noise", severity_scale=severity_scale or self.SEVERITY_SCALE)
        
    def apply(self, dataset: Dataset) -> Dataset:
        def add_noise(x: torch.Tensor) -> torch.Tensor:
            return x + torch.randn_like(x) * self.severity
        return TransformedDataset(dataset, add_noise)

class BrightnessShift(DistributionShift):
    """Adjusts brightness with increasing intensity"""
    SEVERITY_SCALE = [0.5, 0.75, 1.25, 1.5, 2.0]  # Values around 1.0 (original brightness)
    
    def __init__(self, severity_scale: List[float] = None):
        super().__init__(name="brightness", severity_scale=severity_scale or self.SEVERITY_SCALE)
        
    def apply(self, dataset: Dataset) -> Dataset:
        transform = T.ColorJitter(brightness=self.severity)
        return TransformedDataset(dataset, transform)

class Rotation(DistributionShift):
    """Rotates images with increasing angles"""
    SEVERITY_SCALE = [5, 10, 15, 30, 45]  # Degrees of rotation
    
    def __init__(self, severity_scale: List[float] = None):
        super().__init__(name="rotation", severity_scale=severity_scale or self.SEVERITY_SCALE)
        
    def apply(self, dataset: Dataset) -> Dataset:
        transform = T.RandomRotation(degrees=(self.severity, self.severity))
        return TransformedDataset(dataset, transform)

class HorizontalFlip(DistributionShift):
    """Binary shift - either flipped or not"""
    SEVERITY_SCALE = [1.0]  # Only one severity level for binary transforms
    
    def __init__(self, severity_scale: List[float] = None):
        super().__init__(name="horizontal_flip", severity_scale=severity_scale or self.SEVERITY_SCALE)
        
    def apply(self, dataset: Dataset) -> Dataset:
        transform = T.RandomHorizontalFlip(p=1.0)  # Always flip
        return TransformedDataset(dataset, transform)

class VerticalFlip(DistributionShift):
    """Binary shift - either flipped vertically or not"""
    SEVERITY_SCALE = [1.0]  # Only one severity level for binary transforms
    
    def __init__(self, severity_scale: List[float] = None):
        super().__init__(name="vertical_flip", severity_scale=severity_scale or self.SEVERITY_SCALE)
        
    def apply(self, dataset: Dataset) -> Dataset:
        transform = T.RandomVerticalFlip(p=1.0)  # Always flip
        return TransformedDataset(dataset, transform)

class ContrastShift(DistributionShift):
    """Adjusts contrast with increasing intensity"""
    SEVERITY_SCALE = [0.5, 0.75, 1.25, 1.5, 2.0]  # Values around 1.0 (original contrast)
    
    def __init__(self, severity_scale: List[float] = None):
        super().__init__(name="contrast", severity_scale=severity_scale or self.SEVERITY_SCALE)
        
    def apply(self, dataset: Dataset) -> Dataset:
        transform = T.ColorJitter(contrast=self.severity)
        return TransformedDataset(dataset, transform)

class SaturationShift(DistributionShift):
    """Adjusts saturation with increasing intensity"""
    SEVERITY_SCALE = [0.5, 0.75, 1.25, 1.5, 2.0]  # Values around 1.0 (original saturation)
    
    def __init__(self, severity_scale: List[float] = None):
        super().__init__(name="saturation", severity_scale=severity_scale or self.SEVERITY_SCALE)
        
    def apply(self, dataset: Dataset) -> Dataset:
        transform = T.ColorJitter(saturation=self.severity)
        return TransformedDataset(dataset, transform)

class HueShift(DistributionShift):
    """Adjusts hue with increasing intensity"""
    SEVERITY_SCALE = [0.05, 0.1, 0.15, 0.2, 0.25]  # Hue values in range [-0.5, 0.5]
    
    def __init__(self, severity_scale: List[float] = None):
        super().__init__(name="hue", severity_scale=severity_scale or self.SEVERITY_SCALE)
        
    def apply(self, dataset: Dataset) -> Dataset:
        transform = T.ColorJitter(hue=self.severity)
        return TransformedDataset(dataset, transform)

class GaussianBlur(DistributionShift):
    """Applies Gaussian blur with increasing kernel size"""
    SEVERITY_SCALE = [3, 5, 7, 9, 11]  # Kernel sizes (must be odd numbers)

    def __init__(self, severity_scale: List[float] = None):
        super().__init__(name="gaussian_blur", severity_scale=severity_scale or self.SEVERITY_SCALE)
        self._kernel_cache = {}

    def _get_gaussian_kernel(self, kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
        """Create cached 1D Gaussian kernel for separable convolution"""
        cache_key = (kernel_size, sigma, device)
        if cache_key not in self._kernel_cache:
            # Create 1D Gaussian kernel
            coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
            coords = coords - (kernel_size - 1) / 2
            kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            kernel = kernel / kernel.sum()
            self._kernel_cache[cache_key] = kernel
        return self._kernel_cache[cache_key]

    def apply(self, dataset: Dataset) -> Dataset:
        def blur_transform(img):
            kernel_size = int(self.severity)
            if kernel_size % 2 == 0:  # Ensure kernel size is odd
                kernel_size += 1

            # Use separable convolution for 3-4x speedup
            sigma = kernel_size / 6.0  # Better sigma calculation
            device = img.device
            kernel_1d = self._get_gaussian_kernel(kernel_size, sigma, device)

            # Reshape for conv2d: [out_channels, in_channels, height, width]
            kernel_h = kernel_1d.view(1, 1, kernel_size, 1)
            kernel_w = kernel_1d.view(1, 1, 1, kernel_size)

            # Apply separable convolution (horizontal then vertical)
            padding = kernel_size // 2
            img_unsqueezed = img.unsqueeze(0)  # Add batch dimension

            # Process each channel separately to maintain color
            channels = []
            for c in range(img.shape[0]):
                channel = img_unsqueezed[:, c:c+1, :, :]
                # Horizontal blur
                channel = torch.nn.functional.conv2d(channel, kernel_w, padding=(0, padding))
                # Vertical blur
                channel = torch.nn.functional.conv2d(channel, kernel_h, padding=(padding, 0))
                channels.append(channel)

            result = torch.cat(channels, dim=1).squeeze(0)
            return result
        return TransformedDataset(dataset, blur_transform)

class JPEGCompression(DistributionShift):
    """Applies JPEG compression artifacts with increasing compression (lower quality)"""
    SEVERITY_SCALE = [80, 60, 40, 20, 10]  # Quality values (higher = better quality)
    
    def __init__(self, severity_scale: List[float] = None):
        super().__init__(name="jpeg_compression", severity_scale=severity_scale or self.SEVERITY_SCALE)
        
    def apply(self, dataset: Dataset) -> Dataset:
        def compress(img):
            quality = int(self.severity)
            buffer = io.BytesIO()
            img_pil = F.to_pil_image(img)
            img_pil.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            img = F.to_tensor(Image.open(buffer))
            if img.shape[0] == 1:  # Handle grayscale
                return img
            return img
        return TransformedDataset(dataset, compress)

class Pixelation(DistributionShift):
    """Reduces image resolution then upsamples, creating pixelation"""
    SEVERITY_SCALE = [0.5, 0.25, 0.125, 0.0625, 0.03125]  # Scale factors for downsampling
    
    def __init__(self, severity_scale: List[float] = None):
        super().__init__(name="pixelation", severity_scale=severity_scale or self.SEVERITY_SCALE)
        
    def apply(self, dataset: Dataset) -> Dataset:
        def pixelate(img):
            c, h, w = img.shape
            scale = self.severity
            # Downsample and upsample
            down_h, down_w = int(h * scale), int(w * scale)
            if down_h < 1: down_h = 1
            if down_w < 1: down_w = 1
            
            downsized = F.resize(img, [down_h, down_w], interpolation=F.InterpolationMode.NEAREST)
            return F.resize(downsized, [h, w], interpolation=F.InterpolationMode.NEAREST)
        return TransformedDataset(dataset, pixelate)

class PerspectiveTransform(DistributionShift):
    """Applies perspective distortion with increasing intensity"""
    SEVERITY_SCALE = [0.05, 0.1, 0.15, 0.2, 0.3]  # Distortion scales
    
    def __init__(self, severity_scale: List[float] = None):
        super().__init__(name="perspective", severity_scale=severity_scale or self.SEVERITY_SCALE)
        
    def apply(self, dataset: Dataset) -> Dataset:
        def perspective(img):
            height, width = img.shape[-2:]
            startpoints = [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]]
            endpoints = [[width * self.severity, height * self.severity], 
                         [width * (1 - self.severity), height * self.severity],
                         [width * self.severity, height * (1 - self.severity)], 
                         [width * (1 - self.severity), height * (1 - self.severity)]]
            return F.perspective(img, startpoints, endpoints, interpolation=F.InterpolationMode.BILINEAR)
        return TransformedDataset(dataset, perspective)

class ZoomIn(DistributionShift):
    """Zooms in (crops and resizes) with increasing intensity"""
    SEVERITY_SCALE = [0.9, 0.8, 0.7, 0.6, 0.5]  # Smaller values = more zoom
    
    def __init__(self, severity_scale: List[float] = None):
        super().__init__(name="zoom_in", severity_scale=severity_scale or self.SEVERITY_SCALE)
        
    def apply(self, dataset: Dataset) -> Dataset:
        def zoom(img):
            c, h, w = img.shape
            scale = self.severity
            crop_size = int(min(h, w) * scale)
            
            # Use built-in transforms
            transform = T.Compose([
                T.CenterCrop(crop_size),
                T.Resize([h, w])
            ])
            
            return transform(img)
        return TransformedDataset(dataset, zoom)

class ZoomOut(DistributionShift):
    """Zooms out (resizes and pads) with increasing intensity"""
    SEVERITY_SCALE = [0.9, 0.8, 0.7, 0.6, 0.5]  # Smaller values = more zoom out (smaller center image)
    
    def __init__(self, severity_scale: List[float] = None):
        super().__init__(name="zoom_out", severity_scale=severity_scale or self.SEVERITY_SCALE)
        
    def apply(self, dataset: Dataset) -> Dataset:
        def zoom_out(img):
            c, h, w = img.shape
            scale = self.severity
            
            # First resize to smaller dimensions
            small_h, small_w = max(1, int(h * scale)), max(1, int(w * scale))
            
            # Create transform pipeline
            transform = T.Compose([
                T.Resize([small_h, small_w]),
                T.Pad(padding=[(w - small_w) // 2, (h - small_h) // 2, 
                              (w - small_w + 1) // 2, (h - small_h + 1) // 2], 
                      fill=0)
            ])
            
            return transform(img)
        return TransformedDataset(dataset, zoom_out)

class RandomErasing(DistributionShift):
    """Erases a random rectangle with increasing area"""
    SEVERITY_SCALE = [0.1, 0.2, 0.3, 0.4, 0.5]  # Increasing area to erase
    
    def __init__(self, severity_scale: List[float] = None):
        super().__init__(name="random_erasing", severity_scale=severity_scale or self.SEVERITY_SCALE)
        
    def apply(self, dataset: Dataset) -> Dataset:
        transform = T.RandomErasing(p=1.0, scale=(self.severity, self.severity), ratio=(0.3, 3.3), value=0)
        return TransformedDataset(dataset, transform)

class Grayscale(DistributionShift):
    """Converts to grayscale with increasing intensity"""
    SEVERITY_SCALE = [0.2, 0.4, 0.6, 0.8, 1.0]  # 1.0 = full grayscale
    
    def __init__(self, severity_scale: List[float] = None):
        super().__init__(name="grayscale", severity_scale=severity_scale or self.SEVERITY_SCALE)
        
    def apply(self, dataset: Dataset) -> Dataset:
        def to_grayscale(img):
            gray_img = F.rgb_to_grayscale(img, num_output_channels=img.shape[0])
            return self.severity * gray_img + (1 - self.severity) * img
        return TransformedDataset(dataset, to_grayscale)

class Sharpness(DistributionShift):
    """Adjusts sharpness with increasing intensity"""
    SEVERITY_SCALE = [0.5, 1.5, 2.5, 3.5, 5.0]  # Values around 1.0 (original sharpness)
    
    def __init__(self, severity_scale: List[float] = None):
        super().__init__(name="sharpness", severity_scale=severity_scale or self.SEVERITY_SCALE)
        
    def apply(self, dataset: Dataset) -> Dataset:
        def sharpen(img):
            return F.adjust_sharpness(img, self.severity)
        return TransformedDataset(dataset, sharpen)

class ColorJitter(DistributionShift):
    """Applies simultaneous brightness, contrast, saturation, and hue shifts"""
    SEVERITY_SCALE = [0.1, 0.2, 0.3, 0.4, 0.5]  # Controls overall jitter magnitude
    
    def __init__(self, severity_scale: List[float] = None):
        super().__init__(name="color_jitter", severity_scale=severity_scale or self.SEVERITY_SCALE)
        
    def apply(self, dataset: Dataset) -> Dataset:
        transform = T.ColorJitter(
            brightness=self.severity,
            contrast=self.severity, 
            saturation=self.severity, 
            hue=min(self.severity / 2, 0.5)  # Hue has a smaller range
        )
        return TransformedDataset(dataset, transform)

class ElasticTransform(DistributionShift):
    """Applies elastic deformation with increasing intensity"""
    SEVERITY_SCALE = [0.5, 1.0, 2.0, 3.0, 4.0]  # Alpha parameter for elastic transform
    
    def __init__(self, severity_scale: List[float] = None):
        super().__init__(name="elastic", severity_scale=severity_scale or self.SEVERITY_SCALE)
        
    def apply(self, dataset: Dataset) -> Dataset:
        transform = T.ElasticTransform(alpha=float(self.severity * 50), sigma=float(5))
        return TransformedDataset(dataset, transform)
        
class Solarize(DistributionShift):
    """Solarizes image (inverts pixels above threshold) with decreasing threshold"""
    SEVERITY_SCALE = [0.9, 0.7, 0.5, 0.3, 0.1]  # Decreasing thresholds (more solarization)
    
    def __init__(self, severity_scale: List[float] = None):
        super().__init__(name="solarize", severity_scale=severity_scale or self.SEVERITY_SCALE)
        
    def apply(self, dataset: Dataset) -> Dataset:
        def solarize_img(img):
            return F.solarize(img, threshold=self.severity)
        return TransformedDataset(dataset, solarize_img)

class Posterize(DistributionShift):
    """Reduces the number of bits for each color channel"""
    SEVERITY_SCALE = [6, 5, 4, 3, 2]  # Decreasing bits (more posterization)
    
    def __init__(self, severity_scale: List[float] = None):
        super().__init__(name="posterize", severity_scale=severity_scale or self.SEVERITY_SCALE)
        
    def apply(self, dataset: Dataset) -> Dataset:
        def posterize_img(img):
            return F.posterize(img, bits=int(self.severity))
        return TransformedDataset(dataset, posterize_img)

class MotionBlur(DistributionShift):
    """Applies motion blur with increasing kernel size"""
    SEVERITY_SCALE = [3, 5, 7, 9, 11]  # Kernel sizes
    
    def __init__(self, severity_scale: List[float] = None):
        super().__init__(name="motion_blur", severity_scale=severity_scale or self.SEVERITY_SCALE)
        
    def apply(self, dataset: Dataset) -> Dataset:
        def motion_blur(img):
            kernel_size = int(self.severity)
            if kernel_size % 2 == 0:  # Ensure kernel size is odd
                kernel_size += 1
                
            kernel = torch.zeros((kernel_size, kernel_size))
            kernel[kernel_size // 2, :] = 1.0 / kernel_size
            
            channels = []
            for c in range(img.shape[0]):
                # Apply convolution to create motion blur
                ch = img[c].unsqueeze(0).unsqueeze(0)
                weight = kernel.expand(1, 1, kernel_size, kernel_size)
                ch = torch.nn.functional.conv2d(ch, weight.to(ch.device), padding=kernel_size//2)
                channels.append(ch.squeeze())
            
            return torch.stack(channels)
        return TransformedDataset(dataset, motion_blur)

class IdentityShift(DistributionShift):
    """A shift that returns the original dataset without any modifications."""
    
    def __init__(self):
        super().__init__(name="identity", severity_scale=[0.0])
    
    def apply(self, dataset, severity: float = None) -> Any:
        """Return the original dataset without modifications.
        
        Args:
            dataset: The dataset to process
            severity: Ignored for identity shift
            
        Returns:
            The original dataset unchanged
        """
        return dataset

# Import these only if needed for JPEGCompression
try:
    import io
    from PIL import Image
except ImportError:
    pass 