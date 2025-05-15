"""
Pokemon Card Utilities

This module provides shared utility functions used across all Pokemon card analysis tools.
Functions include image loading, data URI conversion, coordinate scaling, and color analysis.

Typical usage example:

    from pokemon_card_utils import load_image, image_to_data_uri

    # Load an image from a URL or data URI
    image = load_image("https://example.com/pokemon_card.jpg")

    # Convert an image to a data URI for API requests
    data_uri = image_to_data_uri(image)

    # Scale coordinates between different image dimensions
    new_coords = scale_coordinates((10, 20, 50, 70), (100, 100), (200, 200))
"""

import os
import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import httpx
from PIL import Image, ImageDraw, ImageFont

# Configure logging
logger = logging.getLogger("pokemon_card_utils")

def load_image(image_url: str) -> Image.Image:
    """
    Load an image from a URL or base64 data URI
    
    Args:
        image_url: URL or base64 data URI of the image
        
    Returns:
        PIL Image object
    """
    if image_url.startswith("data:image/"):
        # Extract base64 data from data URI
        header, base64_data = image_url.split(",", 1)
        image_data = base64.b64decode(base64_data)
        return Image.open(BytesIO(image_data))
    else:
        # Download from URL
        response = httpx.get(image_url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))

def image_to_data_uri(image_or_path: Union[str, Path, Image.Image], format: str = "PNG") -> str:
    """
    Convert an image (file path or PIL Image) to a data URI
    
    Args:
        image_or_path: Image file path (str or Path) or PIL Image object
        format: Image format (PNG, JPEG, etc.)
        
    Returns:
        Data URI string
    """
    if isinstance(image_or_path, (str, Path)):
        # Convert Path to string if needed
        path_str = str(image_or_path)
        # Load image from path
        with Image.open(path_str) as img:
            img_format = format or img.format or "PNG"
            buffer = BytesIO()
            img.save(buffer, format=img_format)
            img_bytes = buffer.getvalue()
    else:
        # Already a PIL Image
        buffer = BytesIO()
        image_or_path.save(buffer, format=format)
        img_bytes = buffer.getvalue()
    
    # Convert to base64
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    mime_type = f"image/{format.lower()}" 
    return f"data:{mime_type};base64,{img_base64}"

def scale_coordinates(coords: Tuple[int, int, int, int], 
                     from_dimensions: Tuple[int, int], 
                     to_dimensions: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Scale coordinates from one image dimension to another.
    
    Args:
        coords: (x1, y1, x2, y2) coordinates to scale
        from_dimensions: (width, height) of the source image
        to_dimensions: (width, height) of the target image
        
    Returns:
        Scaled coordinates (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = coords
    from_width, from_height = from_dimensions
    to_width, to_height = to_dimensions
    
    # Calculate scaling factors
    width_scale = to_width / from_width
    height_scale = to_height / from_height
    
    # Apply scaling
    new_x1 = int(x1 * width_scale)
    new_y1 = int(y1 * height_scale)
    new_x2 = int(x2 * width_scale)
    new_y2 = int(y2 * height_scale)
    
    return (new_x1, new_y1, new_x2, new_y2)

def crop_image(image: Image.Image, coords: Tuple[int, int, int, int]) -> Image.Image:
    """
    Crop an image using the specified coordinates
    
    Args:
        image: PIL Image object
        coords: (x1, y1, x2, y2) coordinates to crop
        
    Returns:
        Cropped PIL Image
    """
    # Ensure coordinates are within bounds
    x1, y1, x2, y2 = coords
    width, height = image.size
    
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    
    # Create a new image by cropping
    return image.crop((x1, y1, x2, y2))

def get_dominant_colors(image: Image.Image, num_colors: int = 5) -> List[Tuple[int, int, int]]:
    """
    Get the dominant colors in an image
    
    Args:
        image: PIL Image object
        num_colors: Number of dominant colors to return
        
    Returns:
        List of (R, G, B) tuples
    """
    # Reduce image size for faster processing if it's large
    max_dimension = 200
    temp_img = image
    if max(image.size) > max_dimension:
        scaling_factor = max_dimension / max(image.size)
        new_size = (int(image.size[0] * scaling_factor), 
                   int(image.size[1] * scaling_factor))
        temp_img = image.resize(new_size, Image.LANCZOS)
    
    # Convert to RGB if not already
    if temp_img.mode != 'RGB':
        temp_img = temp_img.convert('RGB')
    
    # Get color counts
    color_counts = {}
    for pixel in list(temp_img.getdata()):
        if pixel in color_counts:
            color_counts[pixel] += 1
        else:
            color_counts[pixel] = 1
    
    # Sort by frequency
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    return [color for color, count in sorted_colors[:num_colors]]
