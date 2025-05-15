"""
Example tools for the Gemma 3 Tool Calling Framework

This module provides example tools that can be registered with the ToolRegistry
and executed by the ToolExecutor.
"""

import os
import base64
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

import httpx
from PIL import Image, ImageFilter, ImageEnhance
from io import BytesIO

# Configure logging
logger = logging.getLogger("example_tools")

# Image tools

async def analyze_image(image_url: str, analyze_objects: bool = True, analyze_text: bool = False) -> Dict[str, Any]:
    """
    Analyze an image and return information about its contents
    
    Args:
        image_url: URL or base64 data URI of the image to analyze
        analyze_objects: Whether to analyze objects in the image
        analyze_text: Whether to analyze text in the image
        
    Returns:
        Dictionary containing analysis results
    """
    # For demo purposes, return mock data
    # In a real implementation, this would call a vision API
    return {
        "description": "An image containing nature scenery with mountains and a lake",
        "tags": ["nature", "mountains", "lake", "scenic", "outdoors"],
        "colors": ["blue", "green", "white", "brown"],
        "objects": [
            {"name": "mountain", "confidence": 0.95, "bounding_box": {"x": 100, "y": 50, "width": 300, "height": 200}},
            {"name": "lake", "confidence": 0.92, "bounding_box": {"x": 50, "y": 300, "width": 400, "height": 150}},
            {"name": "tree", "confidence": 0.88, "bounding_box": {"x": 450, "y": 100, "width": 100, "height": 300}}
        ] if analyze_objects else [],
        "text": [
            {"text": "Mountain View", "confidence": 0.85, "bounding_box": {"x": 200, "y": 50, "width": 150, "height": 30}}
        ] if analyze_text else [],
        "analysis_time": datetime.now().isoformat()
    }

def apply_image_filter(
    image_url: str, 
    filter_type: str = "blur", 
    intensity: float = 1.0
) -> str:
    """
    Apply a filter to an image and return the processed image
    
    Args:
        image_url: URL or base64 data URI of the image to process
        filter_type: Type of filter to apply (blur, sharpen, grayscale, etc.)
        intensity: Intensity of the filter effect (0.0 to 2.0)
        
    Returns:
        Base64-encoded data URI of the processed image
    """
    # Load the image
    image = _load_image(image_url)
    
    # Apply the filter
    if filter_type == "blur":
        # Apply a blur filter
        filtered_image = image.filter(ImageFilter.GaussianBlur(radius=intensity * 2))
    elif filter_type == "sharpen":
        # Apply a sharpen filter
        filtered_image = image.filter(ImageFilter.SHARPEN)
        # Apply multiple times based on intensity
        for _ in range(int(intensity)):
            filtered_image = filtered_image.filter(ImageFilter.SHARPEN)
    elif filter_type == "grayscale":
        # Convert to grayscale
        filtered_image = image.convert("L")
    elif filter_type == "edge_enhance":
        # Enhance edges
        filtered_image = image.filter(ImageFilter.EDGE_ENHANCE)
        # Apply multiple times based on intensity
        for _ in range(int(intensity)):
            filtered_image = filtered_image.filter(ImageFilter.EDGE_ENHANCE)
    elif filter_type == "brighten":
        # Adjust brightness
        enhancer = ImageEnhance.Brightness(image)
        filtered_image = enhancer.enhance(1.0 + intensity)
    else:
        # Unknown filter type
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    # Convert the processed image to base64
    buffer = BytesIO()
    filtered_image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return f"data:image/png;base64,{base64_image}"

def resize_image(
    image_url: str, 
    width: Optional[int] = None, 
    height: Optional[int] = None, 
    maintain_aspect_ratio: bool = True
) -> str:
    """
    Resize an image to the specified dimensions
    
    Args:
        image_url: URL or base64 data URI of the image to resize
        width: Target width in pixels
        height: Target height in pixels
        maintain_aspect_ratio: Whether to maintain the aspect ratio
        
    Returns:
        Base64-encoded data URI of the resized image
    """
    # Load the image
    image = _load_image(image_url)
    
    # Get original dimensions
    original_width, original_height = image.size
    
    # Calculate new dimensions
    if width and height:
        new_width, new_height = width, height
        if maintain_aspect_ratio:
            # Calculate the scaling factor
            width_ratio = width / original_width
            height_ratio = height / original_height
            
            # Use the smaller ratio to ensure the image fits within the bounds
            ratio = min(width_ratio, height_ratio)
            
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
    elif width:
        # Only width specified
        if maintain_aspect_ratio:
            ratio = width / original_width
            new_width = width
            new_height = int(original_height * ratio)
        else:
            new_width = width
            new_height = original_height
    elif height:
        # Only height specified
        if maintain_aspect_ratio:
            ratio = height / original_height
            new_width = int(original_width * ratio)
            new_height = height
        else:
            new_width = original_width
            new_height = height
    else:
        # No dimensions specified, return original
        raise ValueError("Either width or height must be specified")
    
    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Convert the resized image to base64
    buffer = BytesIO()
    resized_image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return f"data:image/png;base64,{base64_image}"

# Utility tools

async def fetch_url_content(url: str) -> Dict[str, Any]:
    """
    Fetch content from a URL
    
    Args:
        url: URL to fetch
        
    Returns:
        Dictionary containing content type and data
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        
        content_type = response.headers.get("content-type", "text/plain")
        
        if "application/json" in content_type:
            return {
                "content_type": "json",
                "content": response.json()
            }
        elif "text/" in content_type or "application/xml" in content_type:
            return {
                "content_type": "text",
                "content": response.text
            }
        elif "image/" in content_type:
            # Return base64-encoded image
            base64_data = base64.b64encode(response.content).decode("utf-8")
            return {
                "content_type": "image",
                "content": f"data:{content_type};base64,{base64_data}"
            }
        else:
            # Other binary data
            return {
                "content_type": "binary",
                "content": f"Binary data of type {content_type}, {len(response.content)} bytes"
            }

def get_current_time() -> Dict[str, Any]:
    """
    Get the current date and time in different formats
    
    Returns:
        Dictionary containing current time in different formats
    """
    now = datetime.now()
    
    return {
        "iso": now.isoformat(),
        "unix_timestamp": int(now.timestamp()),
        "formatted": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "utc": datetime.utcnow().isoformat()
    }

# Helper functions

def _load_image(image_url: str) -> Image.Image:
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
