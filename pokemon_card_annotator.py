"""
Pokemon Card Annotator

This module provides functionality for annotating Pokemon cards with labels and bounding boxes,
while preserving original image dimensions and quality.

Features:
- Detection of card regions (name, image, type, HP, attacks)
- Annotation with colored boxes (rectangle or circle)
- Custom label colors (red, green, blue, yellow, white, black)
- Resolution-independent annotation that works with any card size
- Preservation of original image quality

Typical usage example:

    import asyncio
    from pokemon_card_annotator import annotate_pokemon_card
    
    async def add_annotations():
        result = await annotate_pokemon_card(
            image_url="https://example.com/pokemon_card.jpg",
            label_color="green",
            box_type="rectangle"
        )
        print(f"Annotations created for {len(result['annotations'])} regions")
        
    asyncio.run(add_annotations())
"""

import os
import base64
import logging
from io import BytesIO
from typing import Dict, List, Any, Optional, Union, Tuple

import httpx
from PIL import Image, ImageDraw, ImageFont

# Import shared utilities
from pokemon_card_utils import load_image, image_to_data_uri, scale_coordinates

# Configure logging
logger = logging.getLogger("pokemon_card_annotator")

async def annotate_pokemon_card(image_url: str, label_color: str = "red", 
                        box_type: str = "rectangle") -> Dict[str, Any]:
    """
    Annotate parts of a Pokemon card with labels and bounding boxes.
    
    This version preserves original image dimensions throughout the process.
    
    Args:
        image_url: URL or base64 data URI of the image
        label_color: Color for the annotation labels (red, green, blue, yellow, white, black)
        box_type: Type of bounding box (rectangle, circle)
        
    Returns:
        Dictionary containing annotated image and metadata
    """
    try:
        # Load the original image (this will preserve original dimensions)
        original_image = load_image(image_url)
        original_dimensions = original_image.size
        logger.info(f"Original image dimensions: {original_dimensions}")
        
        # Create a copy of the original image for annotation
        annotated_image = original_image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        # Map label color to RGB values
        color_map = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "white": (255, 255, 255),
            "black": (0, 0, 0)
        }
        
        rgb_color = color_map.get(label_color.lower(), (255, 0, 0))
        
        # Detect card regions (in a real implementation, this would use ML)
        # For this demo, we're using fixed regions based on the original dimensions
        width, height = original_dimensions
        
        # Define regions based on percentage of original dimensions to ensure scaling works
        regions = {
            "name": (int(width * 0.1), int(height * 0.05), int(width * 0.9), int(height * 0.12)),
            "image": (int(width * 0.1), int(height * 0.15), int(width * 0.9), int(height * 0.6)),
            "type": (int(width * 0.1), int(height * 0.65), int(width * 0.3), int(height * 0.7)),
            "hp": (int(width * 0.7), int(height * 0.05), int(width * 0.9), int(height * 0.12)),
            "attack": (int(width * 0.1), int(height * 0.75), int(width * 0.9), int(height * 0.85)),
            "description": (int(width * 0.1), int(height * 0.9), int(width * 0.9), int(height * 0.95))
        }
        
        # Load a font - try to use Arial, fall back to default if not available
        try:
            # Scale font size based on image dimensions
            font_size = int(min(width, height) * 0.03)  # 3% of the smaller dimension
            font = ImageFont.truetype("Arial", font_size)
        except IOError:
            # Fall back to default font
            font = ImageFont.load_default()
        
        # Draw annotations on the copied image
        for label, (x1, y1, x2, y2) in regions.items():
            # Draw the appropriate box type
            if box_type.lower() == "rectangle":
                draw.rectangle([(x1, y1), (x2, y2)], outline=rgb_color, width=max(1, int(min(width, height) * 0.005)))
            else:  # Default to circle/ellipse
                draw.ellipse([(x1, y1), (x2, y2)], outline=rgb_color, width=max(1, int(min(width, height) * 0.005)))
            
            # Draw the label above the box
            draw.text((x1, max(0, y1 - font_size - 5)), label, fill=rgb_color, font=font)
        
        # Convert the annotated image to a data URI
        annotated_data_uri = image_to_data_uri(annotated_image)
        
        # Return results
        return {
            "annotated_image": annotated_data_uri,
            "original_width": width,
            "original_height": height,
            "annotations": {
                name: {
                    "coordinates": coords,
                    "label": name
                } for name, coords in regions.items()
            },
            "annotation_color": label_color,
            "box_type": box_type
        }
    except Exception as e:
        logger.error(f"Error annotating Pokemon card: {str(e)}")
        return {
            "error": str(e)
        }
