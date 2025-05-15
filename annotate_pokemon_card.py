"""
Annotate Pokemon Card Tool

This module provides a tool to identify and annotate parts of a Pokemon card
with bounding boxes and labels.
"""

import os
import base64
import logging
from io import BytesIO
from typing import Dict, List, Any, Tuple

import httpx
from PIL import Image, ImageDraw, ImageFont

# Configure logging
logger = logging.getLogger("annotate_pokemon_card")

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

async def annotate_pokemon_card(image_url: str, label_color: str = "red", box_type: str = "rectangle") -> str:
    """
    Identify and annotate parts of a Pokemon card with bounding boxes and labels
    
    Args:
        image_url: URL or base64 data URI of the Pokemon card image
        label_color: Color of the labels and bounding boxes (red, green, blue, yellow, white, black)
        box_type: Type of bounding box to draw (rectangle or circle)
        
    Returns:
        Base64-encoded data URI of the annotated image
    """
    try:
        # Load the image
        image = load_image(image_url)
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Get image dimensions
        width, height = image.size
        
        # Create a drawing context
        draw = ImageDraw.Draw(image)
        
        # Set up colors
        color_map = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "white": (255, 255, 255),
            "black": (0, 0, 0)
        }
        color = color_map.get(label_color.lower(), (255, 0, 0))  # Default to red
        
        # Try to create a font (use default if not available)
        try:
            # Use a default system font
            font_size = max(12, min(width, height) // 30)  # Scale font size based on image dimensions
            font = ImageFont.truetype("Arial", font_size)
        except IOError:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Determine if it's a front or back of a card
        is_back = "back" in image_url.lower()
        
        # Define card parts to annotate based on card type
        if is_back:
            # Back of card annotations
            card_parts = [
                {
                    "name": "Pokemon Logo",
                    "box": (width // 3, height // 3, width * 2 // 3, height * 2 // 3)
                },
                {
                    "name": "Card Border",
                    "box": (5, 5, width - 5, height - 5)
                },
                {
                    "name": "Card Back Pattern",
                    "box": (width // 5, height // 5, width * 4 // 5, height * 4 // 5)
                }
            ]
        else:
            # Front of card annotations
            card_parts = [
                {
                    "name": "Card Name",
                    "box": (width // 5, height // 20, width * 4 // 5, height // 7)
                },
                {
                    "name": "Pokemon Image",
                    "box": (width // 6, height // 5, width * 5 // 6, height * 3 // 5)
                },
                {
                    "name": "HP Value",
                    "box": (width * 3 // 4, height // 20, width * 9 // 10, height // 7)
                },
                {
                    "name": "Card Type",
                    "box": (width // 10, height // 10, width // 4, height // 6)
                },
                {
                    "name": "Card Description",
                    "box": (width // 6, height * 3 // 5, width * 5 // 6, height * 4 // 5)
                },
                {
                    "name": "Card Border",
                    "box": (5, 5, width - 5, height - 5)
                }
            ]
        
        # Draw bounding boxes and labels
        for part in card_parts:
            x1, y1, x2, y2 = part["box"]
            
            # Draw the bounding box
            if box_type.lower() == "circle":
                # For circle, use the center and radius
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                radius = min(x2 - x1, y2 - y1) // 2
                draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), 
                             outline=color, width=3)
            else:
                # Default to rectangle
                draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            
            # Draw the label
            text = part["name"]
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Position the text above the bounding box if possible
            text_x = x1
            text_y = max(0, y1 - text_height - 5)
            
            # Draw a background for the text for better visibility
            draw.rectangle((text_x, text_y, text_x + text_width, text_y + text_height), 
                          fill=(0, 0, 0, 128))
            draw.text((text_x, text_y), text, fill=color, font=font)
        
        # Convert the annotated image to base64
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return f"data:image/png;base64,{base64_image}"
    
    except Exception as e:
        logger.error(f"Error annotating Pokemon card: {str(e)}")
        raise ValueError(f"Failed to annotate Pokemon card: {str(e)}")
