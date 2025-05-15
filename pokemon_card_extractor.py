"""
Pokemon Card Extractor

This module provides functionality for extracting cards from graded slabs
and performing operations on graded Pokemon cards.
"""

import os
import base64
import logging
import numpy as np
from io import BytesIO
from typing import Dict, List, Any, Optional, Union, Tuple

import httpx
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Import shared utilities
from pokemon_card_utils import load_image, image_to_data_uri, scale_coordinates, get_dominant_colors, crop_image

# Configure logging
logger = logging.getLogger("pokemon_card_extractor")

async def extract_graded_card(image_url: str) -> Dict[str, Any]:
    """
    Extract a Pokemon card and grade label from a graded card case.
    
    This version preserves original image dimensions throughout the process.
    
    Args:
        image_url: URL or base64 data URI of the image
        
    Returns:
        Dictionary containing extracted card, grade label, and metadata
    """
    try:
        # Load the original image (preserving dimensions)
        original_image = load_image(image_url)
        original_dimensions = original_image.size
        original_width, original_height = original_dimensions
        logger.info(f"Original image dimensions: {original_dimensions}")
        
        # Store original image for later use in cropping
        # We'll work with a copy for detection to avoid modifying the original
        working_image = original_image.copy()
        
        # Create a visualization image for debugging/display
        visualization = original_image.copy()
        viz_draw = ImageDraw.Draw(visualization)
        
        # Detect if this is a graded card (PSA, BGS, etc.) based on color patterns
        # For this demonstration, use color analysis to attempt detection
        dominant_colors = get_dominant_colors(working_image)
        
        # Define color signatures for different grading companies
        # These are approximate RGB signatures for common grading companies
        psa_red = (187, 30, 16)  # PSA red color
        bgs_blue = (0, 83, 165)  # BGS blue color
        cgc_blue = (34, 85, 150)  # CGC blue color
        
        # Check if any of the dominant colors are close to the grading company colors
        is_psa = any(abs(c[0] - psa_red[0]) < 30 and abs(c[1] - psa_red[1]) < 30 and abs(c[2] - psa_red[2]) < 30 
                    for c in dominant_colors)
        is_bgs = any(abs(c[0] - bgs_blue[0]) < 30 and abs(c[1] - bgs_blue[1]) < 30 and abs(c[2] - bgs_blue[2]) < 30 
                    for c in dominant_colors)
        is_cgc = any(abs(c[0] - cgc_blue[0]) < 30 and abs(c[1] - cgc_blue[1]) < 30 and abs(c[2] - cgc_blue[2]) < 30 
                    for c in dominant_colors)
        
        # Fallback - check for typical graded card aspect ratio and features
        # Most graded cards follow a certain width/height ratio
        aspect_ratio = original_width / original_height
        common_graded_ratio = 2.5/3.5  # Approximate ratio of PSA/BGS cases
        ratio_match = abs(aspect_ratio - common_graded_ratio) < 0.2
        
        # Combine signals to determine if this is likely a graded card
        is_graded_card = is_psa or is_bgs or is_cgc or ratio_match
        
        # If not detected as graded, return the original image
        if not is_graded_card:
            logger.info("No graded card detected")
            return {
                "is_graded_card": False,
                "card_image": image_to_data_uri(original_image),
                "visualization": image_to_data_uri(visualization),
                "grade_type": "Unknown",
                "grade_value": None
            }
            
        # Determine grading company
        if is_psa:
            grade_type = "PSA"
        elif is_bgs:
            grade_type = "BGS"
        elif is_cgc:
            grade_type = "CGC"
        else:
            grade_type = "Unknown"
            
        logger.info(f"Detected graded card: {grade_type}")
        
        # Extract card from the graded case
        # For graded cards, the actual card is typically inset from the edges
        # We'll calculate percentages of the original dimensions to get proper crops
        
        # Card region is typically the center portion of the slab
        # These percentages are based on typical PSA/BGS layouts
        card_left_pct = 0.12  # 12% from left edge
        card_top_pct = 0.15   # 15% from top
        card_right_pct = 0.88  # 88% from left (12% from right)
        card_bottom_pct = 0.75  # 75% from top (25% from bottom)
        
        # Calculate actual pixel coordinates from percentages of original dimensions
        card_x1 = int(original_width * card_left_pct)
        card_y1 = int(original_height * card_top_pct)
        card_x2 = int(original_width * card_right_pct)
        card_y2 = int(original_height * card_bottom_pct)
        
        # Add card region to visualization
        viz_draw.rectangle([(card_x1, card_y1), (card_x2, card_y2)], outline=(0, 255, 0), width=3)
        viz_draw.text((card_x1, card_y1 - 20), "Card", fill=(0, 255, 0), font=ImageFont.load_default())
        
        # Extract the card from the original image (not a resized working copy)
        card_coords = (card_x1, card_y1, card_x2, card_y2)
        card_image = crop_image(original_image, card_coords)
        
        # Calculate grade label region based on grading company
        if grade_type == "PSA":
            # PSA grade is typically at bottom center
            label_left_pct = 0.35
            label_top_pct = 0.78
            label_right_pct = 0.65
            label_bottom_pct = 0.88
        elif grade_type == "BGS":
            # BGS grade is typically at bottom right
            label_left_pct = 0.60
            label_top_pct = 0.78
            label_right_pct = 0.90
            label_bottom_pct = 0.88
        else:
            # Default location for other grading companies
            label_left_pct = 0.40
            label_top_pct = 0.78
            label_right_pct = 0.60
            label_bottom_pct = 0.88
        
        # Calculate label coordinates from percentages
        label_x1 = int(original_width * label_left_pct)
        label_y1 = int(original_height * label_top_pct)
        label_x2 = int(original_width * label_right_pct)
        label_y2 = int(original_height * label_bottom_pct)
        
        # Add label region to visualization
        viz_draw.rectangle([(label_x1, label_y1), (label_x2, label_y2)], outline=(255, 0, 0), width=3)
        viz_draw.text((label_x1, label_y1 - 20), "Grade Label", fill=(255, 0, 0), font=ImageFont.load_default())
        
        # Extract grade label from original image
        label_coords = (label_x1, label_y1, label_x2, label_y2)
        label_image = crop_image(original_image, label_coords)
        
        # In production, here we'd run OCR on the label_image to extract the grade value
        # For demonstration, we'll set a dummy value
        grade_value = None
        
        # Convert images to data URIs
        card_base64 = image_to_data_uri(card_image)
        label_base64 = image_to_data_uri(label_image)
        viz_base64 = image_to_data_uri(visualization)
        
        # Return the results
        return {
            "is_graded_card": True,
            "card_image": card_base64, 
            "grade_label_image": label_base64,
            "visualization": viz_base64,
            "grade_type": grade_type,
            "grade_value": grade_value,
            "original_dimensions": original_dimensions,
            "card_crop_coordinates": card_coords,
            "label_crop_coordinates": label_coords
        }
        
    except Exception as e:
        logger.error(f"Error extracting graded card: {str(e)}")
        return {
            "error": str(e),
            "is_graded_card": False
        }
