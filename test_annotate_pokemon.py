#!/usr/bin/env python
"""
Test script for annotating Pokemon card images with bounding boxes and labels.

This script demonstrates the use of the annotate_pokemon_card tool to identify key parts
of Pokemon cards (both front and back) and annotate them with bounding boxes and labels.
It processes images with different annotation styles (colors and box types) and saves the
results to a timestamped output directory for easy comparison and reference.
"""

# Import necessary modules for file operations, JSON data handling, base64 encoding, 
# HTTP requests, asynchronous programming, logging, and path manipulation.
import os
import json
import base64
import httpx
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Configure logging for better debugging and tracking
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Server URL for the Gemma proxy server
SERVER_URL = "http://localhost:1338"

# Path to the Pokemon card images
# These should be real Pokemon card images stored in the docs directory
FRONT_CARD_PATH = Path("docs/pokenmon-card-front.webp")
BACK_CARD_PATH = Path("docs/pokenmon-card-back.webp.webp")

def create_output_dir():
    """
    Create a timestamped output directory for storing annotation results.
    
    The timestamp format ensures unique directory names for each test run,
    making it easy to compare results from different runs and configurations.
    
    Returns:
        Path: Path object pointing to the newly created output directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = Path(f"output/ANNOTATED_CARDS--{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def image_to_data_uri(image_path):
    """
    Convert an image file to a data URI format.
    
    This function reads an image file, determines its MIME type based on the file extension,
    and converts it to a base64-encoded data URI that can be used directly in HTTP requests
    or embedded in HTML/CSS.
    
    Args:
        image_path: Path to the image file to convert
        
    Returns:
        str: A data URI string in the format 'data:[MIME type];base64,[encoded data]'
    """
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    # Get the MIME type based on file extension
    if str(image_path).lower().endswith('.png'):
        mime_type = "image/png"
    elif str(image_path).lower().endswith('.jpg') or str(image_path).lower().endswith('.jpeg'):
        mime_type = "image/jpeg"
    elif str(image_path).lower().endswith('.webp'):
        mime_type = "image/webp"
    else:
        mime_type = "image/png"  # Default to PNG if extension is unknown
    
    # Encode the binary image data as base64
    base64_data = base64.b64encode(image_data).decode("utf-8")
    
    return f"data:{mime_type};base64,{base64_data}"

def save_data_uri_to_file(data_uri, output_path):
    """
    Save a data URI to a file on disk.
    
    This function extracts the binary data from a base64-encoded data URI
    and writes it to a file at the specified path.
    
    Args:
        data_uri: The data URI string to save (format: 'data:[MIME type];base64,[encoded data]')
        output_path: Path where the file should be saved
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    if data_uri.startswith('data:'):
        # Extract the base64 data part from the data URI
        header, base64_data = data_uri.split(',', 1)
        image_data = base64.b64decode(base64_data)
        
        # Save the decoded binary data to the specified file
        with open(output_path, 'wb') as f:
            f.write(image_data)
        
        return True
    return False

async def annotate_card(image_path, output_dir, is_back=False, label_color="red", box_type="rectangle"):
    """
    Annotate a Pokemon card image with bounding boxes and labels.
    
    This function sends a Pokemon card image to the annotate_pokemon_card tool API endpoint,
    which processes the image and returns an annotated version with bounding boxes and labels
    around key card elements. The annotated image is then saved to the specified output directory.
    
    Args:
        image_path: Path to the Pokemon card image file
        output_dir: Directory where the annotated image will be saved
        is_back: Boolean indicating if this is the back of a card (for logging purposes)
        label_color: Color to use for the annotations (red, green, blue, yellow, white, black)
        box_type: Type of bounding box to draw (rectangle or circle)
        
    Returns:
        Path: Path to the saved annotated image if successful, None otherwise
    """
    # Log which side of the card we're annotating for better tracking
    card_type = "back" if is_back else "front"
    logger.info(f"Annotating {card_type} of card: {image_path}")
    
    # Convert the image file to a data URI format for the API request
    image_data_uri = image_to_data_uri(image_path)
    
    # Call the annotate_pokemon_card tool API endpoint
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SERVER_URL}/v1/tools/annotate_pokemon_card",
            json={
                "image_url": image_data_uri,
                "label_color": label_color,
                "box_type": box_type
            }
        )
        
        if response.status_code == 200:
            # Process the successful response
            try:
                # Extract the annotated image data URI from the response
                # Strip quotes that might be included in the response
                annotated_image_uri = response.text.strip('"')
                
                # Generate a descriptive filename for the output
                image_name = Path(image_path).stem
                output_filename = f"{image_name}_annotated_{label_color}_{box_type}.png"
                output_path = output_dir / output_filename
                
                # Save the annotated image to the output directory
                if save_data_uri_to_file(annotated_image_uri, output_path):
                    logger.info(f"Annotated image saved to: {output_path}")
                    return output_path
                else:
                    logger.error(f"Failed to save annotated image")
                    return None
            except Exception as e:
                # Handle any errors during processing of the response
                logger.error(f"Error processing annotated image: {str(e)}")
                return None
        else:
            # Handle API error responses
            logger.error(f"Error: {response.status_code} - {response.text}")
            return None

async def main():
    """
    Main function to test Pokemon card annotation with various styles.
    
    This function orchestrates the entire testing process:
    1. Verifies that the required card images exist
    2. Creates a timestamped output directory
    3. Processes both front and back card images with different annotation styles
    4. Saves all annotated images to the output directory
    5. Creates a summary JSON file with metadata about all processed images
    
    The function tests multiple annotation configurations (combinations of colors and box types)
    to demonstrate the flexibility of the annotation tool.
    """
    # Check if the required card images exist before proceeding
    if not FRONT_CARD_PATH.exists():
        logger.error(f"Error: Front card image not found at {FRONT_CARD_PATH}")
        return
    
    if not BACK_CARD_PATH.exists():
        logger.error(f"Error: Back card image not found at {BACK_CARD_PATH}")
        return
    
    logger.info(f"Using Pokemon card images: Front={FRONT_CARD_PATH}, Back={BACK_CARD_PATH}")
    
    # Create a timestamped output directory for this test run
    output_dir = create_output_dir()
    logger.info(f"Created output directory: {output_dir}")
    
    # Save a copy of this script to the output directory for reproducibility
    with open(__file__, 'r') as f:
        script_content = f.read()
    
    with open(output_dir / "test_script.py", 'w') as f:
        f.write(script_content)
    
    # Define different annotation configurations to test
    # Each configuration is a combination of label color and box type
    annotation_configs = [
        # (label_color, box_type)
        ("red", "rectangle"),
        ("green", "rectangle"),
        ("blue", "circle"),
        ("yellow", "circle")
    ]
    
    # Process the front card with all annotation configurations
    front_annotations = []
    for color, box_type in annotation_configs:
        annotated_path = await annotate_card(FRONT_CARD_PATH, output_dir, is_back=False, 
                                            label_color=color, box_type=box_type)
        if annotated_path:
            front_annotations.append((annotated_path, color, box_type))
    
    # Process the back card with all annotation configurations
    back_annotations = []
    for color, box_type in annotation_configs:
        annotated_path = await annotate_card(BACK_CARD_PATH, output_dir, is_back=True, 
                                           label_color=color, box_type=box_type)
        if annotated_path:
            back_annotations.append((annotated_path, color, box_type))
    
    # Create a comprehensive summary file with metadata about all processed images
    summary = {
        "timestamp": datetime.now().isoformat(),
        "front_card": str(FRONT_CARD_PATH),
        "back_card": str(BACK_CARD_PATH),
        "output_directory": str(output_dir),
        "front_annotations": [{"path": str(path), "color": color, "box_type": box_type} 
                             for path, color, box_type in front_annotations],
        "back_annotations": [{"path": str(path), "color": color, "box_type": box_type} 
                            for path, color, box_type in back_annotations]
    }
    
    # Save the summary as a JSON file for easy parsing and reference
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, indent=2, fp=f)
    
    logger.info(f"\nAll annotations completed. Results saved to: {output_dir}")

if __name__ == "__main__":
    asyncio.run(main())
