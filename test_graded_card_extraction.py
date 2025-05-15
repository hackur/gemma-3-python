#!/usr/bin/env python
"""
Test script for extracting Pokemon cards from grading cases.

This script demonstrates the use of the extract_graded_card tool which:
1. Detects if a Pokemon card is in a professional grading case (PSA, BGS, CGC, etc.)
2. Extracts just the card itself by cropping away the plastic case
3. Separately extracts the grade label showing the numerical grade
"""

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

# Path to the graded Pokemon card images
# You'll need images of graded cards for this test
# The script will check common directories for sample images
SAMPLE_DIRS = [
    "docs/graded_cards",
    "docs",
    "samples/graded_cards",
    "samples"
]

def find_graded_card_images():
    """
    Find graded card images in common directories.
    
    Returns:
        List of Path objects to graded card images
    """
    image_files = []
    
    # Check common directories for graded card images
    for dir_path in SAMPLE_DIRS:
        dir_path = Path(dir_path)
        if dir_path.exists() and dir_path.is_dir():
            # Look for image files with common names suggesting graded cards
            for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                pattern = f"*{ext}"
                for img_path in dir_path.glob(pattern):
                    # Simple heuristic: look for filenames containing "graded", "psa", "bgs", or "cgc"
                    name_lower = img_path.name.lower()
                    if any(term in name_lower for term in ["graded", "psa", "bgs", "cgc", "slab"]):
                        image_files.append(img_path)
    
    return image_files

def create_output_dir():
    """
    Create a timestamped output directory for storing extraction results.
    
    Returns:
        Path: Path object pointing to the newly created output directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = Path(f"output/GRADED_CARDS--{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def image_to_data_uri(image_path):
    """
    Convert an image file to a data URI format.
    
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
    
    Args:
        data_uri: The data URI string to save (format: 'data:[MIME type];base64,[encoded data]')
        output_path: Path where the file should be saved
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    if data_uri and data_uri.startswith('data:'):
        # Extract the base64 data part from the data URI
        header, base64_data = data_uri.split(',', 1)
        image_data = base64.b64decode(base64_data)
        
        # Save the decoded binary data to the specified file
        with open(output_path, 'wb') as f:
            f.write(image_data)
        
        return True
    return False

async def extract_graded_card(image_path, output_dir):
    """
    Extract a card and grade label from a graded Pokemon card image.
    
    Args:
        image_path: Path to the graded card image file
        output_dir: Directory where the extracted images will be saved
        
    Returns:
        dict: Results of the extraction including paths to saved images
    """
    logger.info(f"Processing graded card: {image_path}")
    
    # Convert the image file to a data URI format for the API request
    image_data_uri = image_to_data_uri(image_path)
    
    # Call the extract_graded_card tool API endpoint
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SERVER_URL}/v1/tools/extract_graded_card",
            json={
                "image_url": image_data_uri
            }
        )
        
        if response.status_code == 200:
            # Process the successful response
            try:
                # Print the raw response for debugging
                response_text = response.text
                logger.info(f"Raw API Response: {response_text[:200]}...")
                
                # The response is a JSON string wrapped in quotes, so we need to:
                # 1. Strip the outer quotes if present
                # 2. Parse the resulting string as JSON
                if response_text.startswith('"') and response_text.endswith('"'):
                    # Remove the outer quotes
                    json_str = response_text[1:-1]
                    # Unescape any escaped quotes within the string
                    json_str = json_str.replace('\\"', '"')
                else:
                    json_str = response_text
                
                # Now parse the JSON string
                import json
                result = json.loads(json_str)
                
                # Get base filename without extension for output files
                base_filename = Path(image_path).stem
                
                outputs = {}
                
                # Save the visualization image if available
                if "visualization" in result:
                    viz_path = output_dir / f"{base_filename}_visualization.png"
                    if save_data_uri_to_file(result["visualization"], viz_path):
                        logger.info(f"Visualization image saved to: {viz_path}")
                        outputs["visualization_path"] = str(viz_path)
                
                # Save the extracted card image if available
                if "card_image" in result:
                    card_path = output_dir / f"{base_filename}_card.png"
                    if save_data_uri_to_file(result["card_image"], card_path):
                        logger.info(f"Extracted card image saved to: {card_path}")
                        outputs["card_path"] = str(card_path)
                
                # Save the grade label image if available and the card is graded
                if result.get("is_graded_card", False) and result.get("grade_label_image"):
                    label_path = output_dir / f"{base_filename}_grade_label.png"
                    if save_data_uri_to_file(result["grade_label_image"], label_path):
                        logger.info(f"Grade label image saved to: {label_path}")
                        outputs["label_path"] = str(label_path)
                
                # Add metadata from the result
                outputs["is_graded_card"] = result.get("is_graded_card", False)
                outputs["grade_type"] = result.get("grade_type")
                outputs["grade_value"] = result.get("grade_value")
                
                return outputs
                
            except Exception as e:
                logger.error(f"Error processing extraction result: {str(e)}")
                return None
        else:
            logger.error(f"Error: {response.status_code} - {response.text}")
            return None

async def main():
    """
    Main function to test graded Pokemon card extraction.
    """
    # Find graded card images to process
    image_files = find_graded_card_images()
    
    if not image_files:
        logger.warning(f"No graded card images found in any of the following directories: {SAMPLE_DIRS}")
        logger.info("Creating sample data with regular Pokemon card images...")
        
        # If no graded images were found, use any Pokemon card images we can find
        for dir_path in SAMPLE_DIRS:
            dir_path = Path(dir_path)
            if dir_path.exists() and dir_path.is_dir():
                for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                    pattern = f"*{ext}"
                    for img_path in dir_path.glob(pattern):
                        # Add any Pokemon card images
                        name_lower = img_path.name.lower()
                        if any(term in name_lower for term in ["pokemon", "card"]):
                            image_files.append(img_path)
                            if len(image_files) >= 2:  # Limit to 2 sample images
                                break
    
    if not image_files:
        logger.error("No suitable images found for testing. Please add some graded card images.")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Create output directory
    output_dir = create_output_dir()
    logger.info(f"Created output directory: {output_dir}")
    
    # Save a copy of this script to the output directory for reference
    with open(__file__, 'r') as f:
        script_content = f.read()
    
    with open(output_dir / "test_script.py", 'w') as f:
        f.write(script_content)
    
    # Process each image
    results = []
    for image_path in image_files:
        result = await extract_graded_card(image_path, output_dir)
        if result:
            result["input_image"] = str(image_path)
            results.append(result)
    
    # Create a summary file with all results
    summary = {
        "timestamp": datetime.now().isoformat(),
        "images_processed": len(image_files),
        "successful_extractions": len(results),
        "output_directory": str(output_dir),
        "results": results
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, indent=2, fp=f)
    
    logger.info(f"\nAll extractions completed. Results saved to: {output_dir}")
    logger.info(f"{len(results)} successful extractions out of {len(image_files)} images")

if __name__ == "__main__":
    asyncio.run(main())
