#!/usr/bin/env python
"""
Test script for analyzing Pokemon card images with smart cropping.
This script uses the analyze_image and smart_crop_image tools with Gemma 3 recommended settings.
"""

import os
import json
import base64
import httpx
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Server URL
SERVER_URL = "http://localhost:1338"

# Path to the Pokemon card images
FRONT_CARD_PATH = Path("docs/pokenmon-card-front.webp")
BACK_CARD_PATH = Path("docs/pokenmon-card-back.webp.webp")

# Create output directory with timestamp
def create_output_dir(base_name):
    """Create a timestamped output directory"""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = Path(f"output/{base_name}--{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def image_to_data_uri(image_path):
    """Convert an image file to a data URI"""
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
        mime_type = "image/png"  # Default
    
    # Encode as base64
    base64_data = base64.b64encode(image_data).decode("utf-8")
    
    return f"data:{mime_type};base64,{base64_data}"

def save_data_uri_to_file(data_uri, output_path):
    """Save a data URI to a file"""
    if data_uri.startswith('data:'):
        # Extract the base64 data
        header, base64_data = data_uri.split(',', 1)
        image_data = base64.b64decode(base64_data)
        
        # Save to file
        with open(output_path, 'wb') as f:
            f.write(image_data)
        
        return True
    return False

async def analyze_pokemon_card(image_path, output_dir, is_front=True):
    """Analyze the Pokemon card image with consistent results"""
    card_type = "front" if is_front else "back"
    logger.info(f"Analyzing image: {image_path} ({card_type} of card)")
    
    # Convert image to data URI
    image_data_uri = image_to_data_uri(image_path)
    
    # Call the analyze_image tool with Gemma 3 recommended settings
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SERVER_URL}/v1/tools/analyze_image",
            json={
                "image_url": image_data_uri,
                "analyze_objects": True,
                "analyze_text": True,
                "temperature": 1.0,  # Gemma 3 recommended setting
                "top_k": 64,       # Gemma 3 recommended setting
                "top_p": 0.95,     # Gemma 3 recommended setting
                "min_p": 0.0,      # Gemma 3 recommended setting
                "seed": 42         # For reproducibility
            }
        )
        
        # Save the response to the output directory
        analysis_output_path = output_dir / f"analysis_{card_type}.json"
        
        if response.status_code == 200:
            # Parse the response
            result = response.text
            analysis_results = None
            
            try:
                # The response might be a JSON string
                if result.startswith('"'):
                    result = json.loads(result)
                
                # Try to parse as JSON if it's still a string
                if isinstance(result, str):
                    try:
                        analysis_results = json.loads(result)
                    except:
                        logger.error("Failed to parse JSON response")
                        with open(analysis_output_path, 'w') as f:
                            f.write(result)
                        return None
                else:
                    analysis_results = result
                
                # Save the analysis results
                with open(analysis_output_path, 'w') as f:
                    json.dump(analysis_results, indent=2, fp=f)
                
                # Print the raw analysis results
                logger.info(f"\nAnalysis Results saved to: {analysis_output_path}")
                
                # Format the results for easier reading
                logger.info("\n=== Pokemon Card Analysis ===")
                
                # Card detection results
                is_pokemon_card = analysis_results.get('metadata', {}).get('is_pokemon_card', False)
                is_card_back = analysis_results.get('metadata', {}).get('is_card_back', False)
                pokemon_visible = analysis_results.get('metadata', {}).get('pokemon_character_visible', False)
                
                if is_pokemon_card:
                    if is_card_back:
                        logger.info("✅ DETECTED: This is the BACK of a Pokemon card")
                    else:
                        logger.info("✅ DETECTED: This is the FRONT of a Pokemon card")
                        if pokemon_visible:
                            logger.info("✅ Pokemon character is visible on the card")
                        else:
                            logger.info("⚠️ No Pokemon character detected on the card")
                else:
                    logger.info("❌ This does not appear to be a Pokemon card")
                
                return analysis_results
                
            except Exception as e:
                logger.error(f"Error parsing result: {str(e)}")
                with open(analysis_output_path, 'w') as f:
                    f.write(result)
                return None
        else:
            logger.error(f"Error: {response.status_code} - {response.text}")
            return None

async def smart_crop_image(image_path, output_dir, target_width, target_height, focus_area="center", suffix=""):
    """Smart crop an image and save the result"""
    image_name = Path(image_path).stem
    logger.info(f"Smart cropping image: {image_path} to {target_width}x{target_height} focused on {focus_area}")
    
    # Convert image to data URI
    image_data_uri = image_to_data_uri(image_path)
    
    # Call the smart_crop_image tool
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SERVER_URL}/v1/tools/smart_crop_image",
            json={
                "image_url": image_data_uri,
                "target_width": target_width,
                "target_height": target_height,
                "focus_area": focus_area
            }
        )
        
        if response.status_code == 200:
            # Get the cropped image data URI
            cropped_image_uri = response.text.strip('"')
            
            # Save the cropped image
            output_filename = f"{image_name}_crop_{target_width}x{target_height}_{focus_area}{suffix}.png"
            output_path = output_dir / output_filename
            
            if save_data_uri_to_file(cropped_image_uri, output_path):
                logger.info(f"Cropped image saved to: {output_path}")
                return output_path
            else:
                logger.error(f"Failed to save cropped image")
                return None
        else:
            logger.error(f"Error: {response.status_code} - {response.text}")
            return None

async def main():
    """Main function to test Pokemon card analysis with smart cropping"""
    # Check if the images exist
    if not FRONT_CARD_PATH.exists():
        logger.error(f"Error: Front card image not found at {FRONT_CARD_PATH}")
        return
    
    if not BACK_CARD_PATH.exists():
        logger.error(f"Error: Back card image not found at {BACK_CARD_PATH}")
        return
    
    logger.info(f"Using Pokemon card images: Front={FRONT_CARD_PATH}, Back={BACK_CARD_PATH}")
    
    # Create output directory
    output_dir = create_output_dir("POKEMON_CARD")
    logger.info(f"Created output directory: {output_dir}")
    
    # Save a copy of this script to the output directory
    with open(__file__, 'r') as f:
        script_content = f.read()
    
    with open(output_dir / "test_script.py", 'w') as f:
        f.write(script_content)
    
    # Analyze the original front and back cards
    front_analysis = await analyze_pokemon_card(FRONT_CARD_PATH, output_dir, is_front=True)
    back_analysis = await analyze_pokemon_card(BACK_CARD_PATH, output_dir, is_front=False)
    
    # Smart crop the front card to focus on different areas
    crop_configs = [
        # (width, height, focus_area, suffix)
        (300, 400, "center", "_full"),
        (200, 200, "top", "_top"),
        (200, 200, "center", "_center"),
        (200, 200, "bottom", "_bottom"),
        (150, 100, "top", "_name"),  # For the Pokemon name
        (100, 100, "center", "_character"),  # For the Pokemon character
    ]
    
    cropped_images = []
    
    # Crop the front card
    for width, height, focus, suffix in crop_configs:
        cropped_path = await smart_crop_image(FRONT_CARD_PATH, output_dir, width, height, focus, f"_front{suffix}")
        if cropped_path:
            cropped_images.append((cropped_path, True))  # (path, is_front)
    
    # Crop the back card
    for width, height, focus, suffix in crop_configs[:3]:  # Only use the first 3 configs for back
        cropped_path = await smart_crop_image(BACK_CARD_PATH, output_dir, width, height, focus, f"_back{suffix}")
        if cropped_path:
            cropped_images.append((cropped_path, False))  # (path, is_front)
    
    # Analyze all the cropped images
    for cropped_path, is_front in cropped_images:
        await analyze_pokemon_card(cropped_path, output_dir, is_front=is_front)
    
    logger.info(f"\nAll analyses and crops completed. Results saved to: {output_dir}")
    
    # Create a summary file
    summary = {
        "timestamp": datetime.now().isoformat(),
        "front_card": str(FRONT_CARD_PATH),
        "back_card": str(BACK_CARD_PATH),
        "output_directory": str(output_dir),
        "cropped_images": [str(path) for path, _ in cropped_images],
        "front_analysis_file": "analysis_front.json",
        "back_analysis_file": "analysis_back.json"
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, indent=2, fp=f)

if __name__ == "__main__":
    asyncio.run(main())
