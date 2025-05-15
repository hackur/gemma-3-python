#!/usr/bin/env python
"""
Test script for analyzing a cropped Pokemon card image with consistent results.
This script uses the analyze_image tool with a low temperature and specific seed value.
"""

import os
import json
import base64
import httpx
import asyncio
from pathlib import Path

# Server URL
SERVER_URL = "http://localhost:1338"

# Path to the cropped Pokemon card image
IMAGE_PATH = Path("docs/pokemon-card-cropped-200x300-center.png")
if not IMAGE_PATH.exists():
    # Fallback to another cropped image
    IMAGE_PATH = Path("docs/pokemon-card-cropped-200x300-top.png")
    if not IMAGE_PATH.exists():
        # Fallback to any available cropped image
        for path in Path("docs").glob("pokemon-card-cropped-*.png"):
            IMAGE_PATH = path
            break

def image_to_data_uri(image_path):
    """Convert an image file to a data URI"""
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    # Get the MIME type
    mime_type = "image/png"  # For .png files
    
    # Encode as base64
    base64_data = base64.b64encode(image_data).decode("utf-8")
    
    return f"data:{mime_type};base64,{base64_data}"

async def analyze_pokemon_card(image_path):
    """Analyze the Pokemon card image with consistent results"""
    print(f"Analyzing image: {image_path}")
    
    # Convert image to data URI
    image_data_uri = image_to_data_uri(image_path)
    
    # Call the analyze_image tool with low temperature and specific seed
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SERVER_URL}/v1/tools/analyze_image",
            json={
                "image_url": image_data_uri,
                "analyze_objects": True,
                "analyze_text": True,
                "temperature": 0.1,  # Very low temperature for consistency
                "seed": 42  # Specific seed for reproducibility
            }
        )
        
        if response.status_code == 200:
            # Parse the response
            result = response.text
            try:
                # The response might be a JSON string
                if result.startswith('"'):
                    result = json.loads(result)
                
                # Try to parse as JSON if it's still a string
                if isinstance(result, str):
                    try:
                        analysis = json.loads(result)
                        print("\nAnalysis Results:")
                        print(json.dumps(analysis, indent=2))
                        # Use the parsed JSON for further processing
                        result = analysis
                    except:
                        print("\nAnalysis Results (raw):")
                        print(result)
                else:
                    print("\nAnalysis Results:")
                    print(json.dumps(result, indent=2))
                
                # Generate a human-readable description
                print("\n=== Pokemon Card Analysis ===")
                print("Here's what I can tell you about this Pokemon card:")
                
                if isinstance(result, dict):
                    print(f"Description: {result.get('description', 'Unknown')}")
                    print(f"Tags: {', '.join(result.get('tags', []))}")
                    print(f"Colors: {', '.join(result.get('colors', []))}")
                    
                    # Print objects detected
                    objects = result.get('objects', [])
                    if objects:
                        print("\nObjects detected:")
                        for obj in objects:
                            print(f"- {obj.get('name', 'Unknown')} (confidence: {obj.get('confidence', 0):.2f})")
                    
                    # Print text detected
                    text_items = result.get('text', [])
                    if text_items:
                        print("\nText detected:")
                        for text_item in text_items:
                            print(f"- {text_item.get('text', 'Unknown')} (confidence: {text_item.get('confidence', 0):.2f})")
                    
                    # Print metadata
                    metadata = result.get('metadata', {})
                    if metadata:
                        print("\nImage metadata:")
                        print(f"- Dimensions: {metadata.get('width', 'Unknown')}x{metadata.get('height', 'Unknown')}")
                        print(f"- Format: {metadata.get('format', 'Unknown')}")
                        print(f"- Aspect ratio: {metadata.get('aspect_ratio', 'Unknown')}")
                
                print("\nThis analysis was generated with:")
                print(f"- Temperature: {result.get('metadata', {}).get('temperature', 0.1)}")
                print(f"- Seed: {result.get('metadata', {}).get('seed', 42)}")
                print(f"- Analysis time: {result.get('analysis_time', 'Unknown')}")
            except Exception as e:
                print(f"Error parsing result: {str(e)}")
                print("Raw result:", result)
        else:
            print(f"Error: {response.status_code} - {response.text}")

async def main():
    """Main function"""
    if not IMAGE_PATH.exists():
        print(f"Error: Image file not found: {IMAGE_PATH}")
        return
    
    await analyze_pokemon_card(IMAGE_PATH)

if __name__ == "__main__":
    asyncio.run(main())
