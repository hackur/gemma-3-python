#!/usr/bin/env python
"""
Test script for image tools using the Pokemon card image.
This script tests all the image-related tools registered with the Gemma proxy server.
"""

import os
import base64
import json
import httpx
import asyncio
from pathlib import Path

# Image path
IMAGE_PATH = Path("docs/pokemon-card.webp")

# Server URL
SERVER_URL = "http://localhost:1338"

def image_to_data_uri(image_path):
    """Convert an image file to a data URI"""
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    # Get the MIME type
    mime_type = "image/webp"  # For .webp files
    
    # Encode as base64
    base64_data = base64.b64encode(image_data).decode("utf-8")
    
    return f"data:{mime_type};base64,{base64_data}"

async def test_analyze_image(image_data_uri):
    """Test the analyze_image tool"""
    print("\n=== Testing analyze_image ===")
    
    payload = {
        "model": "gemma-3-4b-it",
        "messages": [
            {"role": "user", "content": "Analyze this Pokemon card image"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "analyze_image",
                    "description": "Analyze an image and return information about its contents",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_url": {
                                "type": "string",
                                "description": "URL or base64 data URI of the image to analyze"
                            },
                            "analyze_objects": {
                                "type": "boolean",
                                "description": "Whether to analyze objects in the image"
                            },
                            "analyze_text": {
                                "type": "boolean",
                                "description": "Whether to analyze text in the image"
                            }
                        },
                        "required": ["image_url"]
                    }
                }
            }
        ],
        "tool_choice": {
            "type": "function",
            "function": {
                "name": "analyze_image"
            }
        }
    }
    
    # Call the tool directly
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SERVER_URL}/v1/tools/analyze_image",
            json={"image_url": image_data_uri, "analyze_objects": True, "analyze_text": True}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.status_code} - {response.text}")

async def test_apply_image_filter(image_data_uri):
    """Test the apply_image_filter tool with different filters"""
    print("\n=== Testing apply_image_filter ===")
    
    filters = ["blur", "sharpen", "grayscale", "edge_enhance", "brighten"]
    
    for filter_type in filters:
        print(f"\nApplying filter: {filter_type}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SERVER_URL}/v1/tools/apply_image_filter",
                json={
                    "image_url": image_data_uri,
                    "filter_type": filter_type,
                    "intensity": 1.0
                }
            )
            
            if response.status_code == 200:
                result = response.json() if isinstance(response.content, dict) else response.text
                print(f"Filter {filter_type} applied successfully")
                
                # Save the filtered image
                try:
                    # The response might be a JSON string containing the data URI
                    if isinstance(result, str):
                        # Check if it's a JSON string
                        if result.startswith('"data:image'):
                            # Remove quotes from JSON string
                            data_uri = json.loads(result)
                        else:
                            data_uri = result
                        
                        if data_uri.startswith("data:image"):
                            # Extract base64 data
                            _, base64_data = data_uri.split(",", 1)
                            image_data = base64.b64decode(base64_data)
                            
                            # Save to file
                            output_path = f"docs/pokemon-card-{filter_type}.png"
                            with open(output_path, "wb") as f:
                                f.write(image_data)
                            print(f"Saved filtered image to {output_path}")
                            continue
                    
                    print(f"Error: Could not extract data URI from result: {result[:100]}...")
                except Exception as e:
                    print(f"Error saving image: {str(e)}")
            else:
                print(f"Error: {response.status_code} - {response.text}")

async def test_resize_image(image_data_uri):
    """Test the resize_image tool"""
    print("\n=== Testing resize_image ===")
    
    # Test different resize options
    resize_options = [
        {"width": 200, "height": 300, "maintain_aspect_ratio": True},
        {"width": 400, "maintain_aspect_ratio": True},
        {"height": 200, "maintain_aspect_ratio": True}
    ]
    
    for i, options in enumerate(resize_options):
        print(f"\nResizing with options: {options}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SERVER_URL}/v1/tools/resize_image",
                json={"image_url": image_data_uri, **options}
            )
            
            if response.status_code == 200:
                result = response.json() if isinstance(response.content, dict) else response.text
                print(f"Resize operation {i+1} completed successfully")
                
                # Save the resized image
                try:
                    # The response might be a JSON string containing the data URI
                    if isinstance(result, str):
                        # Check if it's a JSON string
                        if result.startswith('"data:image'):
                            # Remove quotes from JSON string
                            data_uri = json.loads(result)
                        else:
                            data_uri = result
                        
                        if data_uri.startswith("data:image"):
                            # Extract base64 data
                            _, base64_data = data_uri.split(",", 1)
                            image_data = base64.b64decode(base64_data)
                            
                            # Save to file
                            width = options.get("width", "auto")
                            height = options.get("height", "auto")
                            output_path = f"docs/pokemon-card-resized-{width}x{height}.png"
                            with open(output_path, "wb") as f:
                                f.write(image_data)
                            print(f"Saved resized image to {output_path}")
                            continue
                    
                    print(f"Error: Could not extract data URI from result: {result[:100]}...")
                except Exception as e:
                    print(f"Error saving image: {str(e)}")
            else:
                print(f"Error: {response.status_code} - {response.text}")

async def test_smart_crop_image(image_data_uri):
    """Test the smart_crop_image tool"""
    print("\n=== Testing smart_crop_image ===")
    
    # Test different crop options
    crop_options = [
        {"target_width": 300, "target_height": 300, "focus_area": "center"},
        {"target_width": 200, "target_height": 300, "focus_area": "top"},
        {"target_width": 200, "target_height": 300, "focus_area": "bottom"}
    ]
    
    for i, options in enumerate(crop_options):
        print(f"\nCropping with options: {options}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SERVER_URL}/v1/tools/smart_crop_image",
                json={"image_url": image_data_uri, **options}
            )
            
            if response.status_code == 200:
                result = response.json() if isinstance(response.content, dict) else response.text
                print(f"Smart crop operation {i+1} completed successfully")
                
                # Save the cropped image
                try:
                    # The response might be a JSON string containing the data URI
                    if isinstance(result, str):
                        # Check if it's a JSON string
                        if result.startswith('"data:image'):
                            # Remove quotes from JSON string
                            data_uri = json.loads(result)
                        else:
                            data_uri = result
                        
                        if data_uri.startswith("data:image"):
                            # Extract base64 data
                            _, base64_data = data_uri.split(",", 1)
                            image_data = base64.b64decode(base64_data)
                            
                            # Save to file
                            width = options.get("target_width")
                            height = options.get("target_height")
                            focus = options.get("focus_area")
                            output_path = f"docs/pokemon-card-cropped-{width}x{height}-{focus}.png"
                            with open(output_path, "wb") as f:
                                f.write(image_data)
                            print(f"Saved cropped image to {output_path}")
                            continue
                    
                    print(f"Error: Could not extract data URI from result: {result[:100]}...")
                except Exception as e:
                    print(f"Error saving image: {str(e)}")
            else:
                print(f"Error: {response.status_code} - {response.text}")

async def main():
    """Main function to run all tests"""
    # Convert image to data URI
    image_data_uri = image_to_data_uri(IMAGE_PATH)
    
    # Test all image tools
    await test_analyze_image(image_data_uri)
    await test_apply_image_filter(image_data_uri)
    await test_resize_image(image_data_uri)
    await test_smart_crop_image(image_data_uri)

if __name__ == "__main__":
    asyncio.run(main())
