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

async def analyze_pokemon_card(image_path, is_front=True):
    """Analyze the Pokemon card image with consistent results"""
    print(f"Analyzing image: {image_path} ({'front' if is_front else 'back'} of card)")
    
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
                        print("\nAnalysis Results (raw):")
                        print(result)
                        return False, False
                else:
                    analysis_results = result
                
                # Print the raw analysis results
                print("\nAnalysis Results:")
                print(json.dumps(analysis_results, indent=2))
                
                # Format the results for easier reading
                print("\n=== Pokemon Card Analysis ===")
                
                # Card detection results
                is_pokemon_card = analysis_results.get('metadata', {}).get('is_pokemon_card', False)
                is_card_back = analysis_results.get('metadata', {}).get('is_card_back', False)
                pokemon_visible = analysis_results.get('metadata', {}).get('pokemon_character_visible', False)
                
                if is_pokemon_card:
                    if is_card_back:
                        print("✅ DETECTED: This is the BACK of a Pokemon card")
                    else:
                        print("✅ DETECTED: This is the FRONT of a Pokemon card")
                        if pokemon_visible:
                            print("✅ Pokemon character is visible on the card")
                        else:
                            print("⚠️ No Pokemon character detected on the card")
                else:
                    print("❌ This does not appear to be a Pokemon card")
                
                print(f"\nDescription: {analysis_results.get('description', 'Unknown')}")
                print(f"Tags: {', '.join(analysis_results.get('tags', []))}")
                print(f"Colors: {', '.join(analysis_results.get('colors', []))}")
                
                print("\nObjects detected:")
                for obj in analysis_results.get('objects', []):
                    print(f"- {obj.get('name', 'Unknown')} (confidence: {obj.get('confidence', 0):.2f})")
                
                if analysis_results.get('text', []):
                    print("\nText detected:")
                    for text in analysis_results.get('text', []):
                        print(f"- {text.get('content', 'Unknown')} (confidence: {text.get('confidence', 0):.2f})")
                
                metadata = analysis_results.get('metadata', {})
                print("\nImage metadata:")
                print(f"- Dimensions: {metadata.get('width', 'Unknown')}x{metadata.get('height', 'Unknown')}")
                print(f"- Format: {metadata.get('format', 'Unknown')}")
                print(f"- Aspect ratio: {metadata.get('aspect_ratio', 'Unknown')}")
                print(f"- Average brightness: {metadata.get('average_brightness', 'Unknown')}")
                
                print("\nThis analysis was generated with Gemma 3 configuration:")
                config = metadata.get('model_config', {})
                print(f"- Temperature: {config.get('temperature', 1.0)}")
                print(f"- Top-k: {config.get('top_k', 64)}")
                print(f"- Top-p: {config.get('top_p', 0.95)}")
                print(f"- Min-p: {config.get('min_p', 0.0)}")
                print(f"- Seed: {config.get('seed', 42)}")
                print(f"- Analysis time: {analysis_results.get('analysis_time', 'Unknown')}")
                
                return is_pokemon_card, is_card_back
                
            except Exception as e:
                print(f"Error parsing result: {str(e)}")
                print("Raw result:", result)
                return False, False
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return False, False

async def main():
    """Main function to test both front and back of Pokemon card"""
    # Use the actual Pokemon card image
    pokemon_card = Path("docs/pokemon-card.webp")
    
    if not pokemon_card.exists():
        print(f"Error: Pokemon card image not found at {pokemon_card}")
        return
    
    print(f"Using real Pokemon card image: {pokemon_card}")
    
    # Analyze the Pokemon card
    print("\n=== ANALYZING POKEMON CARD ===\n")
    await analyze_pokemon_card(pokemon_card, is_front=True)
    
    # No additional analysis needed

if __name__ == "__main__":
    asyncio.run(main())
