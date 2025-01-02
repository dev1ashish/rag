# processors/image_processor.py
import os
from typing import Dict, List, Any
import openai
from PIL import Image
from .document_processor import DocumentProcessor
import base64
import io

class ImageProcessor(DocumentProcessor):
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = api_key

    def process(self, file_path: str) -> Dict[str, List[Any]]:
        """Process image file and return description with metadata."""
        try:
            # Open and validate image
            image = Image.open(file_path)
            
            # Convert to RGB if necessary
            if image.mode in ('RGBA', 'P'):
                image = image.convert('RGB')
            
            # Resize image if too large (OpenAI has file size limits)
            max_size = (1024, 1024)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Prepare image for API
            buffered = io.BytesIO()
            image.save(buffered, format=image.format if image.format else "JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Log attempt to call API (without sensitive data)
            print(f"Attempting to process image: {os.path.basename(file_path)}")
            
            response = openai.ChatCompletion.create(
                model="gpt-4o",  # Updated model name
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in detail, including any text, objects, and important information visible."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_str}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            description = response.choices[0].message['content']
            
            return {
                'texts': [description],
                'metadata': [{
                    'source': str(file_path),
                    'type': 'image',
                    'format': image.format,
                    'size': image.size
                }]
            }
        except Exception as e:
            print(f"Error processing image file: {str(e)}")
            return {
                'texts': ['Error processing image file'],
                'metadata': [{'source': str(file_path), 'type': 'error', 'error': str(e)}]
            }