import pandas as pd
from PyPDF2 import PdfReader
import openai
from typing import Dict, List, Any, Optional
import os
from PIL import Image
import base64
import requests
from bs4 import BeautifulSoup

class DocumentProcessor:
    """Base class for document processors"""
    def process(self, file_path: str) -> Dict[str, List[Any]]:
        raise NotImplementedError

class ExcelProcessor(DocumentProcessor):
    def process(self, file_path: str) -> Dict[str, List[Any]]:
        """Process Excel file and return extracted text with metadata."""
        try:
            # Read the Excel file
            df = pd.read_excel(file_path)
            
            # Convert to a single string representation
            text_content = [df.to_string(index=False)]  # Single chunk approach
            
            # Simple metadata that won't cause index issues
            metadata = [{
                'source': str(file_path),
                'type': 'excel',
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'columns': ', '.join(df.columns.tolist())
            }]

            return {
                'texts': text_content,
                'metadata': metadata
            }
        except Exception as e:
            print(f"Error processing Excel file: {str(e)}")
            return {
                'texts': ['Error processing Excel file'],
                'metadata': [{'source': str(file_path), 'type': 'error'}]
            }

class PDFProcessor(DocumentProcessor):
    def process(self, file_path: str) -> Dict[str, List[Any]]:
        """Process PDF file and return extracted text with metadata."""
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                texts = []
                metadata = []
                
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        texts.append(text)
                        metadata.append({
                            'source': os.path.basename(file_path),
                            'type': 'pdf',
                            'page': page_num + 1
                        })
                
                return {
                    'texts': texts,
                    'metadata': metadata
                }
        except Exception as e:
            print(f"Error processing PDF file: {str(e)}")
            return {
                'texts': ['Error processing PDF file'],
                'metadata': [{'source': str(file_path), 'type': 'error'}]
            }

class AudioProcessor(DocumentProcessor):
    def __init__(self, api_key: str):
        """Initialize audio processor with OpenAI API key."""
        self.api_key = api_key
        openai.api_key = api_key

    def process(self, file_path: str) -> Dict[str, List[Any]]:
        """Process audio file and return transcription with metadata."""
        try:
            with open(file_path, "rb") as audio_file:
                transcript = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio_file
                )

            # Handle both dictionary and string responses
            text = transcript.get("text") if isinstance(transcript, dict) else str(transcript)

            return {
                'texts': [text],
                'metadata': [{
                    'source': str(file_path),
                    'type': 'audio',
                    'model': 'whisper-1'
                }]
            }
        except Exception as e:
            print(f"Error processing audio file: {str(e)}")
            return {
                'texts': ['Error processing audio file'],
                'metadata': [{'source': str(file_path), 'type': 'error'}]
            }

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
            import io
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

class WebProcessor(DocumentProcessor):
    def process(self, url: str) -> Dict[str, List[Any]]:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            try:
                soup = BeautifulSoup(response.text, 'html.parser')
            except Exception as e:
                return {
                    'texts': ['Error parsing webpage'],
                    'metadata': [{'source': str(url), 'type': 'error'}]
                }
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'header', 'footer', 'nav']):
                element.decompose()
            
            # Extract text
            text = soup.get_text(separator='\n')
            
            # Clean text
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)
            
            return {
                'texts': [text],
                'metadata': [{
                    'source': url,
                    'type': 'web',
                    'title': soup.title.string if soup.title else url
                }]
            }
        except Exception as e:
            print(f"Error processing webpage: {str(e)}")
            return {
                'texts': ['Error processing webpage'],
                'metadata': [{'source': str(url), 'type': 'error'}]
            }

def get_processor(file_type: str, api_key: Optional[str] = None) -> DocumentProcessor:
    """Factory function to get appropriate processor"""
    processors = {
        'excel': lambda: ExcelProcessor(),
        'pdf': lambda: PDFProcessor(),
        'audio': lambda: AudioProcessor(api_key),
        'image': lambda: ImageProcessor(api_key),
        'web': lambda: WebProcessor()
    }
    
    processor_factory = processors.get(file_type)
    if processor_factory is None:
        raise ValueError(f"Unsupported file type: {file_type}")
        
    if file_type in ['audio', 'image'] and api_key is None:
        raise ValueError(f"API key required for {file_type} processing")
    
    return processor_factory()