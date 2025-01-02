from typing import Dict, Any, Union
from pathlib import Path
import openai

class AudioProcessor:
    def __init__(self, api_key: str):
        """Initialize audio processor with OpenAI API key."""
        self.api_key = api_key
        openai.api_key = api_key

    def process(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process audio file using OpenAI's Whisper API and return transcription with metadata.
        """
        try:
            # Convert to Path object if string
            if isinstance(file_path, str):
                file_path = Path(file_path)

            with open(file_path, "rb") as audio_file:
                transcript = openai.Audio.transcribe(
                    "whisper-1", 
                    audio_file
                )

            return {
                'texts': [transcript["text"]],
                'metadata': [{
                    'source': str(file_path),
                    'type': 'audio_transcript'
                }]
            }
        except Exception as e:
            raise Exception(f"Error processing audio file: {str(e)}")