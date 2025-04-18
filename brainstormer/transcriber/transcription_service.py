"""
TranscriptionService - Audio transcription using OpenAI's API
"""

import os
import numpy as np
import tempfile
import httpx
import json
import yaml
from openai import AsyncOpenAI
from pathlib import Path
from datetime import datetime
import soundfile as sf
import asyncio

class TranscriptionService:
    def __init__(self, model="gpt-4o-mini-transcribe"):
        """
        Initialize the TranscriptionService with OpenAI's API
        
        Args:
            model: OpenAI transcription model to use
        """
        self.model = model
        self.temp_dir = tempfile.gettempdir()
        
        # Initialize AsyncOpenAI client
        self.client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set your OpenAI API key."
            )
        
        # Check for OpenRouter API key
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Please set your OpenRouter API key."
            )
        
        # Load configuration
        self._load_config()
        
        # OpenRouter configuration
        self.openrouter_api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Initialize httpx client (reused for summarization)
        self.http_client = httpx.AsyncClient()
    
    def _load_config(self):
        """Load configuration from config.yaml file"""
        config_path = Path(__file__).parents[2] / "config.yaml"
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Get LLM model for summarization from config
        self.summary_model = config.get('llm', {}).get('model', "google/gemini-2.5-flash-preview")
        # Get summarization prompts and parameters
        self.summarization_system_prompt = config.get('llm', {}).get('summarization_system_prompt', "Default system prompt if not found.")
        self.summarization_user_prompt_template = config.get('llm', {}).get('summarization_user_prompt_template', "Default user prompt template: {transcript}")
        self.summarization_max_tokens = config.get('llm', {}).get('summarization_max_tokens', 500)
        self.summarization_temperature = config.get('llm', {}).get('summarization_temperature', 0.7)
    
    # --- Helper functions for async file operations ---
    async def _write_wav_file(self, path, data, samplerate, subtype):
        """Asynchronously write WAV file."""
        # SoundFile's write can be blocking, run in thread pool
        await asyncio.to_thread(sf.write, path, data, samplerate, subtype=subtype)

    async def _remove_file(self, path):
        """Asynchronously remove file."""
        # os.remove is blocking, run in thread pool
        if await asyncio.to_thread(os.path.exists, path):
            await asyncio.to_thread(os.remove, path)

    async def _read_audio_file(self, path):
        """Asynchronously open and read an audio file's content."""
        # Using asyncio's file operations might be complex with OpenAI client expecting a file-like object.
        # A simpler approach for now is to read into memory within the thread,
        # though this could be inefficient for very large files.
        # Let's try passing the path directly if the async client supports it,
        # otherwise, we'll read it in the thread.
        # The openai async client seems to handle file paths directly or async file objects.
        # We'll open it asynchronously. NOTE: This needs python 3.9+ typically with libraries like aiofiles.
        # For broader compatibility, let's stick to reading the bytes in a thread for now.
        return await asyncio.to_thread(Path(path).read_bytes)

    # --- End Helper Functions ---

    async def transcribe(self, audio_data):
        """
        Transcribe audio data using OpenAI's transcription API asynchronously
        
        Args:
            audio_data: Numpy array containing audio data (16kHz, mono)
            
        Returns:
            Transcribed text
        """
        if audio_data is None or audio_data.size == 0:
            # print("No audio data received for transcription.")
            return ""
            
        # Save audio data to a temporary WAV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_wav = Path(self.temp_dir) / f"openai_temp_{timestamp}.wav"
        
        # Convert float32 numpy array to int16
        # This conversion can stay synchronous as it's CPU-bound and relatively fast
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Save as 16-bit WAV file asynchronously
        await self._write_wav_file(temp_wav, audio_int16, 16000, subtype='PCM_16')
        
        try:
            # Open the audio file asynchronously and send to OpenAI API
            # The async client expects a file-like object or bytes.
            # Let's pass the file bytes read asynchronously.
            audio_bytes = await self._read_audio_file(temp_wav)
            
            if not audio_bytes:
                 print(f"Warning: Temporary audio file {temp_wav} appears empty after writing.")
                 return ""

            # Create parameters dict
            params = {
                "model": self.model,
                "file": (temp_wav.name, audio_bytes, "audio/wav"), # Pass as tuple (filename, content, mimetype)
            }

            # Call the API asynchronously
            transcription = await self.client.audio.transcriptions.create(**params) # Added await

            # Handle the response based on its type
            if hasattr(transcription, 'text'):
                return transcription.text
            elif isinstance(transcription, str):
                return transcription
            elif isinstance(transcription, dict) and 'text' in transcription:
                return transcription['text']
            else:
                print(f"Unexpected response format: {type(transcription)}")
                return str(transcription)
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
            
        finally:
            # Clean up temporary file asynchronously
            await self._remove_file(temp_wav) # Changed to async helper
    
    async def summarize(self, transcript):
        """
        Summarize a long transcript asynchronously to avoid sending too much data to the LLM
        
        Args:
            transcript: The full transcript text to summarize
            
        Returns:
            Summarized transcript
        """
        try:
            # Use OpenRouter API for summarization via httpx
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.summary_model,
                "messages": [
                    {"role": "system", "content": self.summarization_system_prompt},
                    {"role": "user", "content": self.summarization_user_prompt_template.format(transcript=transcript)}
                ],
                "max_tokens": self.summarization_max_tokens,
                "temperature": self.summarization_temperature
            }
            
            # Use the shared httpx client
            response = await self.http_client.post(self.openrouter_api_url, headers=headers, json=data) # Changed to await http_client.post
            response.raise_for_status()
            result = response.json()
            
            return result['choices'][0]['message']['content']
            
        except httpx.RequestError as e: # Catch specific httpx errors
            print(f"Summarization HTTP request error: {e}")
            # Fallback logic
            words = transcript.split()
            if len(words) > 1000:
                return " ".join(words[-1000:]) + " [earlier content truncated due to summarization error]"
            return transcript
        except Exception as e:
            print(f"Summarization error: {e}")
            # Fallback logic (as before)
            words = transcript.split()
            if len(words) > 1000:
                return " ".join(words[-1000:]) + " [earlier content truncated due to summarization error]"
            return transcript

    # Add a method to close the httpx client when done
    async def close(self):
        await self.http_client.aclose() 