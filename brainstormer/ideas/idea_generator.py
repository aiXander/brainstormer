"""
IdeaGenerator - Generates ideas from conversation transcripts using LLMs via OpenRouter
"""

import httpx
import json
import os
from datetime import datetime
import asyncio
import logging # Added logging

# Setup basic logging for this module if needed, or rely on main's config
logger = logging.getLogger(__name__)

class IdeaGenerator:
    def __init__(self, config):
        """
        Initialize the idea generator
        
        Args:
            config: Configuration dictionary containing OpenRouter API key, model settings, and prompts.
        """
        self.api_key = config.get('api_key', os.environ.get('OPENROUTER_API_KEY'))
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or provide in config.")
        
        self.model = config.get('model', 'google/gemini-2.5-flash-preview')
        self.api_url = config.get('api_url', 'https://openrouter.ai/api/v1/chat/completions')
        self.max_tokens = config.get('max_tokens', 1000)
        self.temperature = config.get('temperature', 0.7)
        self.site_url = config.get('site_url', '')
        self.site_name = config.get('site_name', '')
        
        # --- Read prompts from config ---
        self.system_prompt = config.get('system_prompt')
        self.user_prompt_template = config.get('user_prompt_template')
        if not self.system_prompt or not self.user_prompt_template:
             raise ValueError("System prompt and user prompt template must be defined in the LLM config.")
        
        # Path to save generated ideas (using .jsonl as specified in config example)
        self.ideas_log_path = config.get('ideas_log_path', 'ideas_log.jsonl')
        
        # Ensure log directory exists
        log_dir = os.path.dirname(self.ideas_log_path)
        if log_dir:
             os.makedirs(log_dir, exist_ok=True)

        # Initialize httpx client
        self.http_client = httpx.AsyncClient()

    async def generate(self, transcript, image_url=None):
        """
        Generate ideas from a conversation transcript, optionally with an image
        
        Args:
            transcript: The transcript text to analyze
            image_url: Optional URL to an image to include in the prompt
            
        Returns:
            List of idea strings, or an empty list if no useful ideas were generated or an error occurred.
        """
        if not transcript or not transcript.strip():
            logger.info("Skipping idea generation: Empty transcript provided.")
            return []
            
        # --- Prepare the prompt using the template ---
        user_content_text = self.user_prompt_template.format(transcript=transcript)

        # Prepare user content as array for multimodal support
        user_content = [{"type": "text", "text": user_content_text}]
        
        # Add image if provided
        if image_url:
            # Ensure the model supports multimodal input before adding image
            # (Simple check, might need refinement based on specific model capabilities)
            if "vision" in self.model or "gemini" in self.model: 
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            else:
                 logger.warning(f"Image URL provided, but model '{self.model}' might not support image input. Skipping image.")
                 # Send only text if model doesn't support images but image was provided
                 user_content = user_content_text # Fallback to text-only


        # Make API request to OpenRouter using httpx
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Add optional referrer headers if configured
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                 # Use the appropriate format for content (list or string)
                {"role": "user", "content": user_content if image_url and ("vision" in self.model or "gemini" in self.model) else user_content_text}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        ideas_list = [] # Initialize empty list
        try:
            # Use the shared httpx client
            logger.debug(f"Sending request to LLM: {self.model}")
            response = await self.http_client.post(self.api_url, headers=headers, json=data, timeout=60.0) # Added timeout
            response.raise_for_status()
            result = response.json()
            
            # Extract ideas from the response
            if not result.get('choices') or not result['choices'][0].get('message') or not result['choices'][0]['message'].get('content'):
                 logger.warning("LLM response missing expected content structure.")
                 return [] # Return empty list if structure is wrong

            ideas_text = result['choices'][0]['message']['content'].strip()
            
            # --- Handle "NO_IDEAS" response ---
            if ideas_text == "NO_IDEAS":
                logger.info("LLM indicated no useful ideas to generate for this segment.")
                return [] # Return empty list as requested by the prompt design

            ideas_list = [idea.strip() for idea in ideas_text.split('\n') if idea.strip()]
            
            # Log the generated ideas asynchronously only if ideas were generated
            if ideas_list:
                await self._log_ideas(transcript, ideas_list)
            
            return ideas_list
            
        except httpx.TimeoutException:
             logger.error(f"Error generating ideas: Request timed out after 60 seconds.")
             return [] # Return empty list on timeout
        except httpx.RequestError as e:
            logger.error(f"Error generating ideas (HTTP Request): {e}")
            # Optionally return error message for debugging, but usually empty list is better for flow
            # return [f"Failed to generate ideas due to a network error: {e}"] 
            return [] 
        except httpx.HTTPStatusError as e:
             logger.error(f"Error generating ideas (HTTP Status): {e.response.status_code} - {e.response.text}")
             return []
        except Exception as e:
            logger.error(f"Unexpected error generating ideas: {e}", exc_info=True)
            # Optionally return error message
            # return [f"Failed to generate ideas due to an unexpected error: {e}"]
            return []
    
    async def _log_ideas(self, transcript, ideas): # Changed transcript -> transcript_snippet arg name for clarity
        """Log generated ideas asynchronously to a file for later analysis"""
        timestamp = datetime.now().isoformat()
        # Use a smaller snippet for logging to keep log file size reasonable
        transcript_snippet = transcript[:200] + "..." if len(transcript) > 200 else transcript 
        log_entry = {
            "timestamp": timestamp,
            "transcript_snippet": transcript_snippet,
            "ideas": ideas
        }
        
        log_line = json.dumps(log_entry) + '\n'
        
        # Append to log file asynchronously using thread pool
        try:
            await asyncio.to_thread(self._append_to_log, log_line) 
        except Exception as e:
            logger.error(f"Error writing to ideas log file {self.ideas_log_path}: {e}")

    # Helper sync function for file writing to be called by to_thread
    def _append_to_log(self, line):
        try: # Added try-except block here for file writing errors
            with open(self.ideas_log_path, 'a', encoding='utf-8') as f:
                f.write(line)
        except Exception as e:
             logger.error(f"Error writing line to ideas log file {self.ideas_log_path}: {e}")

    # Add a method to close the httpx client when done
    async def close(self):
        await self.http_client.aclose() 