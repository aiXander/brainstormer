"""
Configuration management for the Brainstormer application
"""

import os
import yaml

class Config:
    def __init__(self, config_file="config.yaml"):
        """
        Initialize configuration from a YAML file
        
        Args:
            config_file: Path to the configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()
        
        # Set default configuration values
        self.whisper_model = self.config.get('transcription', {}).get('model', 'gpt-4o-mini-transcribe')
        self.summarize_interval = self.config.get('main', {}).get('summarize_interval', 1800)
        
        # LLM configuration via OpenRouter
        self.llm_config = self.config.get('llm', {})
        if 'api_key' not in self.llm_config:
            self.llm_config['api_key'] = os.environ.get('OPENROUTER_API_KEY')
    
    def _load_config(self):
        """Load configuration from YAML file"""
        # Check if config file exists
        if not os.path.exists(self.config_file):
            print(f"Config file {self.config_file} not found, creating default config")
            self._create_default_config()
        
        # Load config from file
        try:
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
            return self._create_default_config()
    
    def _create_default_config(self):
        """Create a default configuration"""
        default_config = {
            'transcription': {
                'model': 'gpt-4o-mini-transcribe',
            },
            'llm': {
                'model': 'anthropic/claude-3-haiku',
                'api_url': 'https://openrouter.ai/api/v1/chat/completions',
                'max_tokens': 1000,
                'temperature': 0.7,
                'ideas_log_path': 'ideas_log.jsonl',
            },
            'main': {
                'idea_interval': 30,
                'summarize_interval': 1800
            },
            'logging': {
                'transcript_log_dir': "transcripts"
            }
        }
        
        # Write default config to file
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
        except Exception as e:
            print(f"Error creating default config file: {e}")
        
        return default_config 