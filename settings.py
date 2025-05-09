"""
Global settings and configuration for ALIS system.
"""

import os
from datetime import datetime, timezone
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_system_time() -> datetime:
    """Get the system time in UTC"""
    return datetime.fromisoformat("2024-12-23T03:08:52-06:00").astimezone(timezone.utc)

class Settings:
    """Global settings singleton"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize settings"""
        self.BASE_DIR = Path(__file__).resolve().parent
        self.SYSTEM_START_TIME = get_system_time()
        
        self.config: Dict[str, Any] = {
            'api_keys': {
                'anthropic': os.getenv('ANTHROPIC_API_KEY'),
                'openai': os.getenv('OPENAI_API_KEY')
            },
            
            'system': {
                'max_complexity': 1000,
                'novelty_threshold': 0.7,
                'confidence_threshold': 0.8,
                'max_simulation_depth': 5,
                'max_memory_items': 1000,
                'embedding_dimension': 768, # Changed from 1536 to 768 for text-embedding-004
                'start_time': self.SYSTEM_START_TIME
            },
            
            'safety': {
                'unsafe_commands': ['rm -rf', 'format', 'del', 'shutdown'],
                'unsafe_domains': ['malware', 'phishing'],
                'unsafe_paths': ['/system', 'C:\\Windows'],
                'unsafe_content': ['password', 'credit card'],
                'max_action_interval': 300  # 5 minutes
            },
            
            'web_interaction': {
                'timeout': 30,
                'retry_attempts': 3,
                'delay_between_actions': 1.0,
                'screenshot_on_error': True,
                'max_concurrent_sessions': 3
            },
            
            'consciousness': {
                'quantum_simulation_depth': 5,
                'entanglement_threshold': 0.5,
                'temporal_decay_rate': 0.1,
                'complexity_emergence_threshold': 0.7,
                'coherence_threshold': 0.3
            },
            
            'google_cloud': {
                'project_id': os.getenv('GOOGLE_CLOUD_PROJECT_ID'),
                'location': 'us-central1',  # Default region for Vertex AI
                'index_endpoint_name': os.getenv('VERTEX_INDEX_ENDPOINT'),
                'index_name': os.getenv('VERTEX_INDEX_NAME'),
                'deployed_index_id': os.getenv('VERTEX_DEPLOYED_INDEX_ID')
            },
            
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                # 'file': str(self.BASE_DIR / 'logs' / 'alis.log'), # Removed file logging
                # 'max_file_size': 10 * 1024 * 1024  # 10MB # Removed file size limit
            }
        }
    
    def __getitem__(self, key: str) -> Any:
        return self.config[key]
    
    def get_current_time(self) -> datetime:
        """Get current system time"""
        return get_system_time()

# Global settings instance
SETTINGS = Settings()
