# settings.py

import os
from dotenv import load_dotenv

load_dotenv()

SETTINGS = {
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
        'embedding_dimension': 1536
    },
    
    'safety': {
        'unsafe_commands': ['rm -rf', 'format', 'del', 'shutdown'],
        'unsafe_domains': ['malware', 'phishing'],
        'unsafe_paths': ['/system', 'C:\\Windows'],
        'unsafe_content': ['password', 'credit card']
    },
    
    'web_interaction': {
        'timeout': 30,
        'retry_attempts': 3,
        'delay_between_actions': 1.0,
        'screenshot_on_error': True
    },
    
    'consciousness': {
        'quantum_simulation_depth': 5,
        'entanglement_threshold': 0.5,
        'temporal_decay_rate': 0.1,
        'complexity_emergence_threshold': 0.7
    },
    
    'logging': {
        'level': 'INFO',
        'file': 'alice.log',
        'format': '%(asctime)s - %(levelname)s - %(message)s'
    }
}



