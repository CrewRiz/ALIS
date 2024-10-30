# utils/safety.py

import logging
from typing import List, Dict

class SafetyChecker:
    def __init__(self):
        self.unsafe_commands = ['rm -rf', 'format', 'del', 'shutdown']
        self.unsafe_domains = ['malware', 'phishing']
        self.unsafe_paths = ['/system', 'C:\\Windows']
        self.unsafe_content = ['password', 'credit card']
        
    def is_safe_command(self, command: str) -> bool:
        return not any(cmd in command.lower() for cmd in self.unsafe_commands)
        
    def is_safe_url(self, url: str) -> bool:
        return not any(domain in url.lower() for domain in self.unsafe_domains)
        
    def is_safe_path(self, path: str) -> bool:
        return not any(p in path for p in self.unsafe_paths)
        
    def is_safe_content(self, content: str) -> bool:
        return not any(c in content.lower() for c in self.unsafe_content)

class ActionLogger:
    def __init__(self, filename='system_actions.log'):
        logging.basicConfig(
            filename=filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.action_history = []
        
    def log_action(self, message: str, level: str = "INFO"):
        self.action_history.append({
            'timestamp': time.time(),
            'action': message,
            'level': level
        })
        if level == "INFO":
            logging.info(message)
        elif level == "ERROR":
            logging.error(message)

