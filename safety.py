"""
Safety and security utilities for ALIS system.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field

from settings import SETTINGS

@dataclass
class SecurityEvent:
    """Represents a security-related event"""
    event_type: str
    description: str
    severity: str
    timestamp: datetime = field(default_factory=lambda: SETTINGS.get_current_time())
    metadata: Dict[str, Any] = field(default_factory=dict)

class SafetyChecker:
    """Checks for potentially unsafe operations"""
    
    def __init__(self):
        self.settings = SETTINGS['safety']
        self.logger = logging.getLogger(__name__)
        self.event_history: List[SecurityEvent] = []
        
    def check_operation(self, operation_type: str, 
                       content: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Comprehensive safety check for an operation"""
        try:
            checks = {
                'command': self.is_safe_command,
                'url': self.is_safe_url,
                'path': self.is_safe_path,
                'content': self.is_safe_content
            }
            
            if operation_type not in checks:
                return self._create_error_response(
                    f"Unknown operation type: {operation_type}"
                )
            
            check_func = checks[operation_type]
            is_safe = check_func(content)
            
            if not is_safe:
                event = SecurityEvent(
                    event_type=f"unsafe_{operation_type}",
                    description=f"Unsafe {operation_type} detected: {content}",
                    severity="HIGH",
                    metadata=metadata or {}
                )
                self.event_history.append(event)
                self.logger.warning(f"Security event: {event}")
            
            return {
                'safe': is_safe,
                'operation_type': operation_type,
                'timestamp': SETTINGS.get_current_time(),
                'checks_performed': [operation_type],
                'metadata': metadata or {}
            }
            
        except Exception as e:
            self.logger.error(f"Safety check failed: {str(e)}")
            return self._create_error_response(str(e))
    
    def is_safe_command(self, command: str) -> bool:
        """Check if a command is safe to execute"""
        return not any(cmd.lower() in command.lower() 
                      for cmd in self.settings['unsafe_commands'])
    
    def is_safe_url(self, url: str) -> bool:
        """Check if a URL is safe to access"""
        return not any(domain.lower() in url.lower() 
                      for domain in self.settings['unsafe_domains'])
    
    def is_safe_path(self, path: str) -> bool:
        """Check if a file system path is safe to access"""
        path_obj = Path(path).resolve()
        return not any(str(path_obj).startswith(str(Path(p).resolve())) 
                      for p in self.settings['unsafe_paths'])
    
    def is_safe_content(self, content: str) -> bool:
        """Check if content contains sensitive information"""
        return not any(term.lower() in content.lower() 
                      for term in self.settings['unsafe_content'])
    
    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'safe': False,
            'error': error_msg,
            'timestamp': SETTINGS.get_current_time()
        }

class ActionLogger:
    """Logs system actions and maintains action history"""
    
    def __init__(self, filename='system_actions.log'):
        logging.basicConfig(
            filename=filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.action_history: List[Dict[str, Any]] = []
        
    def log_action(self, message: str, level: str = "INFO", 
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log an action with optional metadata"""
        log_entry = {
            'message': message,
            'level': level,
            'timestamp': SETTINGS.get_current_time(),
            'metadata': metadata or {}
        }
        
        self.action_history.append(log_entry)
        
        if level.upper() == "ERROR":
            logging.error(message)
        elif level.upper() == "WARNING":
            logging.warning(message)
        else:
            logging.info(message)
