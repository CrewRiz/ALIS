"""
Pattern detection agent for analyzing data patterns.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import anthropic
import logging
from dataclasses import dataclass, field

from settings import SETTINGS

@dataclass
class Pattern:
    """Represents a detected pattern"""
    pattern_type: str
    description: str
    confidence: float
    timestamp: datetime = field(default_factory=lambda: SETTINGS.get_current_time())
    metadata: Dict[str, Any] = field(default_factory=dict)

class PatternDetectionAgent:
    def __init__(self, client: Optional[anthropic.Client]):
        self.client = client
        self.pattern_history: List[Dict[str, Any]] = []
        self.confidence_threshold = SETTINGS['system']['confidence_threshold']
        self.logger = logging.getLogger(__name__)
        
    async def detect_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect patterns in provided data"""
        if not self.client:
            self.logger.error("No API client available for pattern detection")
            return self._create_error_response("API client not initialized")
            
        try:
            prompt = self._create_pattern_prompt(data)
            
            response = await self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            
            patterns = self._parse_pattern_response(response.content)
            
            # Record pattern detection
            self.pattern_history.append({
                'data': data,
                'patterns': patterns,
                'timestamp': SETTINGS.get_current_time()
            })
            
            return {
                'status': 'success',
                'patterns': patterns,
                'timestamp': SETTINGS.get_current_time()
            }
            
        except Exception as e:
            self.logger.error(f"Pattern detection failed: {str(e)}")
            return self._create_error_response(str(e))
    
    def _create_pattern_prompt(self, data: Dict[str, Any]) -> str:
        """Create prompt for pattern detection"""
        return f"""Analyze the following data for patterns:
        1. Recurring elements
        2. Structural similarities
        3. Temporal patterns
        4. Causal relationships
        5. Quantum-like correlations
        
        Consider:
        - Pattern significance
        - Confidence levels
        - Temporal aspects
        - Potential implications
        
        Data: {data}
        """
    
    def _parse_pattern_response(self, content: str) -> List[Pattern]:
        """Parse pattern detection response into structured format"""
        try:
            # Basic parsing for now - could be enhanced with more sophisticated NLP
            patterns = []
            current_time = SETTINGS.get_current_time()
            
            # Add a general pattern for the content
            patterns.append(Pattern(
                pattern_type="general",
                description=content,
                confidence=0.8,
                timestamp=current_time
            ))
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to parse pattern response: {str(e)}")
            return []
    
    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'status': 'error',
            'error': error_msg,
            'timestamp': SETTINGS.get_current_time()
        }
    
    def get_pattern_history(self, 
                          start_time: Optional[datetime] = None, 
                          pattern_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get filtered pattern history"""
        if not start_time:
            start_time = SETTINGS['system']['start_time']
            
        filtered_history = [
            entry for entry in self.pattern_history
            if entry['timestamp'] >= start_time
        ]
        
        if pattern_type:
            filtered_history = [
                entry for entry in filtered_history
                if any(p.pattern_type == pattern_type for p in entry['patterns'])
            ]
            
        return filtered_history
