# core/base_classes.py

import numpy as np
import time
from typing import Dict, List, Any

class Rule:
    def __init__(self, rule_text):
        self.rule_text = rule_text
        self.usage_count = 0
        self.strength = 1.0
        self.embedding = np.random.randn(64)
        self.confidence = 1.0
        self.creation_time = time.time()
        self.last_used = time.time()
        self.connections = []
        
    def update_embedding(self, new_embedding):
        self.embedding = 0.9 * self.embedding + 0.1 * new_embedding
        
    def increment_usage(self):
        self.usage_count += 1
        self.last_used = time.time()
        self.strength *= 1.1
        
    def decay(self, current_time, decay_rate=0.1):
        time_factor = np.exp(-decay_rate * (current_time - self.last_used))
        self.strength *= time_factor
        
    def __str__(self):
        return self.rule_text

class SystemState:
    def __init__(self):
        self.metrics = {
            'complexity': 0,
            'novelty': 0,
            'confidence': 1.0,
            'resource_usage': {}
        }
        self.active_rules = []
        self.pending_actions = []
        self.knowledge_graph = nx.DiGraph()
        
    def update(self, new_data: Dict):
        self.metrics.update(new_data.get('metrics', {}))
        self.active_rules.extend(new_data.get('new_rules', []))
        self.pending_actions.extend(new_data.get('actions', []))
        
    def get_summary(self) -> Dict:
        return {
            'metrics': self.metrics,
            'active_rules_count': len(self.active_rules),
            'pending_actions_count': len(self.pending_actions)
        }
