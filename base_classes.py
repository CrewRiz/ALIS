import numpy as np
from datetime import datetime, timezone
import networkx as nx
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

@dataclass
class Rule:
    rule_text: str
    creation_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    usage_count: int = 0
    strength: float = 1.0
    confidence: float = 1.0
    embedding: np.ndarray = field(default_factory=lambda: np.random.randn(64))
    connections: List[str] = field(default_factory=list)
    
    def update_embedding(self, new_embedding: np.ndarray) -> None:
        """Update the rule's embedding using exponential moving average"""
        self.embedding = 0.9 * self.embedding + 0.1 * new_embedding
        
    def increment_usage(self) -> None:
        """Increment usage count and update metrics"""
        self.usage_count += 1
        self.last_used = datetime.now(timezone.utc)
        self.strength *= 1.1
        
    def decay(self, current_time: datetime, decay_rate: float = 0.1) -> None:
        """Apply time-based decay to rule strength"""
        if not isinstance(current_time, datetime):
            current_time = datetime.fromisoformat(str(current_time))
        
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
            
        time_diff = (current_time - self.last_used).total_seconds()
        time_factor = np.exp(-decay_rate * time_diff)
        self.strength *= time_factor
        
    def __str__(self) -> str:
        return f"Rule({self.rule_text}, strength={self.strength:.2f}, confidence={self.confidence:.2f})"

class SystemState:
    def __init__(self):
        self.metrics: Dict[str, Any] = {
            'complexity': 0.0,
            'novelty': 0.0,
            'confidence': 1.0,
            'resource_usage': {},
            'last_update': datetime.now(timezone.utc)
        }
        self.active_rules: List[Rule] = []
        self.pending_actions: List[Dict] = []
        self.knowledge_graph = nx.DiGraph()
        
    def update(self, new_data: Dict[str, Any]) -> None:
        """Update system state with new data"""
        self.metrics.update(new_data.get('metrics', {}))
        self.metrics['last_update'] = datetime.now(timezone.utc)
        
        # Add new rules
        new_rules = [Rule(**rule_data) if isinstance(rule_data, dict) else rule_data 
                    for rule_data in new_data.get('new_rules', [])]
        self.active_rules.extend(new_rules)
        
        # Update pending actions
        self.pending_actions.extend(new_data.get('actions', []))
        
        # Update knowledge graph
        if 'graph_updates' in new_data:
            for node, data in new_data['graph_updates'].get('nodes', {}).items():
                self.knowledge_graph.add_node(node, **data)
            for source, target, data in new_data['graph_updates'].get('edges', []):
                self.knowledge_graph.add_edge(source, target, **data)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current system state"""
        return {
            'metrics': self.metrics,
            'active_rules_count': len(self.active_rules),
            'pending_actions_count': len(self.pending_actions),
            'knowledge_graph_size': {
                'nodes': self.knowledge_graph.number_of_nodes(),
                'edges': self.knowledge_graph.number_of_edges()
            },
            'timestamp': datetime.now(timezone.utc)
        }
