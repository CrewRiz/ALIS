# enhanced_learning_system.py

import logging
from datetime import datetime, timezone
import asyncio
from typing import Dict, List, Optional, Any
import numpy as np
import anthropic
import openai
from pathlib import Path

from settings import SETTINGS
from base_classes import SystemState
from web_interaction import WebInteractionAgent
from pattern_detection import PatternDetectionAgent
from rule_generation import RuleGenerationAgent
from analysis import AnalysisAgent
from genetic_memory import RAGMemory
from quantum_consciousness import QuantumConsciousness, FractalComplexitySystem
from safety import CapabilityManager, IncompletenessDetector

class EnhancedLearningSystem:
    def __init__(self):
        self._initialize_logging()
        self._initialize_api_clients()
        self._initialize_components()
        self._initialize_metrics()
        
    def _initialize_logging(self):
        logging.basicConfig(
            level=SETTINGS['logging']['level'],
            format=SETTINGS['logging']['format'],
            filename=SETTINGS['logging']['file']
        )
        self.logger = logging.getLogger(__name__)

    def _initialize_api_clients(self):
        # Initialize API clients with error handling
        try:
            self.claude_client = anthropic.Client(
                api_key=SETTINGS['api_keys']['anthropic']
            ) if SETTINGS['api_keys']['anthropic'] else None
            
            self.openai_client = openai.Client(
                api_key=SETTINGS['api_keys']['openai']
            ) if SETTINGS['api_keys']['openai'] else None
            
        except Exception as e:
            self.logger.error(f"Failed to initialize API clients: {str(e)}")
            raise

    def _initialize_components(self):
        try:
            # Core components
            self.system_state = SystemState()
            self.web_agent = WebInteractionAgent(self.system_state)
            self.pattern_agent = PatternDetectionAgent(self.claude_client)
            self.rule_agent = RuleGenerationAgent(self.claude_client)
            self.analysis_agent = AnalysisAgent(self.openai_client)
            
            # Memory system
            self.memory = RAGMemory(
                dimension=SETTINGS['system']['embedding_dimension']
            )
            
            # Advanced components
            self.consciousness = QuantumConsciousness()
            self.complexity_system = FractalComplexitySystem()
            
            # Management systems
            self.capability_manager = CapabilityManager()
            self.incompleteness_detector = IncompletenessDetector()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}")
            raise

    def _initialize_metrics(self):
        self.complexity = 0.0
        self.novelty = 0.0
        self.incompleteness_flag = False
        self.modification_layers = 0
        self.last_update = datetime.now(timezone.utc)
        self.last_complexity = 0.0
        self.last_novelty = 0.0

    async def _analyze_task_requirements(self, task: str) -> Dict[str, Any]:
        """Analyze task requirements and capabilities needed"""
        try:
            # Basic requirement analysis
            requirements = {
                'web_access': 'web' in task.lower(),
                'pattern_analysis': 'pattern' in task.lower(),
                'learning': 'learn' in task.lower(),
                'timestamp': SETTINGS.get_current_time()
            }
            
            # Add complexity estimation
            requirements['estimated_complexity'] = len(task.split()) * 0.1
            
            return requirements
            
        except Exception as e:
            self.logger.error(f"Failed to analyze task requirements: {str(e)}")
            return {}

    async def process_web_task(self, task: str) -> Dict:
        try:
            # Check capabilities
            requirements = await self._analyze_task_requirements(task)
            if not self.incompleteness_detector.assess_completeness(
                requirements, 
                self.capability_manager.capabilities
            ):
                await self._handle_incompleteness(requirements)
            
            # Get relevant context
            context = self._get_entangled_context(task)
            
            # Simulate possibilities
            possible_actions = await self._generate_actions(task)
            possibilities = await self.consciousness.simulate_possibilities(
                self.system_state.get_summary(),
                possible_actions
            )
            
            # Choose best possibility
            chosen = max(possibilities, key=lambda x: x['probability'])
            final_state = self.consciousness.collapse_wave_function(chosen)
            
            # Execute chosen action
            web_results = await self.web_agent.execute_web_task(chosen['action'])
            
            # Detect patterns
            pattern_results = await self.pattern_agent.detect_patterns({
                'task': task,
                'web_results': web_results,
                'context': context
            })
            patterns = pattern_results.get('patterns', [])
            
            # Generate rules
            new_rules = await self.rule_agent.generate_rules({
                'patterns': patterns,
                'results': web_results,
                'quantum_state': self.consciousness.quantum_state
            })
            
            # Perform analysis
            analysis = await self.analysis_agent.analyze({
                'task': task,
                'patterns': patterns,
                'rules': new_rules,
                'web_results': web_results
            })
            
            # Update complexity system
            self.complexity_system.add_layer('task_execution')
            
            # Store in memory
            self.memory.add_memory({
                'task': task,
                'patterns': patterns,
                'results': web_results,
                'rules': new_rules,
                'analysis': analysis,
                'quantum_state': self.consciousness.quantum_state.copy(),
                'timestamp': SETTINGS.get_current_time()
            })
            
            return {
                'web_results': web_results,
                'patterns': patterns,
                'rules': new_rules,
                'analysis': analysis,
                'system_state': self.system_state.get_summary(),
                'quantum_state': self.consciousness.quantum_state,
                'complexity_layers': [
                    {
                        'type': layer.layer_type,
                        'complexity': layer.complexity_score,
                        'properties': layer.emergent_properties
                    }
                    for layer in self.complexity_system.layers
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Task processing failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    async def _get_entangled_context(self, query: str) -> List[Dict]:
        """Get context considering temporal entanglement"""
        base_context = self.memory.get_relevant_context(query)
        weighted_context = []
        
        for item in base_context:
            if 'timestamp' in item:
                temporal_weight = self._calculate_temporal_weight(item['timestamp'])
                weighted_context.append({
                    **item,
                    'temporal_weight': temporal_weight
                })
        
        weighted_context.sort(key=lambda x: x.get('temporal_weight', 0), reverse=True)
        return weighted_context

    def _calculate_temporal_weight(self, timestamp: datetime) -> float:
        """Calculate weight based on temporal entanglement with proper time handling"""
        if not isinstance(timestamp, datetime):
            timestamp = datetime.fromisoformat(str(timestamp))
            
        # Convert to UTC if not already
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
            
        # Use system's last update time instead of current time
        time_diff = (self.last_update - timestamp).total_seconds()
        
        # Base temporal decay
        base_weight = np.exp(-time_diff / 3600)  # 1-hour characteristic time
        
        # Modify by quantum coherence
        coherence = self.consciousness.quantum_state.get('temporal_coherence', 0.5)
        
        return float(base_weight * coherence)
