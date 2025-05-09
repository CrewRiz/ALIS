# consciousness/quantum_consciousness.py

import networkx as nx
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
import google.generativeai as genai

from settings import SETTINGS

@dataclass
class SimulationResult:
    action: Dict[str, Any]
    simulation: Dict[str, Any]
    probability: float
    depth: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class QuantumConsciousness:
    def __init__(self, gemini_client=None):
        self.possibility_space: Dict[str, Any] = {}
        self.active_simulations: List[SimulationResult] = []
        self.collapsed_states: List[Dict[str, Any]] = []
        self.quantum_state: Dict[str, Any] = {
            'superposition': [],
            'entangled_thoughts': {},
            'wave_function': 1.0,
            'temporal_coherence': 0.5,
            'last_collapse': datetime.now(timezone.utc)
        }
        self.simulation_depth = SETTINGS['consciousness']['quantum_simulation_depth']
        self.entanglement_threshold = SETTINGS['consciousness']['entanglement_threshold']
        self.gemini_client = gemini_client
        
    async def simulate_possibilities(self, initial_state: Dict[str, Any], 
                                  actions: List[Dict[str, Any]]) -> List[SimulationResult]:
        """Simulate multiple possible outcomes from actions using Gemini for enhanced prediction"""
        possibilities: List[SimulationResult] = []
        
        for action in actions:
            for depth in range(self.simulation_depth):
                simulation = await self._simulate_branch(
                    initial_state.copy(), 
                    action,
                    depth
                )
                
                result = SimulationResult(
                    action=action,
                    simulation=simulation,
                    probability=self._calculate_probability(simulation),
                    depth=depth
                )
                possibilities.append(result)
                
        self.quantum_state['superposition'] = possibilities
        self.quantum_state['last_collapse'] = datetime.now(timezone.utc)
        return possibilities
        
    async def _simulate_branch(self, state: Dict[str, Any], 
                             action: Dict[str, Any], 
                             depth: int) -> Dict[str, Any]:
        """Simulate a single possibility branch"""
        current_state = state.copy()
        
        for step in range(depth):
            # Add quantum uncertainty
            current_state = self._add_quantum_uncertainty(current_state, step)
            
            # Predict next state
            current_state = await self._predict_next_state(
                current_state,
                action,
                step
            )
            
            # Update entanglement
            self._update_entanglement(current_state)
        
        return current_state
    
    def _calculate_probability(self, simulation: Dict[str, Any]) -> float:
        """Calculate probability of a simulation outcome"""
        # Base probability from wave function
        base_prob = self.quantum_state['wave_function']
        
        # Modify by temporal coherence
        coherence = self.quantum_state['temporal_coherence']
        
        # Calculate final probability
        return float(base_prob * coherence * np.random.random())
    
    def _add_quantum_uncertainty(self, state: Dict[str, Any], depth: int) -> Dict[str, Any]:
        """Add quantum uncertainty to state based on depth"""
        uncertainty_factor = depth * 0.1
        for key in state:
            if isinstance(state[key], (int, float)):
                state[key] *= (1 + uncertainty_factor * np.random.randn())
        return state
    
    async def _predict_next_state(self, current_state: Dict[str, Any], 
                                action: Dict[str, Any],
                                step: int) -> Dict[str, Any]:
        """Predict the next state based on current state and action using Gemini"""
        # If Gemini client is available, use it for enhanced prediction
        if self.gemini_client:
            try:
                # Create a prompt for Gemini to predict the next state
                prompt = self._create_prediction_prompt(current_state, action, step)
                
                # Call Gemini for prediction
                response = await self.gemini_client.generate_content_async(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.4,
                        # Lower temperature for more deterministic predictions
                    )
                )
                
                # Parse the response
                prediction = self._parse_prediction_response(response.text, current_state)
                if prediction:
                    # Merge the prediction with the current state
                    next_state = {**current_state.copy(), **prediction}
                    next_state['step'] = step
                    next_state['timestamp'] = datetime.now(timezone.utc)
                    next_state['gemini_enhanced'] = True
                    return next_state
            except Exception as e:
                # Fall back to simple prediction on error
                print(f"Gemini prediction failed: {e}. Falling back to simple prediction.")
        
        # Simple state transition as fallback
        next_state = current_state.copy()
        
        # Apply action effects
        if 'effects' in action:
            for key, effect in action['effects'].items():
                if key in next_state:
                    next_state[key] = effect
        
        # Add step-based changes
        next_state['step'] = step
        next_state['timestamp'] = datetime.now(timezone.utc)
        
        return next_state
    
    def _create_prediction_prompt(self, current_state: Dict[str, Any], 
                                action: Dict[str, Any], 
                                step: int) -> str:
        """Create a prompt for Gemini to predict the next state"""
        # Convert state and action to JSON strings for the prompt
        state_str = json.dumps(current_state, default=str, indent=2)
        action_str = json.dumps(action, default=str, indent=2)
        
        return f"""Given the current state and action, predict the next state of the system.

Current State:
{state_str}

Action to be taken:
{action_str}

Step number: {step}

Predict how the state will change after the action is taken. Consider:
1. Direct effects of the action on state variables
2. Secondary effects and interactions between variables
3. Emergent properties or patterns that might appear
4. Uncertainty and probabilistic outcomes

Return a JSON object representing the changes to the state. Include only the fields that change,
plus any new fields that should be added. The original state will be merged with your changes.

Example response format:
{{
  "metric_1": new_value,
  "metric_2": new_value,
  "new_property": value,
  "probability": 0.85
}}
"""
    
    def _parse_prediction_response(self, response_text: str, 
                                 current_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse Gemini's prediction response into a state dictionary"""
        try:
            # Try to parse as JSON
            prediction = json.loads(response_text)
            
            # Validate the prediction has at least some keys from the current state
            # or adds reasonable new keys
            if isinstance(prediction, dict):
                return prediction
            
            return None
        except json.JSONDecodeError:
            # If not valid JSON, try to extract JSON from text
            try:
                # Look for JSON-like content between curly braces
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx+1]
                    prediction = json.loads(json_str)
                    if isinstance(prediction, dict):
                        return prediction
                
                return None
            except Exception:
                return None
        except Exception:
            return None
    
    def collapse_wave_function(self, chosen_possibility: Dict[str, Any]) -> None:
        """Collapse quantum state to chosen possibility"""
        # Record collapsed state
        self.collapsed_states.append({
            'state': chosen_possibility,
            'timestamp': datetime.now(timezone.utc),
            'previous_wave_function': self.quantum_state['wave_function']
        })
        
        # Update quantum state
        self.quantum_state.update({
            'wave_function': chosen_possibility['probability'],
            'superposition': [chosen_possibility],
            'last_collapse': datetime.now(timezone.utc)
        })
        
        # Adjust temporal coherence
        time_since_last = (datetime.now(timezone.utc) - 
                          self.quantum_state['last_collapse']).total_seconds()
        self.quantum_state['temporal_coherence'] *= np.exp(-time_since_last / 3600)
        
        # Clear active simulations
        self.active_simulations = []

@dataclass
class ComplexityLayer:
    """Represents a layer in the fractal complexity system"""
    layer_type: str
    previous_layer: Optional['ComplexityLayer'] = None
    complexity_score: float = 0.0
    patterns: List[Dict[str, Any]] = field(default_factory=list)
    emergent_properties: Dict[str, Any] = field(default_factory=dict)
    creation_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def interact_with_previous(self) -> None:
        """Interact with previous layer to generate emergence"""
        if self.previous_layer:
            shared_patterns = self._find_shared_patterns()
            new_properties = self._generate_emergence(shared_patterns)
            self.emergent_properties.update(new_properties)
            self.complexity_score = len(self.emergent_properties) * 0.1
    
    def _find_shared_patterns(self) -> List[Dict[str, Any]]:
        """Find patterns shared with previous layer"""
        if not self.previous_layer:
            return []
            
        shared = []
        for pattern in self.patterns:
            if pattern in self.previous_layer.patterns:
                shared.append(pattern)
        
        return shared
    
    def _generate_emergence(self, shared_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate emergent properties from shared patterns"""
        properties = {}
        for pattern in shared_patterns:
            property_key = f"emergent_{pattern['type']}"
            properties[property_key] = {
                'strength': len(shared_patterns) * 0.1,
                'origin_patterns': shared_patterns
            }
        return properties

class FractalComplexitySystem:
    """Manages fractal complexity layers and emergent properties"""
    
    def __init__(self):
        self.layers: List[ComplexityLayer] = []
        self.complexity_network = nx.DiGraph()
        self.emergence_threshold = SETTINGS['consciousness']['complexity_emergence_threshold']
        self.creation_time = SETTINGS.get_current_time()
        
    def add_layer(self, layer_type: str) -> None:
        """Add new complexity layer"""
        previous = self.layers[-1] if self.layers else None
        new_layer = ComplexityLayer(
            layer_type=layer_type,
            previous_layer=previous
        )
        
        # Interact with previous layer
        new_layer.interact_with_previous()
        
        # Add to layers list
        self.layers.append(new_layer)
        
        # Update complexity network
        self._update_complexity_network()
        
    def _update_complexity_network(self) -> None:
        """Update the complexity network with new layer interactions"""
        # Add layer node
        layer = self.layers[-1]
        self.complexity_network.add_node(
            layer.layer_type,
            complexity=layer.complexity_score,
            timestamp=layer.creation_time
        )
        
        # Add edges to previous layers
        if layer.previous_layer:
            self.complexity_network.add_edge(
                layer.previous_layer.layer_type,
                layer.layer_type,
                weight=len(layer.emergent_properties)
            )
            
        # Calculate network metrics
        centrality = nx.degree_centrality(self.complexity_network)
        clustering = nx.clustering(self.complexity_network)
        
        # Update layer properties
        for node in self.complexity_network.nodes():
            self.complexity_network.nodes[node].update({
                'centrality': centrality.get(node, 0),
                'clustering': clustering.get(node, 0)
            })
