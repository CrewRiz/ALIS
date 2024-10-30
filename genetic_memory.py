# memory/genetic_memory.py

class LamarckianRule(Rule):
    def __init__(self, rule_text, parent_rules=None):
        super().__init__(rule_text)
        self.parent_rules = parent_rules or []
        self.acquired_traits = {}
        self.experience_log = []
        self.meta_state = {
            'attention_focus': 1.0,
            'self_awareness': 0.0,
            'subjective_value': 0.0
        }
        self.adaptation_history = []
        
    def adapt_from_experience(self, experience):
        """Modify rule structure based on experience"""
        self.experience_log.append(experience)
        
        # Update acquired traits
        for trait, value in experience.traits.items():
            if trait not in self.acquired_traits:
                self.acquired_traits[trait] = value
            else:
                self.acquired_traits[trait] = (
                    0.7 * self.acquired_traits[trait] + 
                    0.3 * value
                )
                
        # Modify rule structure
        self._modify_structure(experience)
        
        # Update meta state
        self._update_meta_state(experience)


