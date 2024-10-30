# agents/analysis.py

from typing import Dict
import openai

class AnalysisAgent:
    def __init__(self, client):
        self.client = client
        self.analysis_history = []
        
    async def analyze_state(self, state: Dict) -> Dict:
        """Analyze current system state"""
        prompt = f"""Analyze the current system state:
        1. Key metrics evaluation
        2. Areas needing improvement
        3. Recommended actions
        4. Resource requirements
        5. Quantum state assessment
        6. Complexity layer analysis
        
        State: {state}
        """
        
        response = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a system analysis expert."},
                {"role": "user", "content": prompt}
            ]
        )
        
        analysis = self._parse_analysis(response.choices[0].message.content)
        self.analysis_history.append({
            'state': state,
            'analysis': analysis,
            'timestamp': time.time()
        })
        
        return analysis
        
    def _parse_analysis(self, content: str) -> Dict:
        """Parse analysis response"""
        return {
            'analysis': content,
            'timestamp': time.time(),
            'confidence': 0.9
        }