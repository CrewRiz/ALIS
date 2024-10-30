# web_interaction.py

import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pyautogui
import time
from typing import Dict, List
import anthropic

class WebInteractionAgent:
    def __init__(self, system_state):
        self.browser = ComputerInteractionSystem()
        self.system_state = system_state
        self.llm_client = anthropic.Client(api_key="your_key")
        self.action_history = []
        self.known_elements = {}
        self.safety_checks = SafetyChecker()

    async def analyze_task(self, task: str) -> Dict:
        """Analyze web interaction task and break it down into steps"""
        prompt = f"""Analyze this web interaction task and break it down into steps:
        Task: {task}
        Consider:
        1. Required browser actions
        2. UI elements needed
        3. Potential challenges
        4. Verification steps
        5. Safety considerations
        """
        
        response = await self.llm_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse_task_analysis(response.content)

    async def execute_web_task(self, task: str) -> Dict:
        """Execute web interaction task"""
        try:
            # Analyze task
            analysis = await self.analyze_task(task)
            
            # Create action plan
            action_plan = self._create_action_plan(analysis)
            
            # Execute actions
            results = await self._execute_action_plan(action_plan)
            
            # Update system state
            self._update_system_state(results)
            
            # Log success
            self._log_action(f"Successfully executed web task: {task}")
            
            return {
                'status': 'success',
                'results': results,
                'action_history': self.action_history[-10:]
            }
            
        except Exception as e:
            self._log_error(f"Web task execution failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _create_action_plan(self, analysis: Dict) -> List[Dict]:
        """Create detailed action plan from analysis"""
        actions = []
        
        for step in analysis['steps']:
            if step['type'] == 'navigation':
                if self.safety_checks.is_safe_url(step['url']):
                    actions.append({
                        'action': 'navigate',
                        'url': step['url']
                    })
                    
            elif step['type'] == 'click':
                actions.append({
                    'action': 'click',
                    'element': step['element'],
                    'verification': step.get('verification')
                })
                
            elif step['type'] == 'input':
                if self.safety_checks.is_safe_content(step['text']):
                    actions.append({
                        'action': 'input',
                        'text': step['text'],
                        'element': step['element']
                    })
                    
        return actions

    async def _execute_action_plan(self, action_plan: List[Dict]) -> Dict:
        """Execute action plan safely"""
        results = {
            'completed_actions': [],
            'failed_actions': [],
            'extracted_data': {}
        }
        
        for action in action_plan:
            try:
                if action['action'] == 'navigate':
                    self.browser.browse_web(action['url'])
                    results['completed_actions'].append(action)
                    
                elif action['action'] == 'click':
                    self.browser.click_element(
                        image_path=action.get('element_image'),
                        coordinates=action.get('coordinates')
                    )
                    results['completed_actions'].append(action)
                    
                elif action['action'] == 'input':
                    self.browser.type_text(action['text'])
                    results['completed_actions'].append(action)
                    
                # Wait for verification if specified
                if action.get('verification'):
                    self._verify_action(action['verification'])
                    
            except Exception as e:
                results['failed_actions'].append({
                    'action': action,
                    'error': str(e)
                })
                
        return results

    def _verify_action(self, verification_data: Dict):
        """Verify action completion"""
        try:
            if verification_data.get('element_present'):
                WebDriverWait(self.browser.driver, 10).until(
                    EC.presence_of_element_located(
                        (By.XPATH, verification_data['element_present'])
                    )
                )
            if verification_data.get('url_contains'):
                WebDriverWait(self.browser.driver, 10).until(
                    EC.url_contains(verification_data['url_contains'])
                )
        except Exception as e:
            raise Exception(f"Verification failed: {str(e)}")

    def _update_system_state(self, results: Dict):
        """Update system state with interaction results"""
        self.system_state.update({
            'metrics': {
                'web_interactions': len(results['completed_actions']),
                'failed_actions': len(results['failed_actions'])
            },
            'actions': results['completed_actions']
        })

    def _log_action(self, message: str, level: str = "INFO"):
        """Log action with timestamp"""
        self.action_history.append({
            'timestamp': time.time(),
            'action': message,
            'level': level
        })
        if level == "INFO":
            logging.info(message)
        elif level == "ERROR":
            logging.error(message)

    def _log_error(self, message: str):
        """Log error with timestamp"""
        self._log_action(message, level="ERROR")
