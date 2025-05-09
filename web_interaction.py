# web_interaction.py

import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pyautogui
import time
from typing import Dict, List, Optional, Any
import google.generativeai as genai
import json

class WebInteractionAgent:
    def __init__(self, system_state, gemini_client=None):
        self.browser = ComputerInteractionSystem()
        self.system_state = system_state
        self.gemini_client = gemini_client
        self.action_history = []
        self.known_elements = {}
        self.safety_checks = SafetyChecker()

    async def analyze_task(self, task: str) -> Dict:
        """Analyze web interaction task and break it down into steps using Gemini"""
        prompt = f"""Analyze this web interaction task and break it down into detailed steps for browser automation:

Task: {task}

Please analyze this web task and provide a structured response with the following:

1. Required browser actions (navigation, clicks, form inputs, etc.)
2. UI elements that need to be interacted with (with descriptions of how to locate them)
3. Potential challenges or edge cases to handle
4. Verification steps to confirm each action was successful
5. Safety considerations and checks

Return your analysis as a JSON object with the following structure:
{{
  "task_summary": "Brief summary of the task",
  "steps": [
    {{
      "type": "navigation",
      "url": "https://example.com",
      "description": "Navigate to example.com"
    }},
    {{
      "type": "click",
      "element": "login_button",
      "element_description": "Button with text 'Login' in the top right corner",
      "verification": {{
        "element_present": "//div[contains(@class, 'login-form')]",
        "url_contains": "login"
      }}
    }},
    {{
      "type": "input",
      "element": "username_field",
      "element_description": "Input field with label 'Username'",
      "text": "username_value_here"
    }}
  ],
  "challenges": [
    "Potential CAPTCHA on login form",
    "Page might have dynamic loading elements"
  ],
  "safety_checks": [
    "Ensure no sensitive data is being transmitted insecurely",
    "Verify domain matches expected target"
  ]
}}
"""
        
        if self.gemini_client:
            try:
                response = await self.gemini_client.generate_content_async(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2,
                    )
                )
                return self._parse_task_analysis(response.text)
            except Exception as e:
                logging.error(f"Gemini task analysis failed: {str(e)}")
                # Fall back to simple analysis
                return self._create_simple_analysis(task)
        else:
            logging.warning("No Gemini client available for web task analysis")
            return self._create_simple_analysis(task)

    def _parse_task_analysis(self, content: str) -> Dict:
        """Parse Gemini's analysis response into structured format"""
        try:
            # Try to parse as JSON directly
            analysis = json.loads(content)
            if isinstance(analysis, dict) and 'steps' in analysis:
                return analysis
        except json.JSONDecodeError:
            # If not valid JSON, try to extract JSON from text
            try:
                # Look for JSON-like content between curly braces
                start_idx = content.find('{')
                end_idx = content.rfind('}')
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx+1]
                    analysis = json.loads(json_str)
                    if isinstance(analysis, dict) and 'steps' in analysis:
                        return analysis
            except Exception:
                pass
        
        # Fallback to simple parsing if JSON extraction fails
        steps = []
        lines = content.split('\n')
        current_step = None
        
        for line in lines:
            if line.strip().startswith(('- ', '* ', '1. ', '2. ', '3. ')):
                if 'navigate' in line.lower() or 'go to' in line.lower():
                    steps.append({
                        'type': 'navigation',
                        'description': line.strip(),
                        'url': self._extract_url(line)
                    })
                elif 'click' in line.lower():
                    steps.append({
                        'type': 'click',
                        'description': line.strip(),
                        'element': self._extract_element(line)
                    })
                elif 'input' in line.lower() or 'type' in line.lower() or 'enter' in line.lower():
                    steps.append({
                        'type': 'input',
                        'description': line.strip(),
                        'element': self._extract_element(line),
                        'text': self._extract_text(line)
                    })
        
        return {
            'task_summary': content.split('\n')[0] if content else 'Web task',
            'steps': steps,
            'challenges': [],
            'safety_checks': []
        }
    
    def _create_simple_analysis(self, task: str) -> Dict:
        """Create a simple task analysis when Gemini is unavailable"""
        # Extract potential URLs
        urls = self._extract_urls_from_text(task)
        
        steps = []
        if urls:
            steps.append({
                'type': 'navigation',
                'url': urls[0],
                'description': f"Navigate to {urls[0]}"
            })
        
        return {
            'task_summary': task,
            'steps': steps,
            'challenges': [],
            'safety_checks': []
        }
    
    def _extract_url(self, text: str) -> str:
        """Extract URL from text"""
        # Simple extraction - could be enhanced with regex
        words = text.split()
        for word in words:
            if word.startswith(('http://', 'https://', 'www.')):
                return word
        return ""
    
    def _extract_urls_from_text(self, text: str) -> List[str]:
        """Extract all URLs from text"""
        urls = []
        words = text.split()
        for word in words:
            if word.startswith(('http://', 'https://', 'www.')):
                urls.append(word)
        return urls
    
    def _extract_element(self, text: str) -> str:
        """Extract element description from text"""
        # Simple extraction - could be enhanced with NLP
        if 'button' in text.lower():
            return 'button'
        elif 'input' in text.lower() or 'field' in text.lower():
            return 'input_field'
        elif 'link' in text.lower():
            return 'link'
        return 'element'
    
    def _extract_text(self, text: str) -> str:
        """Extract text to input from text"""
        # Simple extraction - could be enhanced with NLP
        if '"' in text:
            parts = text.split('"')
            if len(parts) >= 3:
                return parts[1]
        return ""

    async def execute_web_task(self, task: str) -> Dict:
        """Execute web interaction task"""
        try:
            # Analyze task
            analysis = await self.analyze_task(task)
            
            # Create action plan
            action_plan = self._create_action_plan(analysis)
            
            # Execute actions
            results = await self._execute_action_plan(action_plan)
            
            # Extract data from page if needed
            if self.gemini_client:
                extracted_data = await self._extract_data_with_gemini(task)
                results['extracted_data'] = extracted_data
            
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

    async def _extract_data_with_gemini(self, task: str) -> Dict[str, Any]:
        """Extract structured data from web page using Gemini"""
        if not self.gemini_client:
            return {}
            
        try:
            # Get page content
            page_content = self.browser.get_page_content()
            
            # Create prompt for Gemini
            prompt = f"""Extract relevant information from this web page content based on the task:

Task: {task}

Web Page Content:
{page_content[:5000]}  # Limit content length to avoid token limits

Analyze the content and extract structured information relevant to the task.
Return the data as a JSON object with appropriate keys and values.
"""
            
            response = await self.gemini_client.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for factual extraction
                )
            )
            
            # Parse response
            try:
                # Try to parse as JSON
                extracted_data = json.loads(response.text)
                if isinstance(extracted_data, dict):
                    return extracted_data
            except json.JSONDecodeError:
                # If not valid JSON, try to extract JSON from text
                try:
                    start_idx = response.text.find('{')
                    end_idx = response.text.rfind('}')
                    
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = response.text[start_idx:end_idx+1]
                        extracted_data = json.loads(json_str)
                        if isinstance(extracted_data, dict):
                            return extracted_data
                except Exception:
                    pass
            
            # If JSON parsing fails, return the raw text
            return {'raw_extraction': response.text}
            
        except Exception as e:
            logging.error(f"Data extraction with Gemini failed: {str(e)}")
            return {'error': str(e)}

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
