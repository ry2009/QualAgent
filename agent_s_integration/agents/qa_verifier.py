"""
QA Verifier Agent - Extends Agent-S with QualGent verification capabilities
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# QualGent imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from src.core.llm_client import LLMClient, LLMMessage
from src.models.result import AgentDecision, DecisionType, AgentType
from src.core.android_integration import AndroidWorldIntegration

logger = logging.getLogger("qa_verifier_agent")


class QAVerifierAgent:
    """
    QA Verifier Agent that combines Agent-S framework with QualGent verification
    
    Responsibilities:
    - Verify test execution results against expected outcomes
    - Detect functional bugs and UI issues  
    - Validate Android app behavior
    - Provide detailed verification reports
    """
    
    def __init__(self, llm_client_config: Dict[str, Any], android_integration: AndroidWorldIntegration):
        self.llm_client = LLMClient(
            provider=llm_client_config["provider"],
            model=llm_client_config["model"],
            api_key=llm_client_config["api_key"]
        )
        
        self.android_integration = android_integration
        self.config = llm_client_config
        self.is_active = False
        
        logger.info("QA Verifier Agent initialized")
    
    async def activate(self):
        """Activate the verifier agent"""
        self.is_active = True
        logger.info("QA Verifier Agent activated")
    
    async def deactivate(self):
        """Deactivate the verifier agent"""
        self.is_active = False
        logger.info("QA Verifier Agent deactivated")
    
    async def process(self, verification_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process verification request with Agent-S integration
        
        Args:
            verification_context: Context containing goal, execution result, etc.
            
        Returns:
            Verification result with pass/fail status and detailed analysis
        """
        
        if not self.is_active:
            await self.activate()
        
        start_time = datetime.now()
        
        try:
            # Extract context
            goal = verification_context.get("goal")
            execution_result = verification_context.get("execution_result")
            expected_outcome = verification_context.get("expected_outcome")
            
            # Get current UI state for verification
            current_ui_state = await self.android_integration.get_current_ui_state()
            
            # Perform verification analysis
            verification_result = await self._perform_verification_analysis(
                goal, execution_result, expected_outcome, current_ui_state
            )
            
            # Create agent decision record
            decision = AgentDecision(
                agent_type=AgentType.VERIFIER,
                decision_type=DecisionType.VERIFICATION,
                input_data={
                    "goal_title": goal.title if goal else "unknown",
                    "execution_success": execution_result.get("success", False),
                    "ui_elements_count": len(current_ui_state.elements) if current_ui_state else 0
                },
                output_data=verification_result,
                reasoning=verification_result.get("reasoning", ""),
                confidence_score=verification_result.get("confidence", 0.0),
                execution_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                model_used=self.config["model"]
            )
            
            return verification_result
            
        except Exception as e:
            logger.error(f"Verification processing failed: {e}")
            return {
                "passed": False,
                "confidence": 0.0,
                "reasoning": f"Verification failed due to error: {str(e)}",
                "bugs_detected": [],
                "recommendations": ["Retry verification after fixing system error"]
            }
        finally:
            await self.deactivate()
    
    async def _perform_verification_analysis(
        self, 
        goal, 
        execution_result: Dict[str, Any], 
        expected_outcome: str,
        current_ui_state
    ) -> Dict[str, Any]:
        """Perform detailed verification analysis using LLM"""
        
        # Prepare verification prompt
        verification_prompt = self._create_verification_prompt(
            goal, execution_result, expected_outcome, current_ui_state
        )
        
        # Get LLM analysis
        messages = [LLMMessage(role="user", content=verification_prompt)]
        
        llm_response = await self.llm_client.generate_response(
            messages=messages,
            temperature=self.config.get("temperature", 0.1),
            max_tokens=self.config.get("max_tokens", 1500)
        )
        
        # Parse LLM response into structured result
        verification_result = self._parse_verification_response(llm_response.content)
        
        # Add technical verification details
        verification_result.update(await self._add_technical_verification(
            goal, execution_result, current_ui_state
        ))
        
        return verification_result
    
    def _create_verification_prompt(self, goal, execution_result: Dict[str, Any], 
                                  expected_outcome: str, current_ui_state) -> str:
        """Create verification prompt for LLM analysis"""
        
        prompt = f"""
You are a QA Verification Agent analyzing mobile app test execution results.

TEST GOAL:
Title: {goal.title if goal else 'Unknown'}
Description: {goal.description if goal else 'No description'}
Test Type: {goal.test_type if goal else 'Unknown'}
Expected Outcome: {expected_outcome}

EXECUTION RESULT:
Success: {execution_result.get('success', False)}
Actions Executed: {len(execution_result.get('actions_executed', []))}
Error Message: {execution_result.get('error_message', 'None')}

CURRENT UI STATE:
Total Elements: {len(current_ui_state.elements) if current_ui_state else 0}
Current App: {current_ui_state.current_app if current_ui_state else 'Unknown'}
Current Activity: {current_ui_state.current_activity if current_ui_state else 'Unknown'}

VERIFICATION TASKS:
1. Analyze if the execution achieved the expected outcome
2. Check for functional bugs or UI issues
3. Validate app behavior against test requirements
4. Assess overall test success/failure
5. Provide improvement recommendations

RESPONSE FORMAT (JSON):
{{
    "passed": boolean,
    "confidence": float (0.0-1.0),
    "reasoning": "detailed explanation",
    "bugs_detected": [
        {{
            "type": "bug_type",
            "severity": "low|medium|high|critical", 
            "description": "bug description",
            "element_affected": "element_id_if_applicable"
        }}
    ],
    "ui_issues": [
        {{
            "type": "issue_type",
            "description": "UI issue description",
            "impact": "user experience impact"
        }}
    ],
    "recommendations": ["list of recommendations"],
    "test_coverage": {{
        "areas_tested": ["list of areas"],
        "areas_missed": ["list of missed areas"]
    }}
}}

Provide thorough analysis focusing on mobile QA best practices.
"""
        
        return prompt
    
    def _parse_verification_response(self, response_content: str) -> Dict[str, Any]:
        """Parse LLM response into structured verification result"""
        
        try:
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                # Validate required fields
                result.setdefault("passed", False)
                result.setdefault("confidence", 0.0)
                result.setdefault("reasoning", "No reasoning provided")
                result.setdefault("bugs_detected", [])
                result.setdefault("ui_issues", [])
                result.setdefault("recommendations", [])
                result.setdefault("test_coverage", {"areas_tested": [], "areas_missed": []})
                
                return result
            else:
                # Fallback parsing
                return self._fallback_parse_verification(response_content)
                
        except Exception as e:
            logger.warning(f"Failed to parse verification response: {e}")
            return self._fallback_parse_verification(response_content)
    
    def _fallback_parse_verification(self, response_content: str) -> Dict[str, Any]:
        """Fallback parsing when JSON parsing fails"""
        
        # Simple keyword-based parsing
        passed = "passed" in response_content.lower() or "success" in response_content.lower()
        
        # Extract confidence if mentioned
        confidence = 0.5  # Default
        import re
        confidence_match = re.search(r'confidence[:\s]+([0-9.]+)', response_content.lower())
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                if confidence > 1.0:
                    confidence = confidence / 100.0  # Handle percentage
            except:
                pass
        
        return {
            "passed": passed,
            "confidence": confidence,
            "reasoning": response_content[:500] + "..." if len(response_content) > 500 else response_content,
            "bugs_detected": [],
            "ui_issues": [],
            "recommendations": ["Manual review recommended due to parsing issues"],
            "test_coverage": {"areas_tested": [], "areas_missed": []}
        }
    
    async def _add_technical_verification(self, goal, execution_result: Dict[str, Any], 
                                        current_ui_state) -> Dict[str, Any]:
        """Add technical verification details"""
        
        technical_details = {
            "execution_time_ms": 0,
            "actions_verified": 0,
            "ui_elements_analyzed": 0,
            "screenshots_captured": 0
        }
        
        try:
            # Calculate execution time
            if "start_time" in execution_result and "end_time" in execution_result:
                duration = execution_result["end_time"] - execution_result["start_time"]
                technical_details["execution_time_ms"] = int(duration.total_seconds() * 1000)
            
            # Count verified actions
            technical_details["actions_verified"] = len(execution_result.get("actions_executed", []))
            
            # Count UI elements analyzed
            if current_ui_state:
                technical_details["ui_elements_analyzed"] = len(current_ui_state.elements)
            
            # Check for screenshots
            screenshot_actions = [
                action for action in execution_result.get("actions_executed", [])
                if action.get("type") == "screenshot"
            ]
            technical_details["screenshots_captured"] = len(screenshot_actions)
            
        except Exception as e:
            logger.warning(f"Failed to add technical verification details: {e}")
        
        return technical_details
    
    async def detect_ui_regression(self, baseline_ui_state, current_ui_state) -> List[Dict[str, Any]]:
        """Detect UI regressions by comparing states"""
        
        regressions = []
        
        if not baseline_ui_state or not current_ui_state:
            return regressions
        
        try:
            # Compare element counts
            baseline_count = len(baseline_ui_state.elements)
            current_count = len(current_ui_state.elements)
            
            if abs(baseline_count - current_count) > 5:  # Threshold for significant change
                regressions.append({
                    "type": "element_count_change",
                    "severity": "medium",
                    "description": f"UI element count changed from {baseline_count} to {current_count}",
                    "baseline_count": baseline_count,
                    "current_count": current_count
                })
            
            # Compare app context
            if baseline_ui_state.current_app != current_ui_state.current_app:
                regressions.append({
                    "type": "app_context_change",
                    "severity": "high",
                    "description": f"App changed unexpectedly from {baseline_ui_state.current_app} to {current_ui_state.current_app}",
                    "expected_app": baseline_ui_state.current_app,
                    "actual_app": current_ui_state.current_app
                })
                
        except Exception as e:
            logger.warning(f"UI regression detection failed: {e}")
        
        return regressions
    
    def get_verification_metrics(self) -> Dict[str, Any]:
        """Get verification performance metrics"""
        return {
            "agent_type": "qa_verifier",
            "is_active": self.is_active,
            "model_used": self.config.get("model", "unknown"),
            "provider": self.config.get("provider", "unknown")
        } 