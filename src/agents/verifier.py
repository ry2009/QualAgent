import json
import time
from typing import Dict, Any, List, Optional, Tuple
import logging
import asyncio

from .base import BaseAgent, AgentConfig
from ..models.result import AgentDecision, AgentType, DecisionType, ExecutionResult, ResultStatus
from ..models.task import SubGoal, TestGoal
from ..models.ui_state import UIState, UIElement
from ..core.llm_client import LLMClient, LLMMessage

class VerificationResult:
    """Result of a verification check"""
    def __init__(self, 
                 passed: bool, 
                 confidence: float, 
                 reasoning: str,
                 evidence: Dict[str, Any] = None,
                 bug_detected: bool = False,
                 bug_details: Dict[str, Any] = None):
        self.passed = passed
        self.confidence = confidence
        self.reasoning = reasoning
        self.evidence = evidence or {}
        self.bug_detected = bug_detected
        self.bug_details = bug_details or {}

class VerifierAgent(BaseAgent):
    """Agent responsible for verifying app behavior and detecting bugs"""
    
    def __init__(self, config: AgentConfig, logger: Optional[logging.Logger] = None):
        super().__init__(AgentType.VERIFIER, config, logger)
        
    def _initialize_agent(self):
        """Initialize verifier-specific components"""
        self.llm_client = LLMClient(
            provider=self.config.provider,
            model=self.config.model_name,
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )
        
        # Verification strategies (simplified for now)
        self.verification_strategies = {}
        
        # Bug detection patterns (simplified for now)
        self.bug_patterns = {
            "crash": self._detect_crash,
            "freeze": self._detect_freeze,
            "ui_inconsistency": self._detect_ui_inconsistency,
            "accessibility": self._detect_accessibility_issues
        }
        
        # State tracking for bug detection
        self.ui_state_history: List[UIState] = []
        self.performance_history: List[Dict[str, Any]] = []
        self.max_history_size = 50
        
    async def make_decision(self, context: Dict[str, Any], ui_state: Optional[UIState] = None) -> AgentDecision:
        """Make a verification decision based on execution result and current state"""
        verification_type = context.get("verification_type", "complete")
        
        if verification_type == "subgoal":
            return await self._verify_subgoal_execution(context, ui_state)
        elif verification_type == "goal":
            return await self._verify_goal_completion(context, ui_state)
        elif verification_type == "bug_check":
            return await self._perform_bug_detection(context, ui_state)
        elif verification_type == "state_validation":
            return await self._validate_app_state(context, ui_state)
        else:
            return await self._perform_complete_verification(context, ui_state)
    
    async def _verify_subgoal_execution(self, context: Dict[str, Any], ui_state: Optional[UIState]) -> AgentDecision:
        """Verify that a subgoal was executed correctly"""
        subgoal_data = context.get("subgoal")
        execution_result = context.get("execution_result")
        expected_outcome = context.get("expected_outcome")
        
        if not subgoal_data or not execution_result:
            return self.create_decision(
                decision_type=DecisionType.VERIFICATION,
                reasoning="Missing subgoal or execution result data",
                confidence=0.0,
                output_data={"error": "missing_data"}
            )
        
        # Convert dict to SubGoal if needed
        if isinstance(subgoal_data, dict):
            subgoal = self._dict_to_subgoal(subgoal_data)
        else:
            subgoal = subgoal_data
        
        # Perform verification based on subgoal type
        verification_result = await self._verify_subgoal_outcome(subgoal, execution_result, ui_state)
        
        # Check for bugs during verification
        bug_check_result = await self._perform_targeted_bug_detection(subgoal, ui_state)
        
        # Combine results
        overall_passed = verification_result.passed and not bug_check_result.bug_detected
        confidence = min(verification_result.confidence, bug_check_result.confidence)
        
        reasoning = f"Subgoal verification: {verification_result.reasoning}"
        if bug_check_result.bug_detected:
            reasoning += f"; Bug detected: {bug_check_result.reasoning}"
        
        return self.create_decision(
            decision_type=DecisionType.VERIFICATION,
            reasoning=reasoning,
            confidence=confidence,
            output_data={
                "verification_passed": overall_passed,
                "subgoal_verified": verification_result.passed,
                "bug_detected": bug_check_result.bug_detected,
                "verification_evidence": verification_result.evidence,
                "bug_details": bug_check_result.bug_details,
                "recommendations": self._generate_recommendations(verification_result, bug_check_result)
            },
            input_data={
                "subgoal_id": subgoal.id,
                "action_type": subgoal.action_type,
                "expected_result": subgoal.expected_result
            }
        )
    
    async def _verify_subgoal_outcome(self, subgoal: SubGoal, execution_result: Dict[str, Any], 
                                    ui_state: Optional[UIState]) -> VerificationResult:
        """Verify the outcome of a specific subgoal"""
        action_type = subgoal.action_type
        expected_result = subgoal.expected_result
        
        if action_type == "touch":
            return await self._verify_touch_outcome(subgoal, execution_result, ui_state)
        elif action_type == "type":
            return await self._verify_type_outcome(subgoal, execution_result, ui_state)
        elif action_type == "scroll":
            return await self._verify_scroll_outcome(subgoal, execution_result, ui_state)
        elif action_type == "wait":
            return await self._verify_wait_outcome(subgoal, execution_result, ui_state)
        elif action_type == "verify":
            return await self._verify_verification_outcome(subgoal, execution_result, ui_state)
        else:
            return VerificationResult(
                passed=execution_result.get("status") == "success",
                confidence=0.5,
                reasoning=f"Generic verification for {action_type} action"
            )
    
    async def _verify_touch_outcome(self, subgoal: SubGoal, execution_result: Dict[str, Any], 
                                  ui_state: Optional[UIState]) -> VerificationResult:
        """Verify touch action outcome"""
        expected_result = subgoal.expected_result
        
        if not expected_result:
            return VerificationResult(
                passed=execution_result.get("status") == "success",
                confidence=0.7,
                reasoning="Touch executed successfully (no specific expectation)"
            )
        
        # Use LLM to analyze if the current state matches expectations
        verification_result = await self._llm_verify_expectation(
            action="touch",
            expectation=expected_result,
            ui_state=ui_state,
            execution_result=execution_result
        )
        
        return verification_result
    
    async def _verify_type_outcome(self, subgoal: SubGoal, execution_result: Dict[str, Any], 
                                 ui_state: Optional[UIState]) -> VerificationResult:
        """Verify type action outcome"""
        typed_text = execution_result.get("text_entered")
        expected_text = subgoal.parameters.get("text")
        
        # Check if the text was entered correctly
        if typed_text and expected_text:
            text_matches = typed_text == expected_text
            confidence = 0.9 if text_matches else 0.3
            reasoning = f"Text entry {'successful' if text_matches else 'failed'}: expected '{expected_text}', got '{typed_text}'"
        else:
            text_matches = True
            confidence = 0.7
            reasoning = "Text entry completed (no text validation available)"
        
        # Check if text appears in UI elements
        ui_text_present = False
        if ui_state and expected_text:
            elements_with_text = ui_state.find_elements_by_text(expected_text, exact_match=False)
            ui_text_present = len(elements_with_text) > 0
            if ui_text_present:
                confidence = min(confidence + 0.1, 1.0)
                reasoning += f"; Text found in {len(elements_with_text)} UI elements"
        
        return VerificationResult(
            passed=text_matches and (ui_text_present or not expected_text),
            confidence=confidence,
            reasoning=reasoning,
            evidence={
                "typed_text": typed_text,
                "expected_text": expected_text,
                "ui_text_present": ui_text_present
            }
        )
    
    async def _verify_scroll_outcome(self, subgoal: SubGoal, execution_result: Dict[str, Any], 
                                   ui_state: Optional[UIState]) -> VerificationResult:
        """Verify scroll action outcome"""
        scroll_distance = execution_result.get("scroll_distance")
        expected_result = subgoal.expected_result
        
        # Basic verification: scroll was executed
        scroll_executed = scroll_distance is not None
        
        if not expected_result:
            return VerificationResult(
                passed=scroll_executed,
                confidence=0.8,
                reasoning=f"Scroll executed {'successfully' if scroll_executed else 'failed'}"
            )
        
        # Use LLM to verify if scroll achieved expected result
        verification_result = await self._llm_verify_expectation(
            action="scroll",
            expectation=expected_result,
            ui_state=ui_state,
            execution_result=execution_result
        )
        
        return verification_result
    
    async def _verify_wait_outcome(self, subgoal: SubGoal, execution_result: Dict[str, Any], 
                                 ui_state: Optional[UIState]) -> VerificationResult:
        """Verify wait action outcome"""
        duration = subgoal.parameters.get("duration_ms", 1000)
        execution_time = execution_result.get("execution_time_ms", 0)
        
        # Check if wait duration was approximately correct
        duration_correct = abs(execution_time - duration) < (duration * 0.2)  # 20% tolerance
        
        condition = subgoal.parameters.get("condition")
        if condition:
            # Verify that the waited-for condition is now met
            condition_met = await self._check_wait_condition(condition, ui_state)
            return VerificationResult(
                passed=condition_met,
                confidence=0.8 if condition_met else 0.3,
                reasoning=f"Wait condition {'met' if condition_met else 'not met'}: {condition}"
            )
        else:
            return VerificationResult(
                passed=duration_correct,
                confidence=0.9,
                reasoning=f"Wait duration {'correct' if duration_correct else 'incorrect'}: {execution_time}ms vs {duration}ms"
            )
    
    async def _verify_verification_outcome(self, subgoal: SubGoal, execution_result: Dict[str, Any], 
                                         ui_state: Optional[UIState]) -> VerificationResult:
        """Verify verification action outcome"""
        verification_passed = execution_result.get("verification_passed", False)
        actual_outcome = execution_result.get("actual_outcome", "")
        
        return VerificationResult(
            passed=verification_passed,
            confidence=0.9,
            reasoning=f"Verification {'passed' if verification_passed else 'failed'}: {actual_outcome}",
            evidence={
                "verification_type": subgoal.parameters.get("type"),
                "actual_outcome": actual_outcome
            }
        )
    
    async def _llm_verify_expectation(self, action: str, expectation: str, 
                                    ui_state: Optional[UIState], 
                                    execution_result: Dict[str, Any]) -> VerificationResult:
        """Use LLM to verify if the current state matches expectations"""
        try:
            # Prepare context for LLM
            ui_summary = ui_state.get_interactable_elements_summary() if ui_state else {}
            
            system_prompt = """
            You are an expert QA verifier. Your job is to determine if the current app state 
            matches the expected outcome after an action was performed.
            
            Analyze the UI state and execution result to determine:
            1. Whether the expectation was met (true/false)
            2. Your confidence level (0.0 to 1.0)
            3. Clear reasoning for your decision
            4. Any evidence that supports your conclusion
            """
            
            user_prompt = f"""
            Action performed: {action}
            Expected outcome: {expectation}
            
            Execution result:
            {json.dumps(execution_result, indent=2)}
            
            Current UI state:
            {json.dumps(ui_summary, indent=2)}
            
            Did the action achieve the expected outcome?
            """
            
            response_schema = {
                "type": "object",
                "properties": {
                    "expectation_met": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reasoning": {"type": "string"},
                    "evidence": {"type": "object"}
                },
                "required": ["expectation_met", "confidence", "reasoning"]
            }
            
            messages = self.llm_client.create_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            response = await self.llm_client.generate_structured_response(
                messages=messages,
                response_schema=response_schema,
                temperature=0.1
            )
            
            return VerificationResult(
                passed=response.get("expectation_met", False),
                confidence=response.get("confidence", 0.5),
                reasoning=response.get("reasoning", "LLM verification completed"),
                evidence=response.get("evidence", {})
            )
            
        except Exception as e:
            self.logger.error(f"LLM verification failed: {e}")
            return VerificationResult(
                passed=False,
                confidence=0.1,
                reasoning=f"LLM verification error: {str(e)}"
            )
    
    async def _perform_targeted_bug_detection(self, subgoal: SubGoal, ui_state: Optional[UIState]) -> VerificationResult:
        """Perform targeted bug detection based on the action type"""
        bugs_detected = []
        confidence_scores = []
        
        # Update state history
        if ui_state:
            self._update_state_history(ui_state)
        
        # Check for different types of bugs
        for bug_type, detector in self.bug_patterns.items():
            try:
                bug_result = await detector(subgoal, ui_state)
                if bug_result.get("detected", False):
                    bugs_detected.append({
                        "type": bug_type,
                        "severity": bug_result.get("severity", "medium"),
                        "description": bug_result.get("description", ""),
                        "evidence": bug_result.get("evidence", {})
                    })
                confidence_scores.append(bug_result.get("confidence", 0.5))
            except Exception as e:
                self.logger.error(f"Bug detection failed for {bug_type}: {e}")
        
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        if bugs_detected:
            return VerificationResult(
                passed=False,
                confidence=overall_confidence,
                reasoning=f"Detected {len(bugs_detected)} potential bugs",
                bug_detected=True,
                bug_details={"bugs": bugs_detected}
            )
        else:
            return VerificationResult(
                passed=True,
                confidence=overall_confidence,
                reasoning="No bugs detected",
                bug_detected=False
            )
    
    async def _detect_crash(self, subgoal: SubGoal, ui_state: Optional[UIState]) -> Dict[str, Any]:
        """Detect app crashes"""
        if not ui_state:
            return {"detected": True, "confidence": 0.8, "description": "No UI state available - possible crash"}
        
        # Check for crash indicators
        crash_indicators = [
            "unfortunately",
            "has stopped",
            "error",
            "crash",
            "force close"
        ]
        
        for element in ui_state.elements:
            text = element.get_text_content().lower()
            for indicator in crash_indicators:
                if indicator in text:
                    return {
                        "detected": True,
                        "confidence": 0.9,
                        "severity": "critical",
                        "description": f"Crash dialog detected: {text}",
                        "evidence": {"element_text": text, "element_id": element.id}
                    }
        
        return {"detected": False, "confidence": 0.8}
    
    async def _detect_freeze(self, subgoal: SubGoal, ui_state: Optional[UIState]) -> Dict[str, Any]:
        """Detect app freezes"""
        # Check if UI state hasn't changed recently (simplified implementation)
        if len(self.ui_state_history) >= 3:
            recent_hashes = [state.hierarchy_hash for state in self.ui_state_history[-3:]]
            if len(set(recent_hashes)) == 1 and subgoal.action_type in ["touch", "scroll"]:
                return {
                    "detected": True,
                    "confidence": 0.7,
                    "severity": "high",
                    "description": "UI appears frozen - no changes detected after actions",
                    "evidence": {"unchanged_states": 3}
                }
        
        return {"detected": False, "confidence": 0.7}
    
    async def _detect_memory_leak(self, subgoal: SubGoal, ui_state: Optional[UIState]) -> Dict[str, Any]:
        """Detect potential memory leaks"""
        # This would require integration with performance monitoring
        # For now, return no detection
        return {"detected": False, "confidence": 0.5}
    
    async def _detect_ui_inconsistency(self, subgoal: SubGoal, ui_state: Optional[UIState]) -> Dict[str, Any]:
        """Detect UI inconsistencies"""
        if not ui_state:
            return {"detected": False, "confidence": 0.5}
        
        inconsistencies = []
        
        # Check for overlapping elements
        for i, elem1 in enumerate(ui_state.elements):
            if not elem1.bounds:
                continue
            for elem2 in ui_state.elements[i+1:]:
                if not elem2.bounds:
                    continue
                if self._elements_overlap(elem1, elem2):
                    inconsistencies.append(f"Elements overlap: {elem1.id} and {elem2.id}")
        
        # Check for elements outside screen bounds
        for elem in ui_state.elements:
            if elem.bounds:
                left, top, right, bottom = elem.bounds
                if (right > ui_state.screen_width or 
                    bottom > ui_state.screen_height or 
                    left < 0 or top < 0):
                    inconsistencies.append(f"Element outside screen bounds: {elem.id}")
        
        if inconsistencies:
            return {
                "detected": True,
                "confidence": 0.8,
                "severity": "medium",
                "description": f"UI inconsistencies detected: {len(inconsistencies)} issues",
                "evidence": {"inconsistencies": inconsistencies}
            }
        
        return {"detected": False, "confidence": 0.8}
    
    async def _detect_data_corruption(self, subgoal: SubGoal, ui_state: Optional[UIState]) -> Dict[str, Any]:
        """Detect data corruption issues"""
        # This would require more sophisticated data validation
        return {"detected": False, "confidence": 0.5}
    
    async def _detect_accessibility_issues(self, subgoal: SubGoal, ui_state: Optional[UIState]) -> Dict[str, Any]:
        """Detect accessibility issues"""
        if not ui_state:
            return {"detected": False, "confidence": 0.5}
        
        issues = []
        
        # Check for missing content descriptions on important elements
        for elem in ui_state.elements:
            if (elem.element_type.value in ["button", "image_view"] and 
                elem.is_clickable and 
                not elem.content_description and 
                not elem.text):
                issues.append(f"Missing content description: {elem.id}")
        
        # Check for very small touch targets
        for elem in ui_state.elements:
            if elem.is_clickable and elem.bounds:
                width = elem.width or 0
                height = elem.height or 0
                if width < 48 or height < 48:  # Android recommendation: 48dp minimum
                    issues.append(f"Touch target too small: {elem.id} ({width}x{height})")
        
        if issues:
            return {
                "detected": True,
                "confidence": 0.7,
                "severity": "low",
                "description": f"Accessibility issues detected: {len(issues)} issues",
                "evidence": {"issues": issues}
            }
        
        return {"detected": False, "confidence": 0.7}
    
    def _elements_overlap(self, elem1: UIElement, elem2: UIElement) -> bool:
        """Check if two elements overlap"""
        if not elem1.bounds or not elem2.bounds:
            return False
        
        left1, top1, right1, bottom1 = elem1.bounds
        left2, top2, right2, bottom2 = elem2.bounds
        
        return not (right1 <= left2 or right2 <= left1 or bottom1 <= top2 or bottom2 <= top1)
    
    def _update_state_history(self, ui_state: UIState):
        """Update UI state history for trend analysis"""
        self.ui_state_history.append(ui_state)
        if len(self.ui_state_history) > self.max_history_size:
            self.ui_state_history.pop(0)
    
    async def _check_wait_condition(self, condition: str, ui_state: Optional[UIState]) -> bool:
        """Check if a wait condition is met"""
        # Similar implementation to executor's _check_condition
        if "element_present:" in condition:
            element_desc = condition.split("element_present:")[1].strip()
            # Simplified: check if any element contains the description
            if ui_state:
                for elem in ui_state.elements:
                    if element_desc.lower() in elem.get_text_content().lower():
                        return True
        elif "text_present:" in condition:
            text = condition.split("text_present:")[1].strip()
            if ui_state:
                elements = ui_state.find_elements_by_text(text, exact_match=False)
                return len(elements) > 0
        
        return False
    
    def _generate_recommendations(self, verification_result: VerificationResult, 
                                bug_result: VerificationResult) -> List[str]:
        """Generate recommendations based on verification results"""
        recommendations = []
        
        if not verification_result.passed:
            recommendations.append("Review subgoal expectations and modify if necessary")
            recommendations.append("Check element selectors and targeting strategy")
        
        if bug_result.bug_detected:
            for bug in bug_result.bug_details.get("bugs", []):
                if bug["type"] == "crash":
                    recommendations.append("Investigate crash cause and add error handling")
                elif bug["type"] == "freeze":
                    recommendations.append("Add longer wait times or alternative navigation")
                elif bug["type"] == "ui_inconsistency":
                    recommendations.append("Review UI layout and element positioning")
                elif bug["type"] == "accessibility":
                    recommendations.append("Improve accessibility features for better usability")
        
        if verification_result.confidence < 0.7:
            recommendations.append("Gather more verification evidence for better confidence")
        
        return recommendations
    
    def _dict_to_subgoal(self, subgoal_dict: Dict[str, Any]) -> SubGoal:
        """Convert dictionary to SubGoal object"""
        return SubGoal(
            id=subgoal_dict.get("id"),
            description=subgoal_dict.get("description", ""),
            action_type=subgoal_dict.get("action_type", "touch"),
            target_element=subgoal_dict.get("target_element"),
            parameters=subgoal_dict.get("parameters", {}),
            expected_result=subgoal_dict.get("expected_result"),
            execution_order=subgoal_dict.get("execution_order", 0)
        )
    
    async def _validate_agent_specific(self, decision: AgentDecision) -> bool:
        """Validate verifier-specific decisions"""
        if decision.decision_type == DecisionType.VERIFICATION:
            output_data = decision.output_data
            
            # Check if verification result is present
            if "verification_passed" not in output_data:
                self.logger.warning("Verification decision missing verification_passed field")
                return False
            
            # Check if bug detection info is present
            if "bug_detected" not in output_data:
                self.logger.warning("Verification decision missing bug_detected field")
                return False
        
        return True 