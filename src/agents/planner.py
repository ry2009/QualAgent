import json
from typing import Dict, Any, List, Optional
import logging

from .base import BaseAgent, AgentConfig
from ..models.result import AgentDecision, AgentType, DecisionType
from ..models.task import QATask, TestGoal, SubGoal, TaskStatus
from ..models.ui_state import UIState
from ..core.llm_client import LLMClient, LLMMessage

class PlannerAgent(BaseAgent):
    """Agent responsible for planning and decomposing QA goals into executable subgoals"""
    
    def __init__(self, config: AgentConfig, logger: Optional[logging.Logger] = None):
        super().__init__(AgentType.PLANNER, config, logger)
        
    def _initialize_agent(self):
        """Initialize planner-specific components"""
        self.llm_client = LLMClient(
            provider=self.config.provider,
            model=self.config.model_name,
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )
        
        # Planning capabilities (simplified for now)
        self.planning_strategies = {}
        
        # Knowledge base for common app patterns
        self.app_patterns = {
            "settings": ["navigate_to_settings", "find_target_setting", "modify_setting", "verify_change"],
            "login": ["enter_credentials", "submit_form", "verify_login"],
            "search": ["access_search", "enter_query", "execute_search", "verify_results"],
            "navigation": ["identify_nav_element", "tap_element", "verify_navigation"],
            "form_filling": ["identify_fields", "fill_fields", "submit_form", "verify_submission"]
        }
    
    async def make_decision(self, context: Dict[str, Any], ui_state: Optional[UIState] = None) -> AgentDecision:
        """Make a planning decision based on the context"""
        request_type = context.get("request_type", "create_plan")
        
        if request_type == "create_plan":
            return await self._create_initial_plan(context, ui_state)
        elif request_type == "modify_plan":
            return await self._modify_existing_plan(context, ui_state)
        elif request_type == "replan":
            return await self._replan_after_failure(context, ui_state)
        else:
            raise ValueError(f"Unknown planning request type: {request_type}")
    
    async def _create_initial_plan(self, context: Dict[str, Any], ui_state: Optional[UIState]) -> AgentDecision:
        """Create an initial plan for a test goal"""
        goal_description = context.get("goal_description", "")
        app_name = context.get("app_name", "")
        test_type = context.get("test_type", "functional")
        
        if not goal_description:
            return self.create_decision(
                decision_type=DecisionType.PLAN_CREATION,
                reasoning="Missing goal description for planning",
                confidence=0.0,
                output_data={"error": "goal_description required"}
            )
        
        # Use LLM to analyze the goal and create plan
        plan_data = await self._generate_plan_with_llm(
            goal_description, app_name, test_type, ui_state
        )
        
        if not plan_data:
            return self.create_decision(
                decision_type=DecisionType.PLAN_CREATION,
                reasoning="Failed to generate plan with LLM",
                confidence=0.0,
                output_data={"error": "plan_generation_failed"}
            )
        
        # Create subgoals from plan
        subgoals = self._create_subgoals_from_plan(plan_data)
        
        confidence = self._calculate_plan_confidence(plan_data, subgoals)
        
        return self.create_decision(
            decision_type=DecisionType.PLAN_CREATION,
            reasoning=f"Created plan with {len(subgoals)} subgoals for {test_type} test",
            confidence=confidence,
            output_data={
                "subgoals": [self._subgoal_to_dict(sg) for sg in subgoals],
                "plan_metadata": plan_data.get("metadata", {}),
                "estimated_duration": plan_data.get("estimated_duration", 60),
                "risk_factors": plan_data.get("risk_factors", [])
            },
            input_data={
                "goal_description": goal_description,
                "app_name": app_name,
                "test_type": test_type
            }
        )
    
    async def _modify_existing_plan(self, context: Dict[str, Any], ui_state: Optional[UIState]) -> AgentDecision:
        """Modify an existing plan based on new information"""
        current_subgoals = context.get("current_subgoals", [])
        modification_reason = context.get("reason", "")
        failed_subgoal_id = context.get("failed_subgoal_id")
        
        # Analyze the current situation
        analysis = await self._analyze_plan_modification_need(
            current_subgoals, modification_reason, ui_state
        )
        
        if not analysis.get("needs_modification", False):
            return self.create_decision(
                decision_type=DecisionType.PLAN_MODIFICATION,
                reasoning="No plan modification needed",
                confidence=0.9,
                output_data={"action": "no_change", "analysis": analysis}
            )
        
        # Generate modified plan
        modified_plan = await self._generate_plan_modification(
            current_subgoals, analysis, ui_state
        )
        
        confidence = analysis.get("confidence", 0.7)
        
        return self.create_decision(
            decision_type=DecisionType.PLAN_MODIFICATION,
            reasoning=f"Modified plan: {analysis.get('reasoning', 'unknown reason')}",
            confidence=confidence,
            output_data={
                "modified_subgoals": modified_plan.get("subgoals", []),
                "changes_made": modified_plan.get("changes", []),
                "modification_type": analysis.get("modification_type", "unknown")
            },
            input_data={
                "original_subgoals_count": len(current_subgoals),
                "reason": modification_reason,
                "failed_subgoal_id": failed_subgoal_id
            }
        )
    
    async def _replan_after_failure(self, context: Dict[str, Any], ui_state: Optional[UIState]) -> AgentDecision:
        """Replan after a failure has occurred"""
        failed_subgoal = context.get("failed_subgoal")
        error_details = context.get("error_details", {})
        remaining_subgoals = context.get("remaining_subgoals", [])
        
        # Analyze the failure
        failure_analysis = await self._analyze_failure(
            failed_subgoal, error_details, ui_state
        )
        
        # Determine recovery strategy
        recovery_strategy = await self._determine_recovery_strategy(
            failure_analysis, remaining_subgoals, ui_state
        )
        
        if recovery_strategy.get("strategy") == "abort":
            return self.create_decision(
                decision_type=DecisionType.RECOVERY,
                reasoning="Failure is unrecoverable, aborting test",
                confidence=0.9,
                output_data={
                    "action": "abort",
                    "failure_analysis": failure_analysis,
                    "abort_reason": recovery_strategy.get("reason")
                }
            )
        
        # Generate recovery plan
        recovery_plan = await self._generate_recovery_plan(
            recovery_strategy, failed_subgoal, remaining_subgoals, ui_state
        )
        
        confidence = recovery_strategy.get("confidence", 0.6)
        
        return self.create_decision(
            decision_type=DecisionType.RECOVERY,
            reasoning=f"Generated recovery plan: {recovery_strategy.get('strategy')}",
            confidence=confidence,
            output_data={
                "recovery_subgoals": recovery_plan.get("subgoals", []),
                "strategy": recovery_strategy.get("strategy"),
                "skip_to_subgoal": recovery_plan.get("skip_to_subgoal"),
                "retry_with_modifications": recovery_plan.get("retry_with_modifications", False)
            },
            input_data={
                "failed_subgoal_id": failed_subgoal.get("id") if failed_subgoal else None,
                "error_type": error_details.get("error_type"),
                "remaining_count": len(remaining_subgoals)
            }
        )
    
    async def _generate_plan_with_llm(self, goal_description: str, app_name: str, 
                                    test_type: str, ui_state: Optional[UIState]) -> Dict[str, Any]:
        """Use LLM to generate a plan for the given goal"""
        
        # Build context for the LLM
        context_info = []
        if ui_state:
            context_info.append(f"Current app: {ui_state.current_app}")
            context_info.append(f"Current activity: {ui_state.current_activity}")
            
            # Add interactive elements info
            clickable_elements = ui_state.find_clickable_elements()
            if clickable_elements:
                element_descriptions = []
                for elem in clickable_elements[:5]:  # Limit to first 5
                    text = elem.get_text_content()
                    element_descriptions.append(f"- {elem.element_type.value}: {text[:30]}")
                context_info.append(f"Available interactive elements:\n" + "\n".join(element_descriptions))
        
        context_str = "\n".join(context_info) if context_info else "No current UI context available"
        
        # Create system prompt
        system_prompt = self._get_planning_system_prompt(test_type)
        
        # Create user prompt
        user_prompt = f"""
        Plan a {test_type} test for the following goal:
        
        Goal: {goal_description}
        App: {app_name}
        
        Current UI Context:
        {context_str}
        
        Please provide a detailed plan with actionable subgoals.
        """
        
        # Define response schema
        response_schema = {
            "type": "object",
            "properties": {
                "subgoals": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "action_type": {"type": "string"},
                            "target_element": {"type": "string"},
                            "parameters": {"type": "object"},
                            "expected_result": {"type": "string"},
                            "retry_strategies": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["description", "action_type", "expected_result"]
                    }
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "complexity": {"type": "string"},
                        "estimated_duration": {"type": "integer"},
                        "dependencies": {"type": "array", "items": {"type": "string"}},
                        "risk_factors": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "required": ["subgoals", "metadata"]
        }
        
        try:
            messages = self.llm_client.create_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            response = await self.llm_client.generate_structured_response(
                messages=messages,
                response_schema=response_schema,
                temperature=self.config.temperature
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"LLM plan generation failed: {e}")
            return {}
    
    def _get_planning_system_prompt(self, test_type: str) -> str:
        """Get system prompt for planning based on test type"""
        
        base_prompt = """
        You are an expert QA planning agent for mobile app testing. Your job is to create detailed, 
        actionable test plans that can be executed by automation agents.
        
        Key principles:
        1. Break down complex goals into simple, atomic actions
        2. Each subgoal should have a clear action type (touch, type, scroll, wait, verify)
        3. Be specific about target elements and expected results
        4. Consider error scenarios and provide retry strategies
        5. Estimate realistic timeframes and identify risk factors
        
        Available action types:
        - touch: Tap on an element
        - type: Enter text in a field
        - scroll: Scroll in a direction
        - wait: Wait for a condition or time
        - verify: Check if something is as expected
        """
        
        type_specific_prompts = {
            "functional": """
            Focus on core app functionality. Test the main user flows and ensure features work as intended.
            Prioritize happy path scenarios with basic error handling.
            """,
            "ui": """
            Focus on user interface elements, layouts, and visual feedback.
            Test element interactions, state changes, and visual consistency.
            """,
            "integration": """
            Focus on how different parts of the app work together.
            Test data flow between screens, external integrations, and cross-feature interactions.
            """,
            "regression": """
            Focus on ensuring existing functionality still works.
            Test critical paths and previously identified problem areas.
            """
        }
        
        return base_prompt + "\n" + type_specific_prompts.get(test_type, "")
    
    def _create_subgoals_from_plan(self, plan_data: Dict[str, Any]) -> List[SubGoal]:
        """Create SubGoal objects from LLM plan data"""
        subgoals = []
        
        for i, subgoal_data in enumerate(plan_data.get("subgoals", [])):
            subgoal = SubGoal(
                description=subgoal_data.get("description", ""),
                action_type=subgoal_data.get("action_type", "touch"),
                target_element=subgoal_data.get("target_element"),
                parameters=subgoal_data.get("parameters", {}),
                expected_result=subgoal_data.get("expected_result"),
                execution_order=i
            )
            
            # Add retry strategies if provided
            retry_strategies = subgoal_data.get("retry_strategies", [])
            if retry_strategies:
                subgoal.parameters["retry_strategies"] = retry_strategies
            
            subgoals.append(subgoal)
        
        return subgoals
    
    def _calculate_plan_confidence(self, plan_data: Dict[str, Any], subgoals: List[SubGoal]) -> float:
        """Calculate confidence score for the generated plan"""
        confidence = 0.8  # Base confidence
        
        # Adjust based on plan completeness
        if not subgoals:
            return 0.0
        
        # Check if subgoals have required fields
        complete_subgoals = sum(1 for sg in subgoals if sg.description and sg.expected_result)
        completeness_ratio = complete_subgoals / len(subgoals)
        confidence *= completeness_ratio
        
        # Adjust based on metadata quality
        metadata = plan_data.get("metadata", {})
        if metadata.get("complexity") and metadata.get("estimated_duration"):
            confidence += 0.1
        
        # Adjust based on risk factors
        risk_factors = metadata.get("risk_factors", [])
        if len(risk_factors) > 3:
            confidence -= 0.1
        
        return min(max(confidence, 0.0), 1.0)
    
    async def _analyze_plan_modification_need(self, current_subgoals: List[Dict], 
                                            reason: str, ui_state: Optional[UIState]) -> Dict[str, Any]:
        """Analyze if plan modification is needed"""
        # Implement analysis logic
        return {
            "needs_modification": True,
            "confidence": 0.8,
            "reasoning": f"Plan modification needed due to: {reason}",
            "modification_type": "adaptive"
        }
    
    async def _generate_plan_modification(self, current_subgoals: List[Dict], 
                                        analysis: Dict[str, Any], ui_state: Optional[UIState]) -> Dict[str, Any]:
        """Generate modifications to the current plan"""
        # Implement modification logic
        return {
            "subgoals": current_subgoals,  # Placeholder
            "changes": ["example_change"]
        }
    
    async def _analyze_failure(self, failed_subgoal: Dict, error_details: Dict, 
                             ui_state: Optional[UIState]) -> Dict[str, Any]:
        """Analyze a failure to determine recovery strategy"""
        return {
            "error_type": error_details.get("error_type", "unknown"),
            "is_recoverable": True,
            "suggested_strategy": "retry_with_modification"
        }
    
    async def _determine_recovery_strategy(self, failure_analysis: Dict, 
                                         remaining_subgoals: List, ui_state: Optional[UIState]) -> Dict[str, Any]:
        """Determine the best recovery strategy"""
        if failure_analysis.get("is_recoverable", False):
            return {
                "strategy": "retry_with_modification",
                "confidence": 0.7,
                "reason": "Failure appears recoverable"
            }
        else:
            return {
                "strategy": "abort",
                "confidence": 0.9,
                "reason": "Failure is not recoverable"
            }
    
    async def _generate_recovery_plan(self, recovery_strategy: Dict, failed_subgoal: Dict,
                                    remaining_subgoals: List, ui_state: Optional[UIState]) -> Dict[str, Any]:
        """Generate a recovery plan"""
        return {
            "subgoals": [],  # Placeholder
            "retry_with_modifications": True
        }
    
    def _subgoal_to_dict(self, subgoal: SubGoal) -> Dict[str, Any]:
        """Convert SubGoal to dictionary"""
        return {
            "id": subgoal.id,
            "description": subgoal.description,
            "action_type": subgoal.action_type,
            "target_element": subgoal.target_element,
            "parameters": subgoal.parameters,
            "expected_result": subgoal.expected_result,
            "execution_order": subgoal.execution_order,
            "max_retries": subgoal.max_retries
        }
    
    async def _validate_agent_specific(self, decision: AgentDecision) -> bool:
        """Validate planner-specific decisions"""
        if decision.decision_type == DecisionType.PLAN_CREATION:
            output_data = decision.output_data
            subgoals = output_data.get("subgoals", [])
            
            if not subgoals:
                self.logger.warning("Plan creation decision has no subgoals")
                return False
            
            # Validate each subgoal has required fields
            for subgoal in subgoals:
                if not subgoal.get("description") or not subgoal.get("action_type"):
                    self.logger.warning("Subgoal missing required fields")
                    return False
        
        return True 