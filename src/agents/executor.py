import json
import time
from typing import Dict, Any, List, Optional, Tuple
import logging

from .base import BaseAgent, AgentConfig
from ..models.result import AgentDecision, AgentType, DecisionType, ExecutionResult, ResultStatus
from ..models.task import SubGoal, TaskStatus
from ..models.ui_state import UIState, UIElement
from ..core.llm_client import LLMClient, LLMMessage
from ..core.android_integration import AndroidWorldIntegration, AndroidAction

class ExecutorAgent(BaseAgent):
    """Agent responsible for executing subgoals in the Android UI environment"""
    
    def __init__(self, 
                 config: AgentConfig, 
                 android_integration: AndroidWorldIntegration,
                 logger: Optional[logging.Logger] = None):
        self.android_integration = android_integration
        super().__init__(AgentType.EXECUTOR, config, logger)
        
    def _initialize_agent(self):
        """Initialize executor-specific components"""
        self.llm_client = LLMClient(
            provider=self.config.provider,
            model=self.config.model_name,
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )
        
        # Execution strategies for different action types
        self.action_executors = {
            "touch": self._execute_touch_action,
            "type": self._execute_type_action,
            "scroll": self._execute_scroll_action,
            "wait": self._execute_wait_action,
            "verify": self._execute_verify_action,
            "key": self._execute_key_action
        }
        
        # Element locating strategies
        self.element_locators = {
            "resource_id": self._find_by_resource_id,
            "text": self._find_by_text,
            "content_description": self._find_by_content_description,
            "xpath": self._find_by_xpath,
            "coordinates": self._find_by_coordinates,
            "visual": self._find_by_visual_description
        }
        
        # Execution context
        self.current_execution: Optional[ExecutionResult] = None
        self.element_cache: Dict[str, UIElement] = {}
        
    async def make_decision(self, context: Dict[str, Any], ui_state: Optional[UIState] = None) -> AgentDecision:
        """Make an execution decision based on the subgoal and UI state"""
        subgoal_data = context.get("subgoal")
        if not subgoal_data:
            return self.create_decision(
                decision_type=DecisionType.ACTION_EXECUTION,
                reasoning="No subgoal provided for execution",
                confidence=0.0,
                output_data={"error": "missing_subgoal"}
            )
        
        # Convert dict to SubGoal if needed
        if isinstance(subgoal_data, dict):
            subgoal = self._dict_to_subgoal(subgoal_data)
        else:
            subgoal = subgoal_data
        
        # Execute the subgoal
        execution_result = await self._execute_subgoal(subgoal, ui_state)
        
        # Determine decision confidence based on execution result
        confidence = self._calculate_execution_confidence(execution_result, subgoal)
        
        # Create decision based on execution result
        if execution_result.status == ResultStatus.SUCCESS:
            reasoning = f"Successfully executed {subgoal.action_type}: {subgoal.description}"
            decision_type = DecisionType.ACTION_EXECUTION
        else:
            reasoning = f"Failed to execute {subgoal.action_type}: {execution_result.error_message}"
            decision_type = DecisionType.ERROR_HANDLING
        
        return self.create_decision(
            decision_type=decision_type,
            reasoning=reasoning,
            confidence=confidence,
            output_data={
                "execution_result": execution_result.to_log_dict(),
                "subgoal_id": subgoal.id,
                "action_type": subgoal.action_type,
                "success": execution_result.status == ResultStatus.SUCCESS
            },
            input_data={
                "subgoal_description": subgoal.description,
                "target_element": subgoal.target_element,
                "parameters": subgoal.parameters
            }
        )
    
    async def _execute_subgoal(self, subgoal: SubGoal, ui_state: Optional[UIState]) -> ExecutionResult:
        """Execute a specific subgoal"""
        self.logger.info(f"Executing subgoal: {subgoal.description}")
        
        execution_result = ExecutionResult(
            subgoal_id=subgoal.id,
            action_type=subgoal.action_type,
            expected_outcome=subgoal.expected_result
        )
        
        try:
            # Get the appropriate executor for this action type
            executor = self.action_executors.get(subgoal.action_type)
            if not executor:
                raise ValueError(f"Unsupported action type: {subgoal.action_type}")
            
            # Update UI state if needed
            if not ui_state:
                await self.android_integration.update_ui_state()
                ui_state = self.android_integration.get_current_ui_state()
            
            # Execute the action
            android_action = await executor(subgoal, ui_state, execution_result)
            
            if android_action:
                # Execute the action in Android environment
                android_result = await self.android_integration.execute_action(android_action)
                
                # Update execution result with Android result data
                execution_result.actual_coordinates = android_result.actual_coordinates
                execution_result.text_entered = android_result.text_entered
                execution_result.scroll_distance = android_result.scroll_distance
                execution_result.screenshot_before = android_result.screenshot_before
                execution_result.screenshot_after = android_result.screenshot_after
                execution_result.execution_time_ms = android_result.execution_time_ms
                
                if android_result.status == ResultStatus.SUCCESS:
                    execution_result.mark_success(android_result.actual_outcome or "Action completed")
                else:
                    execution_result.mark_failure(
                        android_result.error_message or "Android action failed",
                        android_result.error_type or "android_error"
                    )
            else:
                # For verify actions or other non-Android actions
                if subgoal.action_type == "verify":
                    execution_result.mark_success("Verification completed")
                else:
                    execution_result.mark_success("Action completed without Android interaction")
        
        except Exception as e:
            self.logger.error(f"Error executing subgoal {subgoal.id}: {e}")
            execution_result.mark_failure(str(e), "execution_error")
        
        return execution_result
    
    async def _execute_touch_action(self, subgoal: SubGoal, ui_state: UIState, result: ExecutionResult) -> Optional[AndroidAction]:
        """Execute a touch action"""
        # Find the target element
        target_element = await self._locate_target_element(subgoal, ui_state)
        
        if not target_element:
            raise ValueError(f"Could not locate target element: {subgoal.target_element}")
        
        if not target_element.is_clickable:
            self.logger.warning(f"Element {target_element.id} is not clickable, attempting anyway")
        
        # Get coordinates
        coordinates = target_element.get_center_coordinates()
        
        # Create Android action
        return AndroidAction(
            action_type="touch",
            coordinates=coordinates,
            element_id=target_element.id
        )
    
    async def _execute_type_action(self, subgoal: SubGoal, ui_state: UIState, result: ExecutionResult) -> Optional[AndroidAction]:
        """Execute a type action"""
        # Get text to type
        text = subgoal.parameters.get("text", "")
        if not text:
            raise ValueError("Type action requires text parameter")
        
        # Find the target element (optional for type actions)
        if subgoal.target_element:
            target_element = await self._locate_target_element(subgoal, ui_state)
            if target_element and target_element.is_editable:
                # Touch the element first to focus it
                touch_action = AndroidAction(
                    action_type="touch",
                    coordinates=target_element.get_center_coordinates(),
                    element_id=target_element.id
                )
                await self.android_integration.execute_action(touch_action)
                
                # Wait a bit for focus
                await self.android_integration.execute_action(AndroidAction(
                    action_type="wait",
                    duration_ms=500
                ))
        
        # Create type action
        return AndroidAction(
            action_type="type",
            text=text
        )
    
    async def _execute_scroll_action(self, subgoal: SubGoal, ui_state: UIState, result: ExecutionResult) -> Optional[AndroidAction]:
        """Execute a scroll action"""
        direction = subgoal.parameters.get("direction", "down")
        distance = subgoal.parameters.get("distance", 300)
        
        # Find scroll area or use screen center
        if subgoal.target_element:
            target_element = await self._locate_target_element(subgoal, ui_state)
            if target_element:
                coordinates = target_element.get_center_coordinates()
            else:
                raise ValueError(f"Could not locate scroll target: {subgoal.target_element}")
        else:
            # Use screen center
            coordinates = (ui_state.screen_width // 2, ui_state.screen_height // 2)
        
        return AndroidAction(
            action_type="scroll",
            coordinates=coordinates,
            direction=direction,
            distance=distance
        )
    
    async def _execute_wait_action(self, subgoal: SubGoal, ui_state: UIState, result: ExecutionResult) -> Optional[AndroidAction]:
        """Execute a wait action"""
        duration = subgoal.parameters.get("duration_ms", 1000)
        condition = subgoal.parameters.get("condition")
        
        if condition:
            # Wait for a specific condition
            await self._wait_for_condition(condition, ui_state)
            return None
        else:
            # Simple time-based wait
            return AndroidAction(
                action_type="wait",
                duration_ms=duration
            )
    
    async def _execute_verify_action(self, subgoal: SubGoal, ui_state: UIState, result: ExecutionResult) -> Optional[AndroidAction]:
        """Execute a verification action"""
        verification_type = subgoal.parameters.get("type", "element_present")
        
        if verification_type == "element_present":
            target_element = await self._locate_target_element(subgoal, ui_state)
            if target_element:
                result.actual_outcome = f"Element found: {target_element.get_text_content()}"
                result.verification_passed = True
            else:
                result.actual_outcome = "Element not found"
                result.verification_passed = False
                
        elif verification_type == "text_present":
            expected_text = subgoal.parameters.get("text", "")
            elements_with_text = ui_state.find_elements_by_text(expected_text, exact_match=False)
            if elements_with_text:
                result.actual_outcome = f"Text found in {len(elements_with_text)} elements"
                result.verification_passed = True
            else:
                result.actual_outcome = "Text not found"
                result.verification_passed = False
                
        elif verification_type == "app_state":
            expected_app = subgoal.parameters.get("app")
            expected_activity = subgoal.parameters.get("activity")
            
            app_matches = not expected_app or ui_state.current_app == expected_app
            activity_matches = not expected_activity or ui_state.current_activity == expected_activity
            
            if app_matches and activity_matches:
                result.actual_outcome = f"App state correct: {ui_state.current_app}/{ui_state.current_activity}"
                result.verification_passed = True
            else:
                result.actual_outcome = f"App state incorrect: {ui_state.current_app}/{ui_state.current_activity}"
                result.verification_passed = False
        
        # Verification doesn't need Android action
        return None
    
    async def _execute_key_action(self, subgoal: SubGoal, ui_state: UIState, result: ExecutionResult) -> Optional[AndroidAction]:
        """Execute a key press action"""
        key_code = subgoal.parameters.get("key_code")
        key_name = subgoal.parameters.get("key_name")
        
        # Map common key names to codes
        key_mapping = {
            "back": 4,
            "home": 3,
            "menu": 82,
            "enter": 66,
            "delete": 67,
            "tab": 61
        }
        
        if key_name and key_name.lower() in key_mapping:
            key_code = key_mapping[key_name.lower()]
        
        if not key_code:
            raise ValueError("Key action requires key_code or key_name parameter")
        
        return AndroidAction(
            action_type="key",
            key_code=key_code
        )
    
    async def _locate_target_element(self, subgoal: SubGoal, ui_state: UIState) -> Optional[UIElement]:
        """Locate the target element for a subgoal"""
        if not subgoal.target_element:
            return None
        
        # Check cache first
        cache_key = f"{ui_state.hierarchy_hash}_{subgoal.target_element}"
        if cache_key in self.element_cache:
            return self.element_cache[cache_key]
        
        # Try different locator strategies
        target_spec = subgoal.target_element
        
        # If target_spec is a JSON string, parse it
        if isinstance(target_spec, str) and target_spec.startswith("{"):
            try:
                target_spec = json.loads(target_spec)
            except json.JSONDecodeError:
                pass
        
        element = None
        
        if isinstance(target_spec, dict):
            # Use multiple locator strategies
            for strategy, locator_func in self.element_locators.items():
                if strategy in target_spec:
                    element = await locator_func(target_spec[strategy], ui_state)
                    if element:
                        break
        else:
            # Try as text first, then resource_id
            element = await self._find_by_text(target_spec, ui_state)
            if not element:
                element = await self._find_by_resource_id(target_spec, ui_state)
            if not element:
                element = await self._find_by_visual_description(target_spec, ui_state)
        
        # Cache the result
        if element:
            self.element_cache[cache_key] = element
        
        return element
    
    async def _find_by_resource_id(self, resource_id: str, ui_state: UIState) -> Optional[UIElement]:
        """Find element by resource ID"""
        return ui_state.find_element_by_resource_id(resource_id)
    
    async def _find_by_text(self, text: str, ui_state: UIState) -> Optional[UIElement]:
        """Find element by text content"""
        elements = ui_state.find_elements_by_text(text, exact_match=False)
        return elements[0] if elements else None
    
    async def _find_by_content_description(self, content_desc: str, ui_state: UIState) -> Optional[UIElement]:
        """Find element by content description"""
        elements = ui_state.find_elements(content_description=content_desc)
        return elements[0] if elements else None
    
    async def _find_by_xpath(self, xpath: str, ui_state: UIState) -> Optional[UIElement]:
        """Find element by XPath (simplified implementation)"""
        # This would require a more sophisticated XPath parser
        # For now, return None
        return None
    
    async def _find_by_coordinates(self, coordinates: Tuple[int, int], ui_state: UIState) -> Optional[UIElement]:
        """Find element at specific coordinates"""
        x, y = coordinates
        return ui_state.get_element_at_coordinates(x, y)
    
    async def _find_by_visual_description(self, description: str, ui_state: UIState) -> Optional[UIElement]:
        """Find element by visual description using LLM"""
        try:
            # Get current UI summary
            ui_summary = ui_state.get_interactable_elements_summary()
            
            # Use LLM to identify the element
            system_prompt = """
            You are an expert at identifying UI elements from descriptions.
            Given a description and a list of available elements, identify the most likely target element.
            """
            
            user_prompt = f"""
            Find the UI element that best matches this description: "{description}"
            
            Available elements:
            {json.dumps(ui_summary.get('clickable_elements', []), indent=2)}
            
            Return the element ID of the best match, or null if no good match exists.
            """
            
            messages = self.llm_client.create_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            response = await self.llm_client.generate_response(
                messages=messages,
                temperature=0.1,
                max_tokens=100
            )
            
            # Extract element ID from response
            content = response.content.strip()
            if content and content != "null":
                return ui_state.find_element_by_id(content)
            
        except Exception as e:
            self.logger.error(f"Visual element finding failed: {e}")
        
        return None
    
    async def _wait_for_condition(self, condition: str, ui_state: UIState, timeout_seconds: int = 10):
        """Wait for a specific condition to be met"""
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            # Update UI state
            await self.android_integration.update_ui_state()
            current_ui_state = self.android_integration.get_current_ui_state()
            
            # Check condition
            if await self._check_condition(condition, current_ui_state):
                return True
            
            await asyncio.sleep(0.5)
        
        raise TimeoutError(f"Condition not met within {timeout_seconds} seconds: {condition}")
    
    async def _check_condition(self, condition: str, ui_state: UIState) -> bool:
        """Check if a specific condition is met"""
        # Parse condition (simplified implementation)
        if "element_present:" in condition:
            element_desc = condition.split("element_present:")[1].strip()
            element = await self._locate_target_element_by_description(element_desc, ui_state)
            return element is not None
        elif "text_present:" in condition:
            text = condition.split("text_present:")[1].strip()
            elements = ui_state.find_elements_by_text(text, exact_match=False)
            return len(elements) > 0
        elif "app_changed" in condition:
            # Check if app has changed (would need previous state)
            return True  # Simplified
        
        return False
    
    async def _locate_target_element_by_description(self, description: str, ui_state: UIState) -> Optional[UIElement]:
        """Helper method to locate element by description"""
        return await self._find_by_visual_description(description, ui_state)
    
    def _calculate_execution_confidence(self, execution_result: ExecutionResult, subgoal: SubGoal) -> float:
        """Calculate confidence score for execution result"""
        if execution_result.status == ResultStatus.SUCCESS:
            base_confidence = 0.9
            
            # Adjust based on verification if applicable
            if subgoal.action_type == "verify":
                if execution_result.verification_passed:
                    return base_confidence
                else:
                    return 0.3  # Low confidence for failed verification
            
            # Adjust based on execution time
            if execution_result.execution_time_ms > 10000:  # Very slow execution
                base_confidence -= 0.1
            
            return base_confidence
        else:
            # Failed execution
            return 0.1
    
    def _dict_to_subgoal(self, subgoal_dict: Dict[str, Any]) -> SubGoal:
        """Convert dictionary to SubGoal object"""
        return SubGoal(
            id=subgoal_dict.get("id"),
            description=subgoal_dict.get("description", ""),
            action_type=subgoal_dict.get("action_type", "touch"),
            target_element=subgoal_dict.get("target_element"),
            parameters=subgoal_dict.get("parameters", {}),
            expected_result=subgoal_dict.get("expected_result"),
            execution_order=subgoal_dict.get("execution_order", 0),
            max_retries=subgoal_dict.get("max_retries", 3)
        )
    
    async def _validate_agent_specific(self, decision: AgentDecision) -> bool:
        """Validate executor-specific decisions"""
        if decision.decision_type == DecisionType.ACTION_EXECUTION:
            output_data = decision.output_data
            execution_result = output_data.get("execution_result")
            
            if not execution_result:
                self.logger.warning("Execution decision missing execution result")
                return False
            
            # Check if execution result has required fields
            required_fields = ["status", "action_type", "execution_time_ms"]
            for field in required_fields:
                if field not in execution_result:
                    self.logger.warning(f"Execution result missing field: {field}")
                    return False
        
        return True
    
    def clear_element_cache(self):
        """Clear the element cache"""
        self.element_cache.clear()
        self.logger.debug("Element cache cleared") 