import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from ..agents.base import AgentConfig
from ..agents.planner import PlannerAgent
from ..agents.executor import ExecutorAgent
from ..agents.verifier import VerifierAgent
from ..agents.supervisor import SupervisorAgent
from ..models.task import QATask, TestGoal, SubGoal, TaskStatus
from ..models.result import TestResult, AgentDecision, ExecutionResult, ResultStatus
from ..models.ui_state import UIState
from ..core.android_integration import AndroidWorldIntegration
from ..core.logging import QALogger

class CoordinationState:
    """Tracks the current state of multi-agent coordination"""
    def __init__(self):
        self.current_task: Optional[QATask] = None
        self.current_goal: Optional[TestGoal] = None
        self.current_subgoal: Optional[SubGoal] = None
        self.current_test_result: Optional[TestResult] = None
        self.active_agents: List[str] = []
        self.coordination_phase: str = "idle"  # idle, planning, executing, verifying, supervising
        self.error_count: int = 0
        self.max_errors: int = 5

class MultiAgentCoordinator:
    """Coordinates multiple agents to execute QA tasks"""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 android_integration: AndroidWorldIntegration,
                 logger: Optional[logging.Logger] = None):
        
        self.config = config
        self.android_integration = android_integration
        self.logger = logger or logging.getLogger("coordinator")
        self.qa_logger = QALogger()
        
        # Coordination state
        self.state = CoordinationState()
        
        # Initialize agents
        self._initialize_agents()
        
        # Coordination settings
        self.max_retries = config.get("max_retries", 3)
        self.retry_delays = [1, 2, 5]  # seconds
        self.enable_replanning = config.get("enable_replanning", True)
        self.enable_supervision = config.get("enable_supervision", True)
        
    def _initialize_agents(self):
        """Initialize all agents with their configurations"""
        # Create agent configurations
        agent_configs = {}
        for agent_type in ["planner", "executor", "verifier", "supervisor"]:
            # Get config from nested agents structure if available
            agent_config_data = self.config.get("agents", {}).get(agent_type, {})
            
            agent_config = AgentConfig(
                model_name=agent_config_data.get("model", self.config.get(f"{agent_type}_model", "gpt-4")),
                provider=agent_config_data.get("provider", self.config.get(f"{agent_type}_provider", "openai")),
                api_key=agent_config_data.get("api_key", self.config.get(f"{agent_type}_api_key")),
                base_url=agent_config_data.get("base_url", self.config.get(f"{agent_type}_base_url")),
                max_retries=agent_config_data.get("max_retries", self.config.get(f"{agent_type}_max_retries", 3)),
                timeout_seconds=agent_config_data.get("timeout", self.config.get(f"{agent_type}_timeout", 30)),
                temperature=agent_config_data.get("temperature", self.config.get(f"{agent_type}_temperature", 0.1)),
                max_tokens=agent_config_data.get("max_tokens", self.config.get(f"{agent_type}_max_tokens", 2000)),
                confidence_threshold=agent_config_data.get("confidence_threshold", self.config.get(f"{agent_type}_confidence_threshold", 0.7))
            )
            agent_configs[agent_type] = agent_config
        
        # Initialize agents
        self.planner = PlannerAgent(agent_configs["planner"], self.logger)
        self.executor = ExecutorAgent(agent_configs["executor"], self.android_integration, self.logger)
        self.verifier = VerifierAgent(agent_configs["verifier"], self.logger)
        self.supervisor = SupervisorAgent(agent_configs["supervisor"], self.logger)
        
        self.agents = {
            "planner": self.planner,
            "executor": self.executor,
            "verifier": self.verifier,
            "supervisor": self.supervisor
        }
        
        self.logger.info("All agents initialized successfully")
    
    async def execute_qa_task(self, qa_task: QATask) -> TestResult:
        """Execute a complete QA task with multiple test goals"""
        self.logger.info(f"Starting QA task execution: {qa_task.name}")
        
        # Initialize task state
        self.state.current_task = qa_task
        qa_task.mark_started()
        
        # Connect to Android environment
        if not await self.android_integration.connect():
            raise RuntimeError("Failed to connect to Android environment")
        
        try:
            # Execute each test goal
            for goal in qa_task.test_goals:
                self.logger.info(f"Executing test goal: {goal.title}")
                
                goal_result = await self.execute_test_goal(goal)
                
                # Log goal completion
                self.qa_logger.log_goal_completion(goal, goal_result)
                
                # Check if we should continue with remaining goals
                if goal_result.overall_status == ResultStatus.FAILURE and not self.config.get("continue_on_failure", True):
                    self.logger.warning("Stopping task execution due to goal failure")
                    break
            
            # Mark task as completed
            qa_task.mark_completed()
            
            # Perform final supervision if enabled
            if self.enable_supervision:
                await self._perform_final_supervision(qa_task)
            
            self.logger.info(f"QA task completed: {qa_task.name}")
            
            # Return overall task result
            return self._create_task_result(qa_task)
            
        finally:
            await self.android_integration.disconnect()
            self._cleanup_task_state()
    
    async def execute_test_goal(self, test_goal: TestGoal) -> TestResult:
        """Execute a single test goal"""
        self.state.current_goal = test_goal
        self.state.coordination_phase = "planning"
        
        # Create test result tracker
        test_result = TestResult(
            task_id=self.state.current_task.id,
            goal_id=test_goal.id
        )
        self.state.current_test_result = test_result
        
        try:
            # Activate relevant agents
            self._activate_agents(test_goal)
            
            # Phase 1: Planning
            await self._planning_phase(test_goal, test_result)
            
            # Phase 2: Execution
            await self._execution_phase(test_goal, test_result)
            
            # Phase 3: Final Verification
            await self._final_verification_phase(test_goal, test_result)
            
            # Phase 4: Supervision (if enabled)
            if self.enable_supervision:
                await self._supervision_phase(test_goal, test_result)
            
            # Mark test as completed
            test_result.mark_completed()
            
            return test_result
            
        except Exception as e:
            self.logger.error(f"Error executing test goal {test_goal.id}: {e}")
            test_result.overall_status = ResultStatus.FAILURE
            test_result.mark_completed()
            return test_result
        
        finally:
            self._deactivate_agents()
    
    async def _planning_phase(self, test_goal: TestGoal, test_result: TestResult):
        """Execute the planning phase"""
        self.state.coordination_phase = "planning"
        self.logger.info("Starting planning phase")
        
        # Prepare planning context
        planning_context = {
            "request_type": "create_plan",
            "goal_description": test_goal.description,
            "app_name": test_goal.app_name,
            "test_type": test_goal.test_type
        }
        
        # Get current UI state for context
        await self.android_integration.update_ui_state()
        ui_state = self.android_integration.get_current_ui_state()
        
        # Get planning decision
        planning_decision = await self.planner.process(planning_context, ui_state)
        test_result.add_agent_decision(planning_decision)
        
        if planning_decision.confidence_score < self.planner.config.confidence_threshold:
            self.logger.warning(f"Low confidence planning decision: {planning_decision.confidence_score}")
        
        # Extract subgoals from planning decision
        subgoals_data = planning_decision.output_data.get("subgoals", [])
        if not subgoals_data:
            raise RuntimeError("Planning failed: no subgoals generated")
        
        # Add subgoals to test goal
        for subgoal_data in subgoals_data:
            subgoal = self._dict_to_subgoal(subgoal_data)
            test_goal.add_subgoal(subgoal)
        
        self.logger.info(f"Planning completed: {len(test_goal.subgoals)} subgoals created")
    
    async def _execution_phase(self, test_goal: TestGoal, test_result: TestResult):
        """Execute the execution phase"""
        self.state.coordination_phase = "executing"
        self.logger.info("Starting execution phase")
        
        subgoal_index = 0
        max_subgoals = len(test_goal.subgoals)
        
        while subgoal_index < max_subgoals:
            # Get next subgoal
            if subgoal_index < len(test_goal.subgoals):
                subgoal = test_goal.subgoals[subgoal_index]
            else:
                break
            
            self.state.current_subgoal = subgoal
            self.logger.info(f"Executing subgoal {subgoal_index + 1}/{max_subgoals}: {subgoal.description}")
            
            try:
                # Execute subgoal with retry logic
                execution_success = await self._execute_subgoal_with_retry(subgoal, test_result)
                
                if execution_success:
                    subgoal_index += 1
                else:
                    # Handle failure
                    failure_handled = await self._handle_subgoal_failure(subgoal, test_goal, test_result, subgoal_index)
                    
                    if failure_handled:
                        # May have modified the plan, recalculate max_subgoals
                        max_subgoals = len(test_goal.subgoals)
                        continue
                    else:
                        # Unrecoverable failure
                        self.logger.error(f"Unrecoverable failure at subgoal {subgoal_index}")
                        break
                
            except Exception as e:
                self.logger.error(f"Error executing subgoal {subgoal.id}: {e}")
                subgoal.mark_failed(str(e))
                
                # Try to recover
                if not await self._attempt_error_recovery(subgoal, test_goal, test_result):
                    break
        
        self.logger.info("Execution phase completed")
    
    async def _execute_subgoal_with_retry(self, subgoal: SubGoal, test_result: TestResult) -> bool:
        """Execute a subgoal with retry logic"""
        for attempt in range(self.max_retries):
            try:
                # Update UI state before execution
                await self.android_integration.update_ui_state()
                ui_state = self.android_integration.get_current_ui_state()
                
                # Execute subgoal
                execution_context = {
                    "subgoal": self._subgoal_to_dict(subgoal)
                }
                
                execution_decision = await self.executor.process(execution_context, ui_state)
                test_result.add_agent_decision(execution_decision)
                
                # Get execution result
                execution_result_data = execution_decision.output_data.get("execution_result", {})
                execution_result = ExecutionResult(**execution_result_data)
                test_result.add_execution_result(execution_result)
                
                # Verify execution
                verification_success = await self._verify_subgoal_execution(subgoal, execution_decision, test_result)
                
                if verification_success:
                    subgoal.mark_completed()
                    return True
                else:
                    subgoal.retry_count += 1
                    if attempt < self.max_retries - 1:
                        self.logger.warning(f"Subgoal execution failed, retrying ({attempt + 1}/{self.max_retries})")
                        await asyncio.sleep(self.retry_delays[min(attempt, len(self.retry_delays) - 1)])
                    
            except Exception as e:
                self.logger.error(f"Subgoal execution attempt {attempt + 1} failed: {e}")
                subgoal.retry_count += 1
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delays[min(attempt, len(self.retry_delays) - 1)])
        
        # All retries failed
        subgoal.mark_failed(f"Failed after {self.max_retries} attempts")
        return False
    
    async def _verify_subgoal_execution(self, subgoal: SubGoal, execution_decision: AgentDecision, test_result: TestResult) -> bool:
        """Verify that a subgoal was executed correctly"""
        # Update UI state for verification
        await self.android_integration.update_ui_state()
        ui_state = self.android_integration.get_current_ui_state()
        
        # Prepare verification context
        verification_context = {
            "verification_type": "subgoal",
            "subgoal": self._subgoal_to_dict(subgoal),
            "execution_result": execution_decision.output_data.get("execution_result"),
            "expected_outcome": subgoal.expected_result
        }
        
        # Get verification decision
        verification_decision = await self.verifier.process(verification_context, ui_state)
        test_result.add_agent_decision(verification_decision)
        
        # Check verification result
        verification_passed = verification_decision.output_data.get("verification_passed", False)
        bug_detected = verification_decision.output_data.get("bug_detected", False)
        
        # Log bugs if detected
        if bug_detected:
            bug_details = verification_decision.output_data.get("bug_details", {})
            for bug in bug_details.get("bugs", []):
                test_result.add_bug_detection(
                    bug_type=bug.get("type", "unknown"),
                    description=bug.get("description", ""),
                    severity=bug.get("severity", "medium")
                )
        
        return verification_passed and not bug_detected
    
    async def _handle_subgoal_failure(self, failed_subgoal: SubGoal, test_goal: TestGoal, 
                                    test_result: TestResult, current_index: int) -> bool:
        """Handle subgoal failure and attempt recovery"""
        self.logger.warning(f"Handling failure for subgoal: {failed_subgoal.description}")
        
        if not self.enable_replanning:
            return False
        
        # Prepare replanning context
        replanning_context = {
            "request_type": "replan",
            "failed_subgoal": self._subgoal_to_dict(failed_subgoal),
            "error_details": {
                "error_message": failed_subgoal.error_message,
                "retry_count": failed_subgoal.retry_count
            },
            "remaining_subgoals": [
                self._subgoal_to_dict(sg) for sg in test_goal.subgoals[current_index + 1:]
            ]
        }
        
        # Get current UI state
        await self.android_integration.update_ui_state()
        ui_state = self.android_integration.get_current_ui_state()
        
        # Get replanning decision
        replanning_decision = await self.planner.process(replanning_context, ui_state)
        test_result.add_agent_decision(replanning_decision)
        test_result.replanning_count += 1
        
        # Apply replanning result
        action = replanning_decision.output_data.get("action", "abort")
        
        if action == "abort":
            self.logger.info("Planner recommends aborting due to unrecoverable failure")
            return False
        
        # Apply recovery plan
        recovery_subgoals = replanning_decision.output_data.get("recovery_subgoals", [])
        if recovery_subgoals:
            # Insert recovery subgoals
            for i, recovery_data in enumerate(recovery_subgoals):
                recovery_subgoal = self._dict_to_subgoal(recovery_data)
                test_goal.subgoals.insert(current_index + i, recovery_subgoal)
        
        # Skip to specific subgoal if specified
        skip_to = replanning_decision.output_data.get("skip_to_subgoal")
        if skip_to:
            # Find and skip to specified subgoal
            for i, subgoal in enumerate(test_goal.subgoals):
                if subgoal.id == skip_to:
                    # Mark skipped subgoals
                    for j in range(current_index, i):
                        test_goal.subgoals[j].status = TaskStatus.FAILED
                    current_index = i
                    break
        
        test_result.error_recovery_count += 1
        return True
    
    async def _attempt_error_recovery(self, failed_subgoal: SubGoal, test_goal: TestGoal, test_result: TestResult) -> bool:
        """Attempt generic error recovery"""
        self.state.error_count += 1
        
        if self.state.error_count >= self.state.max_errors:
            self.logger.error("Maximum error count reached, aborting")
            return False
        
        # Wait for UI to stabilize
        await self.android_integration.wait_for_ui_stable(timeout_seconds=5)
        
        # Clear executor cache
        self.executor.clear_element_cache()
        
        return True
    
    async def _final_verification_phase(self, test_goal: TestGoal, test_result: TestResult):
        """Perform final verification of goal completion"""
        self.state.coordination_phase = "verifying"
        self.logger.info("Starting final verification phase")
        
        # Update UI state
        await self.android_integration.update_ui_state()
        ui_state = self.android_integration.get_current_ui_state()
        
        # Prepare verification context
        verification_context = {
            "verification_type": "goal",
            "test_goal": {
                "id": test_goal.id,
                "title": test_goal.title,
                "description": test_goal.description,
                "expected_outcome": test_goal.description  # Simplified
            },
            "execution_summary": test_result.get_summary()
        }
        
        # Get final verification decision
        final_verification = await self.verifier.process(verification_context, ui_state)
        test_result.add_agent_decision(final_verification)
        
        # Update goal status based on verification
        goal_verified = final_verification.output_data.get("verification_passed", False)
        if goal_verified and test_goal.status != TaskStatus.FAILED:
            test_goal.mark_completed()
        
        self.logger.info(f"Final verification completed: {'passed' if goal_verified else 'failed'}")
    
    async def _supervision_phase(self, test_goal: TestGoal, test_result: TestResult):
        """Perform supervision and generate improvement recommendations"""
        self.state.coordination_phase = "supervising"
        self.logger.info("Starting supervision phase")
        
        # Prepare supervision context
        supervision_context = {
            "supervision_type": "test_review",
            "test_result": test_result.get_summary(),
            "test_goal": {
                "id": test_goal.id,
                "title": test_goal.title,
                "description": test_goal.description
            },
            "visual_trace": []  # Would contain screenshots in real implementation
        }
        
        # Get supervision decision
        supervision_decision = await self.supervisor.process(supervision_context)
        test_result.add_agent_decision(supervision_decision)
        
        # Extract supervision feedback
        supervision_feedback = supervision_decision.output_data
        test_result.supervisor_feedback = supervision_feedback
        test_result.improvement_suggestions = supervision_feedback.get("improvement_suggestions", [])
        
        # Log supervision results
        self.qa_logger.log_supervision_feedback(test_goal, supervision_feedback)
        
        self.logger.info("Supervision phase completed")
    
    async def _perform_final_supervision(self, qa_task: QATask):
        """Perform final supervision for the entire task"""
        self.logger.info("Performing final task supervision")
        
        # Collect all test results
        task_summary = qa_task.get_overall_progress()
        
        supervision_context = {
            "supervision_type": "system_optimization",
            "task_summary": task_summary,
            "task_performance": self._get_task_performance_metrics()
        }
        
        final_supervision = await self.supervisor.process(supervision_context)
        
        # Log final supervision
        self.qa_logger.log_task_supervision(qa_task, final_supervision.output_data)
    
    def _activate_agents(self, test_goal: TestGoal):
        """Activate agents for the test goal"""
        task = self.state.current_task
        
        self.planner.activate(task, test_goal)
        self.executor.activate(task, test_goal)
        self.verifier.activate(task, test_goal)
        self.supervisor.activate(task, test_goal)
        
        self.state.active_agents = ["planner", "executor", "verifier", "supervisor"]
    
    def _deactivate_agents(self):
        """Deactivate all agents"""
        for agent in self.agents.values():
            agent.deactivate()
        
        self.state.active_agents = []
    
    def _cleanup_task_state(self):
        """Clean up task state after completion"""
        self.state.current_task = None
        self.state.current_goal = None
        self.state.current_subgoal = None
        self.state.current_test_result = None
        self.state.coordination_phase = "idle"
        self.state.error_count = 0
    
    def _create_task_result(self, qa_task: QATask) -> TestResult:
        """Create overall task result"""
        task_result = TestResult(
            task_id=qa_task.id,
            goal_id="overall_task"
        )
        
        # Aggregate results from all goals
        for goal in qa_task.test_goals:
            task_result.total_subgoals += len(goal.subgoals)
            task_result.successful_subgoals += len([sg for sg in goal.subgoals if sg.status == TaskStatus.COMPLETED])
            task_result.failed_subgoals += len([sg for sg in goal.subgoals if sg.status == TaskStatus.FAILED])
        
        task_result.mark_completed()
        return task_result
    
    def _get_task_performance_metrics(self) -> Dict[str, Any]:
        """Get overall task performance metrics"""
        return {
            "total_goals": len(self.state.current_task.test_goals) if self.state.current_task else 0,
            "agent_performance": {
                agent_name: agent.get_performance_metrics() 
                for agent_name, agent in self.agents.items()
            },
            "android_integration_performance": self.android_integration.get_performance_metrics()
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
            "status": subgoal.status.value,
            "retry_count": subgoal.retry_count,
            "max_retries": subgoal.max_retries
        }
    
    def _dict_to_subgoal(self, subgoal_dict: Dict[str, Any]) -> SubGoal:
        """Convert dictionary to SubGoal"""
        subgoal = SubGoal(
            id=subgoal_dict.get("id"),
            description=subgoal_dict.get("description", ""),
            action_type=subgoal_dict.get("action_type", "touch"),
            target_element=subgoal_dict.get("target_element"),
            parameters=subgoal_dict.get("parameters", {}),
            expected_result=subgoal_dict.get("expected_result"),
            execution_order=subgoal_dict.get("execution_order", 0),
            max_retries=subgoal_dict.get("max_retries", 3)
        )
        
        # Set status if provided
        status_str = subgoal_dict.get("status")
        if status_str:
            subgoal.status = TaskStatus(status_str)
        
        subgoal.retry_count = subgoal_dict.get("retry_count", 0)
        
        return subgoal
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "coordination_phase": self.state.coordination_phase,
            "current_task": self.state.current_task.name if self.state.current_task else None,
            "current_goal": self.state.current_goal.title if self.state.current_goal else None,
            "current_subgoal": self.state.current_subgoal.description if self.state.current_subgoal else None,
            "active_agents": self.state.active_agents,
            "error_count": self.state.error_count,
            "android_connected": self.android_integration.is_connected,
            "agent_status": {
                name: agent.is_active for name, agent in self.agents.items()
            }
        } 