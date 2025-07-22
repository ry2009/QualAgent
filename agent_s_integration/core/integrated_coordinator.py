"""
Integrated QA Coordinator combining Agent-S framework with QualGent enhancements
"""

import asyncio
import logging
import platform
from typing import Dict, Any, List, Optional
from datetime import datetime

# Agent-S imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Agent-S'))

from gui_agents.s2.agents.agent_s import UIAgent
from gui_agents.s2.agents.manager import Manager
from gui_agents.s2.agents.worker import Worker
from gui_agents.s2.agents.grounding import ACI

# QualGent imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from src.models.task import QATask, TestGoal
from src.models.result import TestResult, AgentDecision
from src.core.android_integration import AndroidWorldIntegration
from src.core.logging import QALogger

logger = logging.getLogger("integrated_qa_coordinator")


class IntegratedQACoordinator:
    """
    Integrated coordinator that combines Agent-S framework with QualGent QA enhancements
    
    Uses Agent-S for base UI automation while adding:
    - QA-specific planning and verification
    - Android mobile testing integration  
    - Comprehensive bug detection
    - Multi-agent coordination for QA tasks
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        android_integration: AndroidWorldIntegration,
        qa_logger: QALogger
    ):
        self.config = config
        self.android_integration = android_integration
        self.qa_logger = qa_logger
        self.current_task = None
        self.current_goal = None
        
        # Initialize Agent-S components
        self._initialize_agent_s_components()
        
        # Initialize QualGent enhancements
        self._initialize_qa_enhancements()
        
        logger.info("Integrated QA Coordinator initialized with Agent-S + QualGent")
    
    def _initialize_agent_s_components(self):
        """Initialize core Agent-S components"""
        
        # Engine parameters for LLM
        self.engine_params = {
            "engine_type": self.config.get("planner_provider", "google"),
            "model": self.config.get("planner_model", "gemini-1.5-flash"),
            "api_key": self.config.get("planner_api_key"),
            "temperature": self.config.get("planner_temperature", 0.1),
            "max_tokens": self.config.get("planner_max_tokens", 2000)
        }
        
        # Grounding agent parameters  
        self.grounding_params = {
            "engine_type": self.config.get("executor_provider", "google"),
            "model": self.config.get("executor_model", "gemini-1.5-flash"),
            "api_key": self.config.get("executor_api_key"),
        }
        
        # Initialize Agent-S grounding agent (simulated for Android)
        self.grounding_agent = MockAndroidACI(
            platform="android",
            android_integration=self.android_integration,
            engine_params_for_generation=self.engine_params,
            engine_params_for_grounding=self.grounding_params
        )
        
        # Initialize Agent-S Manager (our enhanced planner)
        self.agent_s_manager = Manager(
            engine_params=self.engine_params,
            grounding_agent=self.grounding_agent,
            local_kb_path="./kb_data",
            embedding_engine=None,  # Will be initialized if needed
            platform="android"
        )
        
        # Initialize Agent-S Worker (our enhanced executor)
        self.agent_s_worker = Worker(
            engine_params=self.engine_params,
            grounding_agent=self.grounding_agent,
            local_kb_path="./kb_data",
            embedding_engine=None,
            platform="android"
        )
        
        logger.info("Agent-S components initialized")
    
    def _initialize_qa_enhancements(self):
        """Initialize QualGent QA-specific enhancements"""
        
        # Import QualGent agents
        from ..agents.qa_verifier import QAVerifierAgent
        from ..agents.qa_supervisor import QASupervisorAgent
        
        # Initialize Verifier Agent (QualGent enhancement)
        self.verifier_agent = QAVerifierAgent(
            llm_client_config={
                "provider": self.config.get("verifier_provider", "google"),
                "model": self.config.get("verifier_model", "gemini-1.5-flash"),
                "api_key": self.config.get("verifier_api_key"),
                "temperature": self.config.get("verifier_temperature", 0.1),
                "max_tokens": self.config.get("verifier_max_tokens", 1500)
            },
            android_integration=self.android_integration
        )
        
        # Initialize Supervisor Agent (QualGent enhancement)
        self.supervisor_agent = QASupervisorAgent(
            llm_client_config={
                "provider": self.config.get("supervisor_provider", "google"),
                "model": self.config.get("supervisor_model", "gemini-1.5-flash"),
                "api_key": self.config.get("supervisor_api_key"),
                "temperature": self.config.get("supervisor_temperature", 0.2),
                "max_tokens": self.config.get("supervisor_max_tokens", 2500)
            }
        )
        
        # Coordination state
        self.coordination_phase = "idle"
        self.active_agents = set()
        self.error_count = 0
        
        logger.info("QualGent QA enhancements initialized")
    
    async def execute_qa_task(self, qa_task: QATask) -> TestResult:
        """
        Execute a complete QA task using integrated Agent-S + QualGent system
        
        Flow:
        1. Agent-S Manager plans the overall approach
        2. For each test goal:
           - Agent-S Worker executes actions
           - QualGent Verifier validates results
           - QualGent Supervisor provides oversight
        3. Generate comprehensive QA report
        """
        
        self.current_task = qa_task
        test_result = TestResult(task_id=qa_task.id)
        
        logger.info(f"Starting integrated QA task execution: {qa_task.name}")
        
        try:
            # Phase 1: Agent-S planning with QA context
            qa_plan = await self._create_qa_plan(qa_task)
            
            # Phase 2: Execute each test goal
            for goal in qa_task.test_goals:
                self.current_goal = goal
                goal_result = await self._execute_test_goal(goal, qa_plan)
                test_result.add_execution_result(goal_result)
            
            # Phase 3: QualGent supervision and evaluation
            final_supervision = await self._perform_final_qa_supervision(qa_task, test_result)
            test_result.supervisor_feedback = final_supervision
            
            test_result.mark_completed()
            
        except Exception as e:
            logger.error(f"Error in integrated QA task execution: {e}")
            test_result.overall_status = "error"
            raise
        
        logger.info(f"Integrated QA task completed: {test_result.overall_status}")
        return test_result
    
    async def _create_qa_plan(self, qa_task: QATask) -> Dict[str, Any]:
        """Use Agent-S Manager to create QA-specific plan"""
        
        self.coordination_phase = "planning"
        self.active_agents.add("agent_s_manager")
        
        # Create QA-specific instruction for Agent-S Manager
        qa_instruction = self._create_qa_instruction(qa_task)
        
        # Get current Android UI state
        ui_state = await self.android_integration.get_current_ui_state()
        
        # Use Agent-S Manager to create plan
        # Note: This would normally use Agent-S's planning methods
        # For now, we'll create a structured plan
        
        qa_plan = {
            "task_id": qa_task.id,
            "instruction": qa_instruction,
            "goals": [
                {
                    "goal_id": goal.id,
                    "title": goal.title,
                    "test_type": goal.test_type,
                    "priority": goal.priority.name,
                    "estimated_steps": self._estimate_steps_for_goal(goal)
                }
                for goal in qa_task.test_goals
            ],
            "overall_strategy": "systematic_mobile_qa_testing",
            "ui_context": ui_state.to_dict() if ui_state else None
        }
        
        # Log planning decision
        planning_decision = AgentDecision(
            agent_type="agent_s_manager",
            decision_type="plan_creation",
            input_data={"qa_task": qa_task.name},
            output_data=qa_plan,
            reasoning=f"Created QA plan for {len(qa_task.test_goals)} test goals",
            confidence_score=0.85
        )
        
        self.qa_logger.log_agent_decision(planning_decision)
        self.active_agents.remove("agent_s_manager")
        
        return qa_plan
    
    async def _execute_test_goal(self, goal: TestGoal, qa_plan: Dict[str, Any]) -> Any:
        """Execute a test goal using Agent-S Worker + QualGent Verifier"""
        
        logger.info(f"Executing test goal: {goal.title}")
        
        # Phase 1: Agent-S Worker execution
        execution_result = await self._execute_with_agent_s_worker(goal, qa_plan)
        
        # Phase 2: QualGent verification
        verification_result = await self._verify_with_qa_verifier(goal, execution_result)
        
        # Phase 3: Update goal status
        if verification_result.get("passed", False):
            goal.mark_completed()
        else:
            # Handle failed verification - could trigger replanning
            if self.config.get("enable_replanning", True):
                await self._handle_verification_failure(goal, verification_result)
        
        return execution_result
    
    async def _execute_with_agent_s_worker(self, goal: TestGoal, qa_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Use Agent-S Worker to execute test goal actions"""
        
        self.coordination_phase = "execution"
        self.active_agents.add("agent_s_worker")
        
        # Create instruction for Agent-S Worker
        worker_instruction = f"""
        Execute QA test goal: {goal.title}
        Description: {goal.description}
        Test Type: {goal.test_type}
        
        Context: Testing Android app {goal.app_name}
        Priority: {goal.priority.name}
        """
        
        # Get current UI state
        ui_state = await self.android_integration.get_current_ui_state()
        
        execution_result = {
            "goal_id": goal.id,
            "instruction": worker_instruction,
            "start_time": datetime.now(),
            "actions_executed": [],
            "ui_states_captured": [],
            "success": False,
            "error_message": None
        }
        
        try:
            # For demonstration, simulate Agent-S Worker execution
            # In reality, this would use Agent-S Worker's action execution
            
            # Mock execution based on test type
            if goal.test_type == "navigation":
                actions = await self._execute_navigation_actions(goal)
            elif goal.test_type == "functional":
                actions = await self._execute_functional_actions(goal)
            elif goal.test_type == "verification":
                actions = await self._execute_verification_actions(goal)
            else:
                actions = await self._execute_generic_actions(goal)
            
            execution_result["actions_executed"] = actions
            execution_result["success"] = True
            
        except Exception as e:
            execution_result["error_message"] = str(e)
            logger.error(f"Agent-S Worker execution failed: {e}")
        
        execution_result["end_time"] = datetime.now()
        
        # Log execution decision
        execution_decision = AgentDecision(
            agent_type="agent_s_worker",
            decision_type="action_execution",
            input_data={"goal": goal.title},
            output_data=execution_result,
            reasoning=f"Executed {len(execution_result['actions_executed'])} actions for {goal.test_type} test",
            confidence_score=0.8 if execution_result["success"] else 0.3
        )
        
        self.qa_logger.log_agent_decision(execution_decision)
        self.active_agents.remove("agent_s_worker")
        
        return execution_result
    
    async def _verify_with_qa_verifier(self, goal: TestGoal, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Use QualGent Verifier to validate execution results"""
        
        self.coordination_phase = "verification"
        self.active_agents.add("qa_verifier")
        
        # Use QualGent Verifier
        verification_context = {
            "goal": goal,
            "execution_result": execution_result,
            "expected_outcome": goal.description
        }
        
        verification_result = await self.verifier_agent.process(verification_context)
        
        self.active_agents.remove("qa_verifier")
        
        return verification_result
    
    async def _perform_final_qa_supervision(self, qa_task: QATask, test_result: TestResult) -> Dict[str, Any]:
        """Use QualGent Supervisor for final evaluation"""
        
        self.coordination_phase = "supervision"
        self.active_agents.add("qa_supervisor")
        
        supervision_context = {
            "qa_task": qa_task,
            "test_result": test_result,
            "overall_progress": qa_task.get_overall_progress()
        }
        
        final_supervision = await self.supervisor_agent.process(supervision_context)
        
        self.active_agents.remove("qa_supervisor")
        self.coordination_phase = "idle"
        
        return final_supervision
    
    # Helper methods for different action types
    async def _execute_navigation_actions(self, goal: TestGoal) -> List[Dict[str, Any]]:
        """Execute navigation-specific actions"""
        actions = []
        
        # Simulate navigation actions
        actions.append({
            "type": "touch",
            "target": "settings_icon",
            "coordinates": (540, 960),
            "timestamp": datetime.now(),
            "success": True
        })
        
        return actions
    
    async def _execute_functional_actions(self, goal: TestGoal) -> List[Dict[str, Any]]:
        """Execute functional test actions"""
        actions = []
        
        # Simulate functional actions  
        actions.append({
            "type": "verify_element",
            "target": "wifi_toggle",
            "expected_state": "enabled",
            "timestamp": datetime.now(),
            "success": True
        })
        
        return actions
    
    async def _execute_verification_actions(self, goal: TestGoal) -> List[Dict[str, Any]]:
        """Execute verification actions"""
        actions = []
        
        # Simulate verification actions
        actions.append({
            "type": "screenshot",
            "purpose": "state_verification", 
            "timestamp": datetime.now(),
            "success": True
        })
        
        return actions
    
    async def _execute_generic_actions(self, goal: TestGoal) -> List[Dict[str, Any]]:
        """Execute generic actions"""
        return [{
            "type": "generic_test",
            "goal": goal.title,
            "timestamp": datetime.now(),
            "success": True
        }]
    
    def _create_qa_instruction(self, qa_task: QATask) -> str:
        """Create QA-specific instruction for Agent-S"""
        return f"""
        Execute comprehensive QA testing for: {qa_task.name}
        
        Description: {qa_task.description}
        App Under Test: {qa_task.app_under_test}
        
        Test Goals ({len(qa_task.test_goals)}):
        {chr(10).join(f"- {goal.title}: {goal.description}" for goal in qa_task.test_goals)}
        
        Requirements:
        - Perform systematic mobile app testing
        - Validate UI functionality and behavior
        - Detect bugs and usability issues
        - Ensure comprehensive test coverage
        """
    
    def _estimate_steps_for_goal(self, goal: TestGoal) -> int:
        """Estimate number of steps needed for a goal"""
        if goal.test_type == "navigation":
            return 3
        elif goal.test_type == "functional":
            return 5
        elif goal.test_type == "verification":
            return 2
        else:
            return 4
    
    async def _handle_verification_failure(self, goal: TestGoal, verification_result: Dict[str, Any]):
        """Handle verification failure with potential replanning"""
        logger.warning(f"Verification failed for goal: {goal.title}")
        
        # Could trigger Agent-S Manager replanning here
        # For now, just log the failure
        self.error_count += 1
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get integrated system status"""
        return {
            "coordination_phase": self.coordination_phase,
            "current_task": self.current_task.name if self.current_task else None,
            "current_goal": self.current_goal.title if self.current_goal else None,
            "active_agents": list(self.active_agents),
            "error_count": self.error_count,
            "agent_s_components": {
                "manager_initialized": hasattr(self, 'agent_s_manager'),
                "worker_initialized": hasattr(self, 'agent_s_worker'),
                "grounding_agent_initialized": hasattr(self, 'grounding_agent')
            },
            "qa_enhancements": {
                "verifier_initialized": hasattr(self, 'verifier_agent'),
                "supervisor_initialized": hasattr(self, 'supervisor_agent')
            },
            "android_integration": self.android_integration.get_performance_metrics()
        }


class MockAndroidACI(ACI):
    """Mock Android ACI that integrates with QualGent's Android system"""
    
    def __init__(self, platform: str, android_integration: AndroidWorldIntegration, 
                 engine_params_for_generation: Dict, engine_params_for_grounding: Dict):
        self.platform = platform
        self.android_integration = android_integration
        self.engine_params_for_generation = engine_params_for_generation
        self.engine_params_for_grounding = engine_params_for_grounding
    
    async def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action through QualGent's Android integration"""
        try:
            # Route action through our Android integration
            result = await self.android_integration.execute_action(
                action_type=action.get("type", "touch"),
                element_id=action.get("element_id"),
                coordinates=action.get("coordinates"),
                text=action.get("text")
            )
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)} 