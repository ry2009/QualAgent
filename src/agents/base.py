from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
import time
import logging
from dataclasses import dataclass

from ..models.result import AgentDecision, AgentType, DecisionType
from ..models.ui_state import UIState
from ..models.task import QATask, TestGoal, SubGoal

@dataclass
class AgentConfig:
    """Configuration for an agent"""
    model_name: str = "gpt-4"
    provider: str = "openai"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_retries: int = 3
    timeout_seconds: int = 30
    temperature: float = 0.1
    max_tokens: int = 2000
    confidence_threshold: float = 0.7

class BaseAgent(ABC):
    """Base class for all agents in the multi-agent QA system"""
    
    def __init__(self, 
                 agent_type: AgentType, 
                 config: AgentConfig,
                 logger: Optional[logging.Logger] = None):
        self.agent_type = agent_type
        self.config = config
        self.logger = logger or logging.getLogger(f"{agent_type.value}_agent")
        
        # Agent state
        self.is_active = False
        self.current_task: Optional[QATask] = None
        self.current_goal: Optional[TestGoal] = None
        self.decision_history: List[AgentDecision] = []
        
        # Performance tracking
        self.total_decisions = 0
        self.successful_decisions = 0
        self.average_response_time = 0.0
        
        # Initialize the agent-specific components
        self._initialize_agent()
    
    @abstractmethod
    def _initialize_agent(self):
        """Initialize agent-specific components"""
        pass
    
    @abstractmethod
    async def make_decision(self, 
                          context: Dict[str, Any],
                          ui_state: Optional[UIState] = None) -> AgentDecision:
        """Make a decision based on the current context and UI state"""
        pass
    
    def activate(self, task: QATask, goal: Optional[TestGoal] = None):
        """Activate the agent for a specific task and goal"""
        self.is_active = True
        self.current_task = task
        self.current_goal = goal
        self.logger.info(f"{self.agent_type.value} agent activated for task: {task.name}")
    
    def deactivate(self):
        """Deactivate the agent"""
        self.is_active = False
        self.current_task = None
        self.current_goal = None
        self.logger.info(f"{self.agent_type.value} agent deactivated")
    
    async def process(self, 
                     context: Dict[str, Any],
                     ui_state: Optional[UIState] = None) -> AgentDecision:
        """Process a request and return a decision"""
        if not self.is_active:
            raise RuntimeError(f"{self.agent_type.value} agent is not active")
        
        start_time = time.time()
        
        try:
            # Pre-processing
            context = await self._preprocess_context(context, ui_state)
            
            # Make the decision
            decision = await self.make_decision(context, ui_state)
            
            # Post-processing
            decision = await self._postprocess_decision(decision, context)
            
            # Record performance
            execution_time = int((time.time() - start_time) * 1000)
            decision.execution_time_ms = execution_time
            
            # Update statistics
            self._update_performance_stats(decision, execution_time)
            
            # Store decision
            self.decision_history.append(decision)
            
            self.logger.info(f"Decision made: {decision.decision_type.value} "
                           f"(confidence: {decision.confidence_score:.2f}, "
                           f"time: {execution_time}ms)")
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            
            # Create error decision
            error_decision = AgentDecision(
                agent_type=self.agent_type,
                decision_type=DecisionType.ERROR_HANDLING,
                reasoning=f"Error occurred: {str(e)}",
                confidence_score=0.0,
                execution_time_ms=int((time.time() - start_time) * 1000),
                context={"error": str(e), "original_context": context}
            )
            
            self.decision_history.append(error_decision)
            return error_decision
    
    async def _preprocess_context(self, 
                                context: Dict[str, Any], 
                                ui_state: Optional[UIState]) -> Dict[str, Any]:
        """Preprocess the context before making a decision"""
        # Add agent-specific context
        context["agent_type"] = self.agent_type.value
        context["timestamp"] = datetime.now().isoformat()
        
        # Add task and goal context if available
        if self.current_task:
            context["task_id"] = self.current_task.id
            context["task_name"] = self.current_task.name
            context["app_under_test"] = self.current_task.app_under_test
        
        if self.current_goal:
            context["goal_id"] = self.current_goal.id
            context["goal_title"] = self.current_goal.title
            context["goal_progress"] = self.current_goal.get_completion_percentage()
        
        # Add UI state summary if available
        if ui_state:
            context["ui_summary"] = ui_state.get_interactable_elements_summary()
        
        return context
    
    async def _postprocess_decision(self, 
                                  decision: AgentDecision, 
                                  context: Dict[str, Any]) -> AgentDecision:
        """Postprocess the decision after it's made"""
        # Add context to decision
        decision.context.update({
            "task_id": context.get("task_id"),
            "goal_id": context.get("goal_id"),
            "agent_config": {
                "model": self.config.model_name,
                "provider": self.config.provider,
                "temperature": self.config.temperature
            }
        })
        
        # Validate confidence score
        if decision.confidence_score < self.config.confidence_threshold:
            self.logger.warning(f"Low confidence decision: {decision.confidence_score:.2f}")
            decision.context["low_confidence_warning"] = True
        
        return decision
    
    def _update_performance_stats(self, decision: AgentDecision, execution_time: int):
        """Update agent performance statistics"""
        self.total_decisions += 1
        
        if decision.confidence_score >= self.config.confidence_threshold:
            self.successful_decisions += 1
        
        # Update average response time
        self.average_response_time = (
            (self.average_response_time * (self.total_decisions - 1) + execution_time) 
            / self.total_decisions
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        success_rate = (
            self.successful_decisions / self.total_decisions 
            if self.total_decisions > 0 else 0.0
        )
        
        return {
            "agent_type": self.agent_type.value,
            "total_decisions": self.total_decisions,
            "successful_decisions": self.successful_decisions,
            "success_rate": round(success_rate * 100, 2),
            "average_response_time_ms": round(self.average_response_time, 2),
            "is_active": self.is_active,
            "current_task_id": self.current_task.id if self.current_task else None,
            "current_goal_id": self.current_goal.id if self.current_goal else None,
            "decision_history_length": len(self.decision_history)
        }
    
    def get_recent_decisions(self, count: int = 10) -> List[AgentDecision]:
        """Get the most recent decisions made by this agent"""
        return self.decision_history[-count:] if self.decision_history else []
    
    def clear_history(self):
        """Clear decision history (useful for new tasks)"""
        self.decision_history.clear()
        self.logger.info(f"{self.agent_type.value} agent history cleared")
    
    async def validate_decision(self, decision: AgentDecision) -> bool:
        """Validate a decision before execution"""
        # Basic validation
        if not decision.reasoning:
            self.logger.warning("Decision lacks reasoning")
            return False
        
        if decision.confidence_score <= 0:
            self.logger.warning("Decision has zero or negative confidence")
            return False
        
        # Agent-specific validation
        return await self._validate_agent_specific(decision)
    
    @abstractmethod
    async def _validate_agent_specific(self, decision: AgentDecision) -> bool:
        """Agent-specific decision validation"""
        pass
    
    def create_decision(self, 
                       decision_type: DecisionType,
                       reasoning: str,
                       confidence: float,
                       output_data: Dict[str, Any] = None,
                       input_data: Dict[str, Any] = None) -> AgentDecision:
        """Helper method to create a decision with common fields"""
        return AgentDecision(
            agent_type=self.agent_type,
            decision_type=decision_type,
            reasoning=reasoning,
            confidence_score=confidence,
            output_data=output_data or {},
            input_data=input_data or {},
            model_used=self.config.model_name
        )
    
    def __str__(self):
        return f"{self.agent_type.value.title()}Agent(active={self.is_active})"
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(type={self.agent_type.value}, "
                f"active={self.is_active}, decisions={self.total_decisions})") 