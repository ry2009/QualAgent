import json
import statistics
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

from .base import BaseAgent, AgentConfig
from ..models.result import AgentDecision, AgentType, DecisionType, TestResult, ExecutionResult
from ..models.task import QATask, TestGoal
from ..models.ui_state import UIState
from ..core.llm_client import LLMClient, LLMMessage

class SupervisorAgent(BaseAgent):
    """Agent responsible for supervising test execution and providing strategic improvements"""
    
    def __init__(self, config: AgentConfig, logger: Optional[logging.Logger] = None):
        super().__init__(AgentType.SUPERVISOR, config, logger)
        
    def _initialize_agent(self):
        """Initialize supervisor-specific components"""
        self.llm_client = LLMClient(
            provider=self.config.provider,
            model=self.config.model_name,
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )
        
        # Evaluation history for trend analysis
        self.evaluation_history: List[Dict[str, Any]] = []
        
        # Benchmarks and thresholds
        self.performance_benchmarks = {
            "success_rate_threshold": 0.85,
            "average_execution_time_threshold": 5000,  # ms
            "bug_detection_accuracy_threshold": 0.90,
            "false_positive_threshold": 0.10
        }
        
    async def make_decision(self, context: Dict[str, Any], ui_state: Optional[UIState] = None) -> AgentDecision:
        """Make a supervision decision based on test results and agent performance"""
        supervision_type = context.get("supervision_type", "test_review")
        
        if supervision_type == "test_review":
            return await self._review_test_execution(context, ui_state)
        else:
            return await self._comprehensive_supervision(context, ui_state)
    
    async def _review_test_execution(self, context: Dict[str, Any], ui_state: Optional[UIState]) -> AgentDecision:
        """Review a completed test execution and provide feedback"""
        test_result = context.get("test_result")
        
        if not test_result:
            return self.create_decision(
                decision_type=DecisionType.SUPERVISION,
                reasoning="No test result provided for review",
                confidence=0.0,
                output_data={"error": "missing_test_result"}
            )
        
        # Simple supervision analysis
        success_rate = test_result.get("success_rate", 0.0)
        duration = test_result.get("total_duration_ms", 0)
        bugs_detected = test_result.get("bugs_detected", [])
        
        # Generate basic feedback
        feedback = {
            "overall_assessment": self._generate_assessment(success_rate),
            "quality_score": success_rate,
            "improvement_suggestions": self._generate_suggestions(success_rate, duration),
            "benchmarks_met": self._check_benchmarks(test_result)
        }
        
        return self.create_decision(
            decision_type=DecisionType.SUPERVISION,
            reasoning=f"Test supervision completed: {feedback['overall_assessment']}",
            confidence=0.9,
            output_data=feedback
        )
    
    async def _comprehensive_supervision(self, context: Dict[str, Any], ui_state: Optional[UIState]) -> AgentDecision:
        """Perform comprehensive supervision"""
        return self.create_decision(
            decision_type=DecisionType.SUPERVISION,
            reasoning="Comprehensive supervision completed",
            confidence=0.8,
            output_data={"status": "completed"}
        )
    
    def _generate_assessment(self, success_rate: float) -> str:
        """Generate overall assessment"""
        if success_rate >= 0.9:
            return "Excellent: Test execution meets all quality standards"
        elif success_rate >= 0.8:
            return "Good: Test execution is solid with minor areas for improvement"
        elif success_rate >= 0.6:
            return "Fair: Test execution shows moderate success"
        else:
            return "Poor: Test execution needs significant improvements"
    
    def _generate_suggestions(self, success_rate: float, duration: int) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        if success_rate < 0.85:
            suggestions.append("Improve element locating strategies to increase success rate")
        
        if duration > 30000:  # 30 seconds
            suggestions.append("Optimize execution timing to reduce overall test duration")
        
        return suggestions
    
    def _check_benchmarks(self, test_result: Dict[str, Any]) -> Dict[str, bool]:
        """Check if test result meets performance benchmarks"""
        return {
            "success_rate": test_result.get("success_rate", 0) >= self.performance_benchmarks["success_rate_threshold"],
            "execution_time": test_result.get("average_execution_time_ms", 0) <= self.performance_benchmarks["average_execution_time_threshold"]
        }
    
    async def _validate_agent_specific(self, decision: AgentDecision) -> bool:
        """Validate supervisor-specific decisions"""
        return True 