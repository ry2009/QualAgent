from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

class AgentType(Enum):
    PLANNER = "planner"
    EXECUTOR = "executor"
    VERIFIER = "verifier"
    SUPERVISOR = "supervisor"

class DecisionType(Enum):
    PLAN_CREATION = "plan_creation"
    PLAN_MODIFICATION = "plan_modification"
    ACTION_EXECUTION = "action_execution"
    VERIFICATION = "verification"
    SUPERVISION = "supervision"
    ERROR_HANDLING = "error_handling"
    RECOVERY = "recovery"

class ResultStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    ERROR = "error"

@dataclass
class AgentDecision:
    """Represents a decision made by any agent in the system"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: AgentType = AgentType.PLANNER
    decision_type: DecisionType = DecisionType.PLAN_CREATION
    timestamp: datetime = field(default_factory=datetime.now)
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    confidence_score: float = 0.0  # 0.0 to 1.0
    execution_time_ms: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    parent_decision_id: Optional[str] = None
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    
    def to_log_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "id": self.id,
            "agent": self.agent_type.value,
            "decision_type": self.decision_type.value,
            "timestamp": self.timestamp.isoformat(),
            "reasoning": self.reasoning,
            "confidence": self.confidence_score,
            "execution_time_ms": self.execution_time_ms,
            "model": self.model_used,
            "tokens": self.tokens_used,
            "input_summary": self._summarize_data(self.input_data),
            "output_summary": self._summarize_data(self.output_data)
        }
    
    def _summarize_data(self, data: Dict[str, Any]) -> str:
        """Create a summary of input/output data for logging"""
        if not data:
            return "empty"
        summary_parts = []
        for key, value in data.items():
            if isinstance(value, (str, int, float, bool)):
                summary_parts.append(f"{key}:{value}")
            elif isinstance(value, (list, dict)):
                summary_parts.append(f"{key}:[{type(value).__name__}]")
        return ", ".join(summary_parts[:5])  # Limit to first 5 items

@dataclass
class ExecutionResult:
    """Result of executing a specific action or subgoal"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subgoal_id: str = ""
    status: ResultStatus = ResultStatus.SUCCESS
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time_ms: int = 0
    action_type: str = ""
    target_element: Optional[str] = None
    parameters_used: Dict[str, Any] = field(default_factory=dict)
    
    # Screenshot data
    screenshot_before: Optional[bytes] = None
    screenshot_after: Optional[bytes] = None
    
    # UI state information
    ui_elements_before: List[Dict[str, Any]] = field(default_factory=list)
    ui_elements_after: List[Dict[str, Any]] = field(default_factory=list)
    
    # Execution details
    actual_coordinates: Optional[tuple] = None
    text_entered: Optional[str] = None
    scroll_distance: Optional[tuple] = None
    
    # Result verification
    expected_outcome: Optional[str] = None
    actual_outcome: Optional[str] = None
    verification_passed: bool = False
    
    # Error information
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Agent decisions that led to this result
    related_decisions: List[str] = field(default_factory=list)
    
    def mark_success(self, actual_outcome: str = None):
        """Mark execution as successful"""
        self.status = ResultStatus.SUCCESS
        self.verification_passed = True
        if actual_outcome:
            self.actual_outcome = actual_outcome
    
    def mark_failure(self, error_msg: str, error_type: str = None):
        """Mark execution as failed"""
        self.status = ResultStatus.FAILURE
        self.verification_passed = False
        self.error_message = error_msg
        self.error_type = error_type or "execution_error"
    
    def add_decision(self, decision_id: str):
        """Add a related agent decision"""
        if decision_id not in self.related_decisions:
            self.related_decisions.append(decision_id)
    
    def to_log_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "id": self.id,
            "subgoal_id": self.subgoal_id,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "execution_time_ms": self.execution_time_ms,
            "action_type": self.action_type,
            "target_element": self.target_element,
            "verification_passed": self.verification_passed,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "actual_coordinates": self.actual_coordinates,
            "related_decisions": self.related_decisions
        }

@dataclass
class TestResult:
    """Overall result of a complete test execution"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    goal_id: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_duration_ms: int = 0
    
    # Results tracking
    execution_results: List[ExecutionResult] = field(default_factory=list)
    agent_decisions: List[AgentDecision] = field(default_factory=list)
    
    # Status and metrics
    overall_status: ResultStatus = ResultStatus.SUCCESS
    total_subgoals: int = 0
    successful_subgoals: int = 0
    failed_subgoals: int = 0
    skipped_subgoals: int = 0
    
    # Performance metrics
    average_execution_time_ms: float = 0.0
    success_rate: float = 0.0
    error_recovery_count: int = 0
    replanning_count: int = 0
    
    # Bug detection
    bugs_detected: List[Dict[str, Any]] = field(default_factory=list)
    false_positives: int = 0
    false_negatives: int = 0
    
    # Supervisor feedback
    supervisor_feedback: Dict[str, Any] = field(default_factory=dict)
    improvement_suggestions: List[str] = field(default_factory=list)
    
    # Resource usage
    total_tokens_used: int = 0
    total_api_calls: int = 0
    screenshots_captured: int = 0
    
    def add_execution_result(self, result: ExecutionResult):
        """Add an execution result"""
        self.execution_results.append(result)
        
        # Update counters
        if result.status == ResultStatus.SUCCESS:
            self.successful_subgoals += 1
        elif result.status == ResultStatus.FAILURE:
            self.failed_subgoals += 1
        elif result.status == ResultStatus.SKIPPED:
            self.skipped_subgoals += 1
            
        self.total_subgoals = len(self.execution_results)
        self._update_metrics()
    
    def add_agent_decision(self, decision: AgentDecision):
        """Add an agent decision"""
        self.agent_decisions.append(decision)
        self.total_api_calls += 1
        if decision.tokens_used:
            self.total_tokens_used += decision.tokens_used
    
    def add_bug_detection(self, bug_type: str, description: str, severity: str = "medium", 
                         element_id: str = None, screenshot: bytes = None):
        """Record a detected bug"""
        bug_report = {
            "id": str(uuid.uuid4()),
            "type": bug_type,
            "description": description,
            "severity": severity,
            "element_id": element_id,
            "timestamp": datetime.now().isoformat(),
            "has_screenshot": screenshot is not None
        }
        self.bugs_detected.append(bug_report)
    
    def mark_completed(self):
        """Mark the test as completed and calculate final metrics"""
        self.end_time = datetime.now()
        self.total_duration_ms = int((self.end_time - self.start_time).total_seconds() * 1000)
        self._update_metrics()
        self._determine_overall_status()
    
    def _update_metrics(self):
        """Update calculated metrics"""
        if self.total_subgoals > 0:
            self.success_rate = self.successful_subgoals / self.total_subgoals
        
        if self.execution_results:
            total_time = sum(r.execution_time_ms for r in self.execution_results)
            self.average_execution_time_ms = total_time / len(self.execution_results)
    
    def _determine_overall_status(self):
        """Determine the overall test status"""
        if self.failed_subgoals == 0:
            self.overall_status = ResultStatus.SUCCESS
        elif self.successful_subgoals == 0:
            self.overall_status = ResultStatus.FAILURE
        else:
            self.overall_status = ResultStatus.PARTIAL
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the test results"""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "overall_status": self.overall_status.value,
            "duration_ms": self.total_duration_ms,
            "success_rate": round(self.success_rate * 100, 2),
            "total_subgoals": self.total_subgoals,
            "successful_subgoals": self.successful_subgoals,
            "failed_subgoals": self.failed_subgoals,
            "bugs_detected": len(self.bugs_detected),
            "error_recoveries": self.error_recovery_count,
            "replannings": self.replanning_count,
            "tokens_used": self.total_tokens_used,
            "api_calls": self.total_api_calls,
            "improvement_suggestions": len(self.improvement_suggestions)
        } 