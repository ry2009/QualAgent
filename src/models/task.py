from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class SubGoal:
    """Represents a single actionable subgoal in a QA test"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    action_type: str = ""  # touch, type, scroll, wait, verify
    target_element: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_result: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    execution_order: int = 0
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    def mark_completed(self):
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()

    def mark_failed(self, error: str):
        self.status = TaskStatus.FAILED
        self.error_message = error
        self.completed_at = datetime.now()

    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries and self.status == TaskStatus.FAILED

@dataclass 
class TestGoal:
    """Represents a high-level QA test goal"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    app_name: str = ""
    test_type: str = ""  # functional, ui, integration, regression
    priority: Priority = Priority.MEDIUM
    subgoals: List[SubGoal] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: Optional[int] = None  # seconds
    actual_duration: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)

    def add_subgoal(self, subgoal: SubGoal):
        subgoal.execution_order = len(self.subgoals)
        self.subgoals.append(subgoal)

    def get_next_subgoal(self) -> Optional[SubGoal]:
        """Get the next pending subgoal to execute"""
        for subgoal in sorted(self.subgoals, key=lambda x: x.execution_order):
            if subgoal.status == TaskStatus.PENDING:
                return subgoal
        return None

    def get_failed_subgoals(self) -> List[SubGoal]:
        """Get all failed subgoals that can be retried"""
        return [sg for sg in self.subgoals if sg.status == TaskStatus.FAILED and sg.can_retry()]

    def mark_started(self):
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now()

    def mark_completed(self):
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        if self.started_at:
            self.actual_duration = int((self.completed_at - self.started_at).total_seconds())

    def get_completion_percentage(self) -> float:
        if not self.subgoals:
            return 0.0
        completed = len([sg for sg in self.subgoals if sg.status == TaskStatus.COMPLETED])
        return (completed / len(self.subgoals)) * 100

@dataclass
class QATask:
    """Main QA task that contains multiple test goals"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    app_under_test: str = ""
    test_goals: List[TestGoal] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: str = "system"
    environment: Dict[str, Any] = field(default_factory=dict)
    device_requirements: Dict[str, Any] = field(default_factory=dict)
    test_data: Dict[str, Any] = field(default_factory=dict)

    def add_test_goal(self, goal: TestGoal):
        self.test_goals.append(goal)

    def get_active_goal(self) -> Optional[TestGoal]:
        """Get the currently active test goal"""
        for goal in self.test_goals:
            if goal.status == TaskStatus.IN_PROGRESS:
                return goal
        # If no goal is in progress, get the next pending one
        for goal in self.test_goals:
            if goal.status == TaskStatus.PENDING:
                return goal
        return None

    def get_overall_progress(self) -> Dict[str, Any]:
        """Get overall progress statistics"""
        total_goals = len(self.test_goals)
        completed_goals = len([g for g in self.test_goals if g.status == TaskStatus.COMPLETED])
        failed_goals = len([g for g in self.test_goals if g.status == TaskStatus.FAILED])
        
        total_subgoals = sum(len(g.subgoals) for g in self.test_goals)
        completed_subgoals = sum(len([sg for sg in g.subgoals if sg.status == TaskStatus.COMPLETED]) for g in self.test_goals)
        
        return {
            "total_goals": total_goals,
            "completed_goals": completed_goals,
            "failed_goals": failed_goals,
            "goal_completion_rate": (completed_goals / total_goals * 100) if total_goals > 0 else 0,
            "total_subgoals": total_subgoals,
            "completed_subgoals": completed_subgoals,
            "subgoal_completion_rate": (completed_subgoals / total_subgoals * 100) if total_subgoals > 0 else 0
        }

    def mark_started(self):
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now()

    def mark_completed(self):
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        # Mark any remaining goals as completed if all subgoals are done
        for goal in self.test_goals:
            if goal.status == TaskStatus.IN_PROGRESS:
                if all(sg.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] for sg in goal.subgoals):
                    goal.mark_completed() 