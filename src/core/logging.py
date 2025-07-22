import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import threading
import asyncio
from dataclasses import asdict

from ..models.task import QATask, TestGoal, SubGoal
from ..models.result import TestResult, AgentDecision, ExecutionResult
from ..models.ui_state import UIState

class QALogger:
    """Comprehensive logging system for QA operations"""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 session_id: Optional[str] = None,
                 enable_file_logging: bool = True,
                 enable_console_logging: bool = True,
                 log_level: str = "INFO"):
        
        self.log_dir = Path(log_dir)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        
        # Create log directory
        if self.enable_file_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.session_dir = self.log_dir / f"session_{self.session_id}"
            self.session_dir.mkdir(exist_ok=True)
        
        # Initialize loggers
        self._setup_loggers(log_level)
        
        # Log files for different types of data
        self.log_files = {
            "tasks": self.session_dir / "tasks.jsonl" if self.enable_file_logging else None,
            "goals": self.session_dir / "goals.jsonl" if self.enable_file_logging else None,
            "subgoals": self.session_dir / "subgoals.jsonl" if self.enable_file_logging else None,
            "decisions": self.session_dir / "agent_decisions.jsonl" if self.enable_file_logging else None,
            "executions": self.session_dir / "executions.jsonl" if self.enable_file_logging else None,
            "ui_states": self.session_dir / "ui_states.jsonl" if self.enable_file_logging else None,
            "supervision": self.session_dir / "supervision.jsonl" if self.enable_file_logging else None,
            "performance": self.session_dir / "performance.jsonl" if self.enable_file_logging else None,
            "errors": self.session_dir / "errors.jsonl" if self.enable_file_logging else None
        }
        
        # Thread-safe logging
        self._lock = threading.Lock()
        
        # Session metadata
        self.session_metadata = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "version": "1.0.0",
            "log_level": log_level
        }
        
        self._write_session_metadata()
        
        self.logger.info(f"QA Logger initialized - Session: {self.session_id}")
    
    def _setup_loggers(self, log_level: str):
        """Setup logging infrastructure"""
        # Main logger
        self.logger = logging.getLogger("qa_logger")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        if self.enable_console_logging:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.enable_file_logging:
            file_handler = logging.FileHandler(self.session_dir / "qa_system.log")
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def _write_session_metadata(self):
        """Write session metadata"""
        if self.enable_file_logging:
            metadata_file = self.session_dir / "session_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(self.session_metadata, f, indent=2)
    
    def _write_json_log(self, log_type: str, data: Dict[str, Any]):
        """Write JSON log entry"""
        if not self.enable_file_logging:
            return
        
        log_file = self.log_files.get(log_type)
        if not log_file:
            return
        
        # Add timestamp and session info
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "log_type": log_type,
            **data
        }
        
        # Thread-safe writing
        with self._lock:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
    
    def log_task_start(self, task: QATask):
        """Log task start"""
        self.logger.info(f"Task started: {task.name}")
        
        task_data = {
            "event": "task_start",
            "task_id": task.id,
            "task_name": task.name,
            "description": task.description,
            "app_under_test": task.app_under_test,
            "total_goals": len(task.test_goals),
            "created_by": task.created_by,
            "environment": task.environment,
            "device_requirements": task.device_requirements
        }
        
        self._write_json_log("tasks", task_data)
    
    def log_task_completion(self, task: QATask, result: TestResult):
        """Log task completion"""
        self.logger.info(f"Task completed: {task.name} - Status: {result.overall_status.value}")
        
        progress = task.get_overall_progress()
        
        task_data = {
            "event": "task_complete",
            "task_id": task.id,
            "task_name": task.name,
            "status": task.status.value,
            "duration_seconds": (task.completed_at - task.started_at).total_seconds() if task.started_at and task.completed_at else 0,
            "progress": progress,
            "result_summary": result.get_summary()
        }
        
        self._write_json_log("tasks", task_data)
    
    def log_goal_start(self, goal: TestGoal):
        """Log goal start"""
        self.logger.info(f"Goal started: {goal.title}")
        
        goal_data = {
            "event": "goal_start",
            "goal_id": goal.id,
            "title": goal.title,
            "description": goal.description,
            "app_name": goal.app_name,
            "test_type": goal.test_type,
            "priority": goal.priority.value,
            "total_subgoals": len(goal.subgoals),
            "estimated_duration": goal.estimated_duration,
            "tags": goal.tags,
            "prerequisites": goal.prerequisites
        }
        
        self._write_json_log("goals", goal_data)
    
    def log_goal_completion(self, goal: TestGoal, result: TestResult):
        """Log goal completion"""
        self.logger.info(f"Goal completed: {goal.title} - {goal.get_completion_percentage():.1f}% complete")
        
        goal_data = {
            "event": "goal_complete",
            "goal_id": goal.id,
            "title": goal.title,
            "status": goal.status.value,
            "completion_percentage": goal.get_completion_percentage(),
            "actual_duration": goal.actual_duration,
            "estimated_duration": goal.estimated_duration,
            "result_summary": result.get_summary()
        }
        
        self._write_json_log("goals", goal_data)
    
    def log_subgoal_execution(self, subgoal: SubGoal, execution_result: ExecutionResult):
        """Log subgoal execution"""
        self.logger.info(f"Subgoal executed: {subgoal.description} - Status: {execution_result.status.value}")
        
        subgoal_data = {
            "event": "subgoal_execution",
            "subgoal_id": subgoal.id,
            "description": subgoal.description,
            "action_type": subgoal.action_type,
            "target_element": subgoal.target_element,
            "parameters": subgoal.parameters,
            "expected_result": subgoal.expected_result,
            "execution_order": subgoal.execution_order,
            "retry_count": subgoal.retry_count,
            "status": subgoal.status.value,
            "execution_result": execution_result.to_log_dict()
        }
        
        self._write_json_log("subgoals", subgoal_data)
    
    def log_agent_decision(self, decision: AgentDecision):
        """Log agent decision"""
        self.logger.debug(f"Agent decision: {decision.agent_type.value} - {decision.decision_type.value}")
        
        decision_data = decision.to_log_dict()
        decision_data["event"] = "agent_decision"
        
        self._write_json_log("decisions", decision_data)
    
    def log_execution_result(self, result: ExecutionResult):
        """Log execution result"""
        self.logger.debug(f"Execution result: {result.action_type} - {result.status.value}")
        
        execution_data = result.to_log_dict()
        execution_data["event"] = "execution_result"
        
        self._write_json_log("executions", execution_data)
    
    def log_ui_state(self, ui_state: UIState, context: str = ""):
        """Log UI state"""
        self.logger.debug(f"UI state captured: {len(ui_state.elements)} elements - {context}")
        
        ui_data = {
            "event": "ui_state_capture",
            "context": context,
            "ui_state": ui_state.to_dict()
        }
        
        self._write_json_log("ui_states", ui_data)
    
    def log_supervision_feedback(self, goal: TestGoal, feedback: Dict[str, Any]):
        """Log supervision feedback"""
        self.logger.info(f"Supervision feedback for goal: {goal.title}")
        
        supervision_data = {
            "event": "supervision_feedback",
            "goal_id": goal.id,
            "goal_title": goal.title,
            "feedback": feedback
        }
        
        self._write_json_log("supervision", supervision_data)
    
    def log_task_supervision(self, task: QATask, feedback: Dict[str, Any]):
        """Log task-level supervision"""
        self.logger.info(f"Task supervision for: {task.name}")
        
        supervision_data = {
            "event": "task_supervision",
            "task_id": task.id,
            "task_name": task.name,
            "feedback": feedback
        }
        
        self._write_json_log("supervision", supervision_data)
    
    def log_performance_metrics(self, metrics: Dict[str, Any], context: str = ""):
        """Log performance metrics"""
        self.logger.debug(f"Performance metrics logged: {context}")
        
        performance_data = {
            "event": "performance_metrics",
            "context": context,
            "metrics": metrics
        }
        
        self._write_json_log("performance", performance_data)
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context"""
        self.logger.error(f"Error occurred: {str(error)}", exc_info=True)
        
        error_data = {
            "event": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        
        self._write_json_log("errors", error_data)
    
    def log_bug_detection(self, bug_type: str, description: str, severity: str, evidence: Dict[str, Any] = None):
        """Log bug detection"""
        self.logger.warning(f"Bug detected: {bug_type} - {description}")
        
        bug_data = {
            "event": "bug_detection",
            "bug_type": bug_type,
            "description": description,
            "severity": severity,
            "evidence": evidence or {}
        }
        
        self._write_json_log("errors", bug_data)
    
    def log_system_event(self, event_type: str, details: Dict[str, Any]):
        """Log system-level events"""
        self.logger.info(f"System event: {event_type}")
        
        system_data = {
            "event": event_type,
            "details": details
        }
        
        # Route to appropriate log file based on event type
        log_type = "performance" if "performance" in event_type else "tasks"
        self._write_json_log(log_type, system_data)
    
    def create_execution_report(self, task: QATask) -> Dict[str, Any]:
        """Create comprehensive execution report"""
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "session_id": self.session_id,
                "task_id": task.id,
                "task_name": task.name
            },
            "task_summary": {
                "name": task.name,
                "description": task.description,
                "app_under_test": task.app_under_test,
                "total_goals": len(task.test_goals),
                "status": task.status.value,
                "progress": task.get_overall_progress()
            },
            "goals_summary": [],
            "performance_summary": {},
            "issues_summary": {
                "bugs_detected": [],
                "errors_encountered": [],
                "failures": []
            },
            "recommendations": []
        }
        
        # Add goal summaries
        for goal in task.test_goals:
            goal_summary = {
                "id": goal.id,
                "title": goal.title,
                "status": goal.status.value,
                "completion_percentage": goal.get_completion_percentage(),
                "total_subgoals": len(goal.subgoals),
                "successful_subgoals": len([sg for sg in goal.subgoals if sg.status.value == "completed"]),
                "failed_subgoals": len([sg for sg in goal.subgoals if sg.status.value == "failed"])
            }
            report["goals_summary"].append(goal_summary)
        
        # Save report
        if self.enable_file_logging:
            report_file = self.session_dir / f"execution_report_{task.id}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def create_session_summary(self) -> Dict[str, Any]:
        """Create session summary"""
        summary = {
            "session_metadata": self.session_metadata,
            "end_time": datetime.now().isoformat(),
            "log_files": {k: str(v) for k, v in self.log_files.items() if v},
            "statistics": self._calculate_session_statistics()
        }
        
        if self.enable_file_logging:
            summary_file = self.session_dir / "session_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
        
        return summary
    
    def _calculate_session_statistics(self) -> Dict[str, Any]:
        """Calculate session statistics from log files"""
        stats = {
            "total_tasks": 0,
            "total_goals": 0,
            "total_subgoals": 0,
            "total_decisions": 0,
            "total_executions": 0,
            "total_errors": 0,
            "total_bugs": 0
        }
        
        if not self.enable_file_logging:
            return stats
        
        try:
            # Count entries in each log file
            for log_type, log_file in self.log_files.items():
                if log_file and log_file.exists():
                    with open(log_file, 'r') as f:
                        count = sum(1 for line in f if line.strip())
                    stats[f"total_{log_type}"] = count
        
        except Exception as e:
            self.logger.error(f"Error calculating session statistics: {e}")
        
        return stats
    
    async def export_logs(self, export_format: str = "json", output_file: Optional[str] = None) -> str:
        """Export logs in specified format"""
        if export_format.lower() not in ["json", "csv"]:
            raise ValueError("Export format must be 'json' or 'csv'")
        
        if not output_file:
            output_file = f"qa_logs_export_{self.session_id}.{export_format.lower()}"
        
        try:
            if export_format.lower() == "json":
                await self._export_json(output_file)
            else:
                await self._export_csv(output_file)
            
            self.logger.info(f"Logs exported to: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            raise
    
    async def _export_json(self, output_file: str):
        """Export logs as consolidated JSON"""
        all_logs = {"session_metadata": self.session_metadata, "logs": {}}
        
        for log_type, log_file in self.log_files.items():
            if log_file and log_file.exists():
                logs = []
                with open(log_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            logs.append(json.loads(line))
                all_logs["logs"][log_type] = logs
        
        with open(output_file, 'w') as f:
            json.dump(all_logs, f, indent=2)
    
    async def _export_csv(self, output_file: str):
        """Export logs as CSV (simplified)"""
        import csv
        
        # Create a simplified CSV with key metrics
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'log_type', 'event', 'details']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for log_type, log_file in self.log_files.items():
                if log_file and log_file.exists():
                    with open(log_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                log_entry = json.loads(line)
                                writer.writerow({
                                    'timestamp': log_entry.get('timestamp'),
                                    'log_type': log_entry.get('log_type'),
                                    'event': log_entry.get('event'),
                                    'details': json.dumps(log_entry)
                                })
    
    def close(self):
        """Close logger and cleanup"""
        self.logger.info("Closing QA Logger")
        
        # Create final session summary
        self.create_session_summary()
        
        # Close handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler) 