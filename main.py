#!/usr/bin/env python3
"""
QualGent - Multi-Agent QA System
Main application entry point
"""

import asyncio
import logging
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.coordinator import MultiAgentCoordinator
from src.core.android_integration import AndroidWorldIntegration
from src.core.logging import QALogger
from src.core.evaluation import EvaluationEngine
from src.models.task import QATask, TestGoal, SubGoal, Priority
from src.models.result import TestResult

class QualGentApp:
    """Main QualGent application class"""
    
    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        
        # Initialize core components
        self.qa_logger = QALogger(
            log_dir=self.config.get("log_dir", "logs"),
            enable_file_logging=self.config.get("enable_file_logging", True),
            enable_console_logging=self.config.get("enable_console_logging", True),
            log_level=self.config.get("log_level", "INFO")
        )
        
        self.android_integration = AndroidWorldIntegration(
            task_name=self.config.get("android_task", "default"),
            avd_name=self.config.get("avd_name", "AndroidWorldAvd"),
            enable_screenshots=self.config.get("enable_screenshots", True),
            screenshot_quality=self.config.get("screenshot_quality", 80)
        )
        
        self.coordinator = MultiAgentCoordinator(
            config=self.config,
            android_integration=self.android_integration,
            logger=self.logger
        )
        
        self.evaluation_engine = EvaluationEngine(
            baseline_metrics=self.config.get("baseline_metrics"),
            logger=self.logger
        )
        
        self.logger.info("QualGent application initialized")
    
    def _load_config(self, config_file: str = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            # General settings
            "log_dir": "logs",
            "enable_file_logging": True,
            "enable_console_logging": True,
            "log_level": "INFO",
            
            # Android settings
            "android_task": "default",
            "avd_name": "AndroidWorldAvd",
            "enable_screenshots": True,
            "screenshot_quality": 80,
            
            # Agent settings
            "planner_model": "gpt-4",
            "planner_provider": "openai",
            "planner_temperature": 0.1,
            "planner_max_tokens": 2000,
            
            "executor_model": "gpt-4",
            "executor_provider": "openai", 
            "executor_temperature": 0.1,
            "executor_max_tokens": 1500,
            
            "verifier_model": "gpt-4",
            "verifier_provider": "openai",
            "verifier_temperature": 0.1,
            "verifier_max_tokens": 1500,
            
            "supervisor_model": "gpt-4",
            "supervisor_provider": "openai",
            "supervisor_temperature": 0.2,
            "supervisor_max_tokens": 2500,
            
            # Coordination settings
            "max_retries": 3,
            "enable_replanning": True,
            "enable_supervision": True,
            "continue_on_failure": True,
            
            # API keys (from environment variables or config file)
            "planner_api_key": os.getenv("OPENAI_API_KEY"),
            "executor_api_key": os.getenv("OPENAI_API_KEY"),
            "verifier_api_key": os.getenv("OPENAI_API_KEY"),
            "supervisor_api_key": os.getenv("OPENAI_API_KEY"),
            
            # Baseline metrics for evaluation
            "baseline_metrics": {
                "success_rate": 0.85,
                "bug_detection_accuracy": 0.90,
                "false_positive_rate": 0.10,
                "average_execution_time": 5000,
                "efficiency_score": 0.80,
                "consistency_score": 0.85
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                default_config.update(file_config)
            except Exception as e:
                print(f"Warning: Could not load config file {config_file}: {e}")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup application logging"""
        logger = logging.getLogger("qualgent_app")
        logger.setLevel(getattr(logging, self.config.get("log_level", "INFO")))
        
        # Console handler
        if not logger.handlers:  # Avoid duplicate handlers
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    async def run_qa_task(self, task_definition: Dict[str, Any]) -> TestResult:
        """Run a complete QA task"""
        self.logger.info(f"Starting QA task: {task_definition.get('name', 'Unnamed Task')}")
        
        try:
            # Create QA task from definition
            qa_task = self._create_qa_task_from_definition(task_definition)
            
            # Log task start
            self.qa_logger.log_task_start(qa_task)
            
            # Execute task
            task_result = await self.coordinator.execute_qa_task(qa_task)
            
            # Log task completion
            self.qa_logger.log_task_completion(qa_task, task_result)
            
            # Evaluate task execution
            evaluation_report = await self.evaluation_engine.evaluate_task_execution(
                task=qa_task,
                test_results=[task_result],
                agent_decisions=task_result.agent_decisions
            )
            
            # Export evaluation report
            report_file = f"evaluation_report_{qa_task.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.evaluation_engine.export_evaluation_report(evaluation_report, report_file)
            
            self.logger.info(f"QA task completed. Overall score: {evaluation_report['overall_score']:.3f}")
            
            return task_result
            
        except Exception as e:
            self.logger.error(f"Error executing QA task: {e}", exc_info=True)
            self.qa_logger.log_error(e, {"task_definition": task_definition})
            raise
    
    def _create_qa_task_from_definition(self, task_def: Dict[str, Any]) -> QATask:
        """Create QA task from definition dictionary"""
        qa_task = QATask(
            name=task_def.get("name", "Test Task"),
            description=task_def.get("description", ""),
            app_under_test=task_def.get("app_under_test", ""),
            environment=task_def.get("environment", {}),
            device_requirements=task_def.get("device_requirements", {}),
            test_data=task_def.get("test_data", {})
        )
        
        # Add test goals
        for goal_def in task_def.get("test_goals", []):
            test_goal = TestGoal(
                title=goal_def.get("title", "Test Goal"),
                description=goal_def.get("description", ""),
                app_name=goal_def.get("app_name", task_def.get("app_under_test", "")),
                test_type=goal_def.get("test_type", "functional"),
                priority=Priority[goal_def.get("priority", "MEDIUM")],
                estimated_duration=goal_def.get("estimated_duration"),
                tags=goal_def.get("tags", []),
                prerequisites=goal_def.get("prerequisites", [])
            )
            
            qa_task.add_test_goal(test_goal)
        
        return qa_task
    
    async def run_predefined_tests(self) -> List[TestResult]:
        """Run a set of predefined tests"""
        predefined_tests = [
            self._create_wifi_settings_test(),
            self._create_app_navigation_test(),
            self._create_form_filling_test()
        ]
        
        results = []
        for test_definition in predefined_tests:
            try:
                result = await self.run_qa_task(test_definition)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Predefined test failed: {e}")
        
        return results
    
    def _create_wifi_settings_test(self) -> Dict[str, Any]:
        """Create WiFi settings test definition"""
        return {
            "name": "WiFi Settings Toggle Test",
            "description": "Test turning WiFi on and off through Android settings",
            "app_under_test": "com.android.settings",
            "test_goals": [
                {
                    "title": "Navigate to WiFi Settings",
                    "description": "Navigate to WiFi settings page in Android Settings app",
                    "test_type": "navigation",
                    "priority": "high"
                },
                {
                    "title": "Toggle WiFi State",
                    "description": "Turn WiFi off and then back on, verifying state changes",
                    "test_type": "functional",
                    "priority": "high"
                }
            ],
            "environment": {
                "android_version": "13",
                "device_type": "emulator"
            }
        }
    
    def _create_app_navigation_test(self) -> Dict[str, Any]:
        """Create app navigation test definition"""
        return {
            "name": "App Navigation Test",
            "description": "Test basic navigation patterns in a sample app",
            "app_under_test": "com.example.sampleapp",
            "test_goals": [
                {
                    "title": "Launch App",
                    "description": "Successfully launch the target application",
                    "test_type": "functional",
                    "priority": "critical"
                },
                {
                    "title": "Navigate Main Menu",
                    "description": "Navigate through main menu options",
                    "test_type": "ui",
                    "priority": "high"
                },
                {
                    "title": "Return to Home",
                    "description": "Successfully return to home screen using back navigation",
                    "test_type": "navigation",
                    "priority": "medium"
                }
            ]
        }
    
    def _create_form_filling_test(self) -> Dict[str, Any]:
        """Create form filling test definition"""
        return {
            "name": "Form Input Test",
            "description": "Test form filling and submission functionality",
            "app_under_test": "com.example.formapp",
            "test_goals": [
                {
                    "title": "Fill User Registration Form",
                    "description": "Fill out a user registration form with valid data",
                    "test_type": "functional",
                    "priority": "high"
                },
                {
                    "title": "Submit Form",
                    "description": "Submit the completed form and verify success",
                    "test_type": "integration",
                    "priority": "high"
                },
                {
                    "title": "Validate Error Handling",
                    "description": "Test form validation with invalid inputs",
                    "test_type": "functional",
                    "priority": "medium"
                }
            ],
            "test_data": {
                "valid_user": {
                    "name": "Test User",
                    "email": "test@example.com",
                    "phone": "555-1234"
                },
                "invalid_user": {
                    "name": "",
                    "email": "invalid-email",
                    "phone": "abc"
                }
            }
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        coordinator_status = await self.coordinator.get_system_status()
        
        return {
            "application": "QualGent Multi-Agent QA System",
            "version": "1.0.0",
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "coordinator": coordinator_status,
            "android_integration": self.android_integration.get_performance_metrics(),
            "evaluation_summary": self.evaluation_engine.get_evaluation_summary()
        }
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down QualGent application")
        
        try:
            # Disconnect from Android
            await self.android_integration.disconnect()
            
            # Close logging
            self.qa_logger.close()
            
            self.logger.info("QualGent application shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

async def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="QualGent Multi-Agent QA System")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--task", help="Task definition JSON file")
    parser.add_argument("--predefined", action="store_true", help="Run predefined tests")
    parser.add_argument("--status", action="store_true", help="Show system status")
    
    args = parser.parse_args()
    
    # Initialize application
    app = QualGentApp(config_file=args.config)
    
    try:
        if args.status:
            # Show system status
            status = await app.get_system_status()
            print(json.dumps(status, indent=2))
            
        elif args.task:
            # Run specific task from file
            with open(args.task, 'r') as f:
                task_definition = json.load(f)
            
            result = await app.run_qa_task(task_definition)
            print(f"Task completed with status: {result.overall_status.value}")
            
        elif args.predefined:
            # Run predefined tests
            results = await app.run_predefined_tests()
            print(f"Completed {len(results)} predefined tests")
            
        else:
            # Interactive mode - run WiFi test as example
            print("QualGent Multi-Agent QA System")
            print("Running example WiFi settings test...")
            
            wifi_test = app._create_wifi_settings_test()
            result = await app.run_qa_task(wifi_test)
            
            print(f"Test completed with status: {result.overall_status.value}")
            print(f"Success rate: {result.success_rate:.1%}")
            print(f"Total duration: {result.total_duration_ms / 1000:.1f} seconds")
            
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Application error: {e}")
        return 1
    finally:
        await app.shutdown()
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 