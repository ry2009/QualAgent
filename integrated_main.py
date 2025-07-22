#!/usr/bin/env python3
"""
Integrated Agent-S + QualGent Multi-Agent QA System
Main application demonstrating the combined framework
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

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "Agent-S"))
sys.path.insert(0, str(Path(__file__).parent / "agent_s_integration"))

from src.core.android_integration import AndroidWorldIntegration
from src.core.logging import QALogger
from src.models.task import QATask, TestGoal, Priority
from src.models.result import TestResult
from agent_s_integration.core.integrated_coordinator import IntegratedQACoordinator

class IntegratedQualGentApp:
    """Main application class for integrated Agent-S + QualGent system"""
    
    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        
        # Initialize QualGent components
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
        
        # Initialize integrated coordinator (Agent-S + QualGent)
        self.coordinator = IntegratedQACoordinator(
            config=self.config,
            android_integration=self.android_integration,
            qa_logger=self.qa_logger
        )
        
        self.logger.info("Integrated Agent-S + QualGent application initialized")
    
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
            
            # Agent-S + QualGent agent settings
            "planner_model": "gemini-1.5-flash",
            "planner_provider": "google",
            "planner_temperature": 0.1,
            "planner_max_tokens": 2000,
            
            "executor_model": "gemini-1.5-flash",
            "executor_provider": "google", 
            "executor_temperature": 0.1,
            "executor_max_tokens": 1500,
            
            "verifier_model": "gemini-1.5-flash",
            "verifier_provider": "google",
            "verifier_temperature": 0.1,
            "verifier_max_tokens": 1500,
            
            "supervisor_model": "gemini-1.5-flash",
            "supervisor_provider": "google",
            "supervisor_temperature": 0.2,
            "supervisor_max_tokens": 2500,
            
            # API keys
            "planner_api_key": os.getenv("GCP_API_KEY"),
            "executor_api_key": os.getenv("GCP_API_KEY"),
            "verifier_api_key": os.getenv("GCP_API_KEY"),
            "supervisor_api_key": os.getenv("GCP_API_KEY"),
            
            # Coordination settings
            "max_retries": 3,
            "enable_replanning": True,
            "enable_supervision": True,
            "continue_on_failure": True
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                # Handle nested config structure
                if "agents" in file_config:
                    for agent_type, agent_config in file_config["agents"].items():
                        for key, value in agent_config.items():
                            default_config[f"{agent_type}_{key}"] = value
                
                # Handle other config sections
                for section, section_config in file_config.items():
                    if section != "agents" and isinstance(section_config, dict):
                        default_config.update(section_config)
                    elif not isinstance(section_config, dict):
                        default_config[section] = section_config
                        
            except Exception as e:
                print(f"Warning: Could not load config file {config_file}: {e}")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup application logging"""
        logger = logging.getLogger("integrated_qualgent_app")
        logger.setLevel(getattr(logging, self.config.get("log_level", "INFO")))
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    async def run_qa_task(self, task_definition: Dict[str, Any]) -> TestResult:
        """Run a complete QA task using integrated Agent-S + QualGent system"""
        
        self.logger.info(f"Starting integrated QA task: {task_definition.get('name', 'Unnamed Task')}")
        
        try:
            # Create QA task from definition
            qa_task = self._create_qa_task_from_definition(task_definition)
            
            # Log task start
            self.qa_logger.log_task_start(qa_task)
            
            # Execute task with integrated coordinator
            task_result = await self.coordinator.execute_qa_task(qa_task)
            
            # Log task completion
            self.qa_logger.log_task_completion(qa_task, task_result)
            
            self.logger.info(f"Integrated QA task completed. Status: {task_result.overall_status}")
            
            return task_result
            
        except Exception as e:
            self.logger.error(f"Error executing integrated QA task: {e}", exc_info=True)
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
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get integrated system status"""
        coordinator_status = await self.coordinator.get_system_status()
        
        return {
            "application": "Integrated Agent-S + QualGent Multi-Agent QA System",
            "version": "1.0.0",
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "framework_integration": "Agent-S + QualGent",
            "coordinator": coordinator_status,
            "android_integration": self.android_integration.get_performance_metrics()
        }
    
    def _create_demo_wifi_test(self) -> Dict[str, Any]:
        """Create demo WiFi test for integrated system"""
        return {
            "name": "Agent-S + QualGent WiFi Integration Test",
            "description": "Demonstration of integrated Agent-S and QualGent system testing WiFi functionality",
            "app_under_test": "com.android.settings",
            "test_goals": [
                {
                    "title": "Navigate to WiFi Settings (Agent-S + QualGent)",
                    "description": "Use Agent-S Manager for planning and QualGent for verification",
                    "test_type": "navigation",
                    "priority": "HIGH"
                },
                {
                    "title": "Toggle WiFi State (Integrated Execution)",
                    "description": "Use Agent-S Worker with QualGent verification and supervision",
                    "test_type": "functional",
                    "priority": "CRITICAL"
                }
            ],
            "environment": {
                "android_version": "13",
                "device_type": "emulator",
                "framework": "Agent-S + QualGent"
            }
        }
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down integrated Agent-S + QualGent application")
        
        try:
            await self.android_integration.disconnect()
            self.qa_logger.close()
            self.logger.info("Integrated application shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

async def main():
    """Main application entry point for integrated system"""
    parser = argparse.ArgumentParser(description="Integrated Agent-S + QualGent Multi-Agent QA System")
    parser.add_argument("--config", help="Configuration file path", default="config/gemini_config.json")
    parser.add_argument("--task", help="Task definition JSON file")
    parser.add_argument("--status", action="store_true", help="Show integrated system status")
    parser.add_argument("--demo", action="store_true", help="Run demonstration test")
    
    args = parser.parse_args()
    
    # Initialize integrated application
    app = IntegratedQualGentApp(config_file=args.config)
    
    try:
        if args.status:
            # Show integrated system status
            status = await app.get_system_status()
            print(json.dumps(status, indent=2))
            
        elif args.task:
            # Run specific task from file
            with open(args.task, 'r') as f:
                task_definition = json.load(f)
            
            result = await app.run_qa_task(task_definition)
            print(f"Integrated task completed with status: {result.overall_status}")
            
        elif args.demo:
            # Run demonstration test
            print("Integrated Agent-S + QualGent Multi-Agent QA System")
            print("Running demonstration test...")
            
            demo_test = app._create_demo_wifi_test()
            result = await app.run_qa_task(demo_test)
            
            print(f"Demo test completed with status: {result.overall_status}")
            print(f"Success rate: {result.success_rate:.1%}")
            print("Integration demonstration complete!")
            
        else:
            print("Integrated Agent-S + QualGent Multi-Agent QA System")
            print("Use --status, --task <file>, or --demo to run the system")
            
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Integrated application error: {e}")
        return 1
    finally:
        await app.shutdown()
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 