#!/usr/bin/env python3
"""
Test script to verify QualGent works with Gemini API
Tests basic LLM integration without requiring Android emulator
"""

import asyncio
import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.llm_client import LLMClient, LLMMessage
from src.agents.base import AgentConfig
from src.agents.planner import PlannerAgent
from src.models.task import QATask, TestGoal
from src.models.ui_state import UIState, UIElement

class GeminiIntegrationTest:
    """Test Gemini API integration"""
    
    def __init__(self):
        self.api_key = os.getenv("GCP_API_KEY")
        self.test_results = []
        
    async def run_all_tests(self):
        """Run all integration tests"""
        print("QualGent Gemini Integration Test")
        print("=" * 40)
        
        # Test 1: Basic LLM Client
        await self.test_llm_client()
        
        # Test 2: Agent Configuration
        await self.test_agent_configuration()
        
        # Test 3: Planner Agent
        await self.test_planner_agent()
        
        # Test 4: Data Models
        await self.test_data_models()
        
        # Print results
        self.print_test_results()
        
        return all(result["passed"] for result in self.test_results)
    
    async def test_llm_client(self):
        """Test basic LLM client functionality"""
        print("\nTest 1: LLM Client Integration")
        print("-" * 30)
        
        try:
            # Initialize Gemini client
            client = LLMClient(
                provider="google",
                model="gemini-pro",
                api_key=self.api_key
            )
            
            # Test basic message generation
            messages = [
                LLMMessage(role="user", content="Respond with exactly 'TEST_SUCCESS' and nothing else.")
            ]
            
            response = await client.generate_response(messages, temperature=0.0, max_tokens=10)
            
            # Validate response
            if response.content and "TEST_SUCCESS" in response.content:
                print("[PASS] Basic LLM response generation")
                print(f"[INFO] Response: {response.content.strip()}")
                print(f"[INFO] Tokens used: {response.tokens_used}")
                print(f"[INFO] Response time: {response.response_time_ms}ms")
                
                self.test_results.append({
                    "test": "LLM Client Basic",
                    "passed": True,
                    "details": f"Response in {response.response_time_ms}ms"
                })
            else:
                print(f"[FAIL] Unexpected response: {response.content}")
                self.test_results.append({
                    "test": "LLM Client Basic",
                    "passed": False,
                    "error": f"Unexpected response: {response.content}"
                })
                
        except Exception as e:
            print(f"[FAIL] LLM client error: {e}")
            self.test_results.append({
                "test": "LLM Client Basic",
                "passed": False,
                "error": str(e)
            })
    
    async def test_agent_configuration(self):
        """Test agent configuration with Gemini"""
        print("\nTest 2: Agent Configuration")
        print("-" * 30)
        
        try:
            # Create agent config
            config = AgentConfig(
                model_name="gemini-pro",
                provider="google",
                api_key=self.api_key,
                temperature=0.1,
                max_tokens=1000
            )
            
            print("[PASS] Agent configuration created")
            print(f"[INFO] Model: {config.model_name}")
            print(f"[INFO] Provider: {config.provider}")
            print(f"[INFO] Temperature: {config.temperature}")
            
            self.test_results.append({
                "test": "Agent Configuration",
                "passed": True,
                "details": f"Config for {config.model_name}"
            })
            
        except Exception as e:
            print(f"[FAIL] Agent configuration error: {e}")
            self.test_results.append({
                "test": "Agent Configuration",
                "passed": False,
                "error": str(e)
            })
    
    async def test_planner_agent(self):
        """Test planner agent with Gemini"""
        print("\nTest 3: Planner Agent")
        print("-" * 30)
        
        try:
            # Create planner agent
            config = AgentConfig(
                model_name="gemini-pro",
                provider="google",
                api_key=self.api_key,
                temperature=0.1,
                max_tokens=1500
            )
            
            planner = PlannerAgent(config)
            
            # Create test task
            qa_task = QATask(
                name="Test WiFi Settings",
                description="Test WiFi toggle functionality",
                app_under_test="com.android.settings"
            )
            
            test_goal = TestGoal(
                title="Toggle WiFi",
                description="Turn WiFi on and off",
                app_name="com.android.settings",
                test_type="functional"
            )
            
            # Activate agent
            planner.activate(qa_task, test_goal)
            print("[PASS] Planner agent activated")
            
            # Test planning decision (simplified)
            planning_context = {
                "request_type": "create_plan",
                "goal_description": "Toggle WiFi on and off in Android settings",
                "app_name": "com.android.settings",
                "test_type": "functional"
            }
            
            # Note: This would normally require UI state, but we'll test without it
            decision = await planner.make_decision(planning_context)
            
            if decision and decision.reasoning:
                print("[PASS] Planning decision generated")
                print(f"[INFO] Decision type: {decision.decision_type.value}")
                print(f"[INFO] Confidence: {decision.confidence_score:.2f}")
                print(f"[INFO] Reasoning: {decision.reasoning[:100]}...")
                
                self.test_results.append({
                    "test": "Planner Agent",
                    "passed": True,
                    "details": f"Confidence: {decision.confidence_score:.2f}"
                })
            else:
                print("[FAIL] No planning decision generated")
                self.test_results.append({
                    "test": "Planner Agent",
                    "passed": False,
                    "error": "No decision generated"
                })
                
        except Exception as e:
            print(f"[FAIL] Planner agent error: {e}")
            self.test_results.append({
                "test": "Planner Agent",
                "passed": False,
                "error": str(e)
            })
    
    async def test_data_models(self):
        """Test data model functionality"""
        print("\nTest 4: Data Models")
        print("-" * 30)
        
        try:
            # Test QA Task model
            qa_task = QATask(
                name="Test Task",
                description="Test task description",
                app_under_test="com.example.app"
            )
            
            test_goal = TestGoal(
                title="Test Goal",
                description="Test goal description",
                test_type="functional"
            )
            
            qa_task.add_test_goal(test_goal)
            
            # Test UI State model
            ui_element = UIElement(
                resource_id="com.android.settings:id/switch_widget",
                class_name="android.widget.Switch",
                text="WiFi",
                bounds=(100, 200, 300, 250),
                is_clickable=True,
                is_enabled=True
            )
            
            ui_state = UIState(
                screen_width=1080,
                screen_height=1920,
                current_app="com.android.settings"
            )
            
            ui_state.add_element(ui_element)
            
            # Validate models
            assert qa_task.name == "Test Task"
            assert len(qa_task.test_goals) == 1
            assert test_goal.title == "Test Goal"
            assert ui_element.is_clickable == True
            assert len(ui_state.elements) == 1
            assert ui_state.hierarchy_hash is not None
            
            print("[PASS] Data models validation")
            print(f"[INFO] Task has {len(qa_task.test_goals)} goals")
            print(f"[INFO] UI state has {len(ui_state.elements)} elements")
            print(f"[INFO] UI hierarchy hash: {ui_state.hierarchy_hash[:8]}...")
            
            self.test_results.append({
                "test": "Data Models",
                "passed": True,
                "details": f"Models working correctly"
            })
            
        except Exception as e:
            print(f"[FAIL] Data models error: {e}")
            self.test_results.append({
                "test": "Data Models",
                "passed": False,
                "error": str(e)
            })
    
    def print_test_results(self):
        """Print final test results"""
        print("\n" + "=" * 40)
        print("Test Results Summary")
        print("=" * 40)
        
        passed_tests = sum(1 for result in self.test_results if result["passed"])
        total_tests = len(self.test_results)
        
        print(f"Tests passed: {passed_tests}/{total_tests}")
        
        for result in self.test_results:
            status = "[PASS]" if result["passed"] else "[FAIL]"
            print(f"{status} {result['test']}")
            
            if result["passed"] and "details" in result:
                print(f"       {result['details']}")
            elif not result["passed"] and "error" in result:
                print(f"       Error: {result['error']}")
        
        if passed_tests == total_tests:
            print("\nAll tests passed! Gemini integration is working.")
            print("\nNext steps:")
            print("1. Run validation: python scripts/validate_setup.py")
            print("2. Test with Android: python main.py --config config/gemini_config.json --status")
        else:
            print(f"\n{total_tests - passed_tests} test(s) failed.")
            print("Check error messages above and verify Gemini API key.")

async def main():
    """Main test runner"""
    print("Starting Gemini integration tests...")
    
    # Ensure environment variable is set for Google API
    if not os.getenv("GCP_API_KEY"):
        print("Warning: GCP_API_KEY environment variable not set")
    
    tester = GeminiIntegrationTest()
    success = await tester.run_all_tests()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 