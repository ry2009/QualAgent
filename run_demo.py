#!/usr/bin/env python3
"""
QualGent Demo Script
Demonstrates the full capabilities of the multi-agent QA system
"""

import asyncio
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import QualGentApp

class QualGentDemo:
    """Demo class for showcasing QualGent capabilities"""
    
    def __init__(self):
        print("QualGent Multi-Agent QA System Demo")
        print("=" * 50)
        
        # Initialize with demo configuration
        self.app = QualGentApp()
        self.demo_results = []
        
    async def run_complete_demo(self):
        """Run complete demonstration of all features"""
        try:
            print("\n Demo Overview:")
            print("1. System Status Check")
            print("2. WiFi Settings Test (AndroidWorld)")
            print("3. Email Search Test (Complex UI)")
            print("4. Performance Evaluation")
            print("5. Results Summary")
            
            # Step 1: System Status
            await self._demo_system_status()
            
            # Step 2: WiFi Test
            await self._demo_wifi_test()
            
            # Step 3: Email Test
            await self._demo_email_test()
            
            # Step 4: Performance Analysis
            await self._demo_performance_analysis()
            
            # Step 5: Summary
            await self._demo_summary()
            
        except KeyboardInterrupt:
            print("\n  Demo interrupted by user")
        except Exception as e:
            print(f"\n Demo error: {e}")
        finally:
            await self.app.shutdown()
    
    async def _demo_system_status(self):
        """Demonstrate system status functionality"""
        print("\n" + "="*50)
        print("🔍 STEP 1: System Status Check")
        print("="*50)
        
        status = await self.app.get_system_status()
        
        print(f" Application: {status['application']}")
        print(f" Version: {status['version']}")
        print(f" Status: {status['status']}")
        print(f" Android Connected: {status['android_integration']['is_connected']}")
        
        # Show agent status
        print("\nAgent Status:")
        for agent_name, is_active in status['coordinator']['agent_status'].items():
            status_icon = "ACTIVE" if is_active else "INACTIVE"
            print(f"  {status_icon} {agent_name.title()} Agent: {'Active' if is_active else 'Inactive'}")
        
        print("\n System status check completed")
        await asyncio.sleep(2)
    
    async def _demo_wifi_test(self):
        """Demonstrate WiFi settings test"""
        print("\n" + "="*50)
        print("📱 STEP 2: WiFi Settings Test")
        print("="*50)
        
        wifi_task = {
            "name": "Demo WiFi Settings Test",
            "description": "Demonstration of WiFi toggle functionality testing",
            "app_under_test": "com.android.settings",
            "test_goals": [
                {
                    "title": "Navigate to WiFi Settings",
                    "description": "Navigate from main settings to WiFi settings page",
                    "app_name": "com.android.settings",
                    "test_type": "navigation",
                    "priority": "high"
                },
                {
                    "title": "Toggle WiFi State",
                    "description": "Toggle WiFi on/off and verify state changes",
                    "app_name": "com.android.settings",
                    "test_type": "functional",
                    "priority": "critical"
                }
            ]
        }
        
        print(" Starting WiFi test execution...")
        start_time = time.time()
        
        try:
            result = await self.app.run_qa_task(wifi_task)
            execution_time = time.time() - start_time
            
            print(f"\n WiFi Test Results:")
            print(f"  Status: {result.overall_status.value}")
            print(f"  Duration: {execution_time:.1f} seconds")
            print(f"  Success Rate: {result.success_rate:.1%}")
            print(f"  Bugs Detected: {len(result.bugs_detected)}")
            print(f"  Replanning Count: {result.replanning_count}")
            
            self.demo_results.append({
                "test": "WiFi Settings",
                "status": result.overall_status.value,
                "duration": execution_time,
                "success_rate": result.success_rate,
                "bugs": len(result.bugs_detected)
            })
            
        except Exception as e:
            print(f"WiFi test failed: {e}")
            self.demo_results.append({
                "test": "WiFi Settings",
                "status": "failed",
                "error": str(e)
            })
        
        await asyncio.sleep(2)
    
    async def _demo_email_test(self):
        """Demonstrate email search test"""
        print("\n" + "="*50)
        print("STEP 3: Email Search Test")
        print("="*50)
        
        email_task = {
            "name": "Demo Email Search Test",
            "description": "Demonstration of email search and interaction functionality",
            "app_under_test": "com.google.android.gm",
            "test_goals": [
                {
                    "title": "Launch Gmail Application",
                    "description": "Successfully launch Gmail and verify inbox",
                    "app_name": "com.google.android.gm",
                    "test_type": "functional",
                    "priority": "critical"
                },
                {
                    "title": "Perform Email Search",
                    "description": "Execute search for specific email content",
                    "app_name": "com.google.android.gm",
                    "test_type": "functional",
                    "priority": "high"
                },
                {
                    "title": "Verify Search Results",
                    "description": "Validate search results are relevant and accessible",
                    "app_name": "com.google.android.gm",
                    "test_type": "verification",
                    "priority": "high"
                }
            ],
            "test_data": {
                "search_query": "important meeting"
            }
        }
        
        print(" Starting email test execution...")
        start_time = time.time()
        
        try:
            result = await self.app.run_qa_task(email_task)
            execution_time = time.time() - start_time
            
            print(f"\n Email Test Results:")
            print(f"  Status: {result.overall_status.value}")
            print(f"  Duration: {execution_time:.1f} seconds")
            print(f"  Success Rate: {result.success_rate:.1%}")
            print(f"  Search Operations: {result.total_subgoals}")
            print(f"  Agent Decisions: {len(result.agent_decisions)}")
            
            self.demo_results.append({
                "test": "Email Search",
                "status": result.overall_status.value,
                "duration": execution_time,
                "success_rate": result.success_rate,
                "decisions": len(result.agent_decisions)
            })
            
        except Exception as e:
            print(f" Email test failed: {e}")
            self.demo_results.append({
                "test": "Email Search",
                "status": "failed",
                "error": str(e)
            })
        
        await asyncio.sleep(2)
    
    async def _demo_performance_analysis(self):
        """Demonstrate performance evaluation capabilities"""
        print("\n" + "="*50)
        print("📈 STEP 4: Performance Evaluation")
        print("="*50)
        
        # Get evaluation summary
        eval_summary = self.app.evaluation_engine.get_evaluation_summary()
        
        print("📊 Evaluation Metrics:")
        if eval_summary.get("total_evaluations", 0) > 0:
            print(f"  Total Evaluations: {eval_summary['total_evaluations']}")
            print(f"  Latest Score: {eval_summary['latest_score']:.3f}")
            print(f"  Average Score: {eval_summary['average_score']:.3f}")
            print(f"  Best Score: {eval_summary['best_score']:.3f}")
            print(f"  Trend: {eval_summary['score_trend']}")
        else:
            print("  No evaluations available yet")
        
        # Show agent performance
        print("\nAgent Performance:")
        coordinator_status = await self.app.coordinator.get_system_status()
        
        for agent_name, agent in self.app.coordinator.agents.items():
            metrics = agent.get_performance_metrics()
            print(f"  🔹 {agent_name.title()} Agent:")
            print(f"    • Decisions: {metrics['total_decisions']}")
            print(f"    • Success Rate: {metrics['success_rate']:.1f}%")
            print(f"    • Avg Response: {metrics['average_response_time_ms']:.0f}ms")
        
        await asyncio.sleep(2)
    
    async def _demo_summary(self):
        """Show final demo summary"""
        print("\n" + "="*50)
        print("STEP 5: Demo Summary")
        print("="*50)
        
        print("Test Results Summary:")
        for i, result in enumerate(self.demo_results, 1):
            status_icon = "PASS" if result.get("status") == "success" else "FAIL"
            print(f"  {i}. {status_icon} {result['test']}")
            
            if "error" in result:
                print(f"     Error: {result['error']}")
            else:
                print(f"     Status: {result.get('status', 'unknown')}")
                if "duration" in result:
                    print(f"     Duration: {result['duration']:.1f}s")
                if "success_rate" in result:
                    print(f"     Success: {result['success_rate']:.1%}")
        
        print("\nSystem Capabilities Demonstrated:")
        print("  Multi-agent coordination")
        print("  Android UI automation")
        print("  Dynamic replanning")
        print("  Bug detection")
        print("  Performance evaluation")
        print("  Comprehensive logging")
        print("  Visual trace analysis")
        print("  Supervision feedback")
        
        print("\n Generated Artifacts:")
        print(f"  Logs: {self.app.qa_logger.session_dir}")
        print(f"  Reports: evaluation_report_*.json")
        print(f"  Session: session_summary.json")
        
        print("\n Demo completed successfully!")
        print("🔗 Check the logs directory for detailed results")

async def run_interactive_demo():
    """Run interactive demo with user choices"""
    print("🎮 QualGent Interactive Demo")
    print("Choose demo mode:")
    print("1. Full Demo (all features)")
    print("2. Quick Demo (essential features)")
    print("3. Custom Test (load your task)")
    print("4. System Status Only")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        demo = QualGentDemo()
        
        if choice == "1":
            await demo.run_complete_demo()
        elif choice == "2":
            await demo._demo_system_status()
            await demo._demo_wifi_test()
            await demo._demo_summary()
        elif choice == "3":
            task_file = input("Enter task file path: ").strip()
            if Path(task_file).exists():
                with open(task_file, 'r') as f:
                    task_def = json.load(f)
                result = await demo.app.run_qa_task(task_def)
                print(f"Custom test completed: {result.overall_status.value}")
            else:
                print("Task file not found")
        elif choice == "4":
            await demo._demo_system_status()
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\nDemo cancelled by user")
    except Exception as e:
        print(f"Demo error: {e}")

def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        asyncio.run(run_interactive_demo())
    else:
        demo = QualGentDemo()
        asyncio.run(demo.run_complete_demo())

if __name__ == "__main__":
    main() 