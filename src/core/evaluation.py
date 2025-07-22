import json
import statistics
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging

from ..models.task import QATask, TestGoal, SubGoal, TaskStatus
from ..models.result import TestResult, AgentDecision, ExecutionResult, ResultStatus
from ..models.ui_state import UIState

class EvaluationMetrics:
    """Container for evaluation metrics"""
    def __init__(self):
        # Accuracy metrics
        self.success_rate: float = 0.0
        self.bug_detection_accuracy: float = 0.0
        self.false_positive_rate: float = 0.0
        self.false_negative_rate: float = 0.0
        
        # Performance metrics
        self.average_execution_time: float = 0.0
        self.throughput: float = 0.0  # subgoals per minute
        self.efficiency_score: float = 0.0
        
        # Reliability metrics
        self.consistency_score: float = 0.0
        self.error_recovery_rate: float = 0.0
        self.replanning_frequency: float = 0.0
        
        # Quality metrics
        self.test_coverage_score: float = 0.0
        self.agent_confidence_average: float = 0.0
        self.supervision_quality_score: float = 0.0

class EvaluationEngine:
    """Engine for evaluating QA system performance and generating reports"""
    
    def __init__(self, 
                 baseline_metrics: Optional[Dict[str, float]] = None,
                 logger: Optional[logging.Logger] = None):
        
        self.logger = logger or logging.getLogger("evaluation_engine")
        
        # Baseline metrics for comparison
        self.baseline_metrics = baseline_metrics or {
            "success_rate": 0.85,
            "bug_detection_accuracy": 0.90,
            "false_positive_rate": 0.10,
            "average_execution_time": 5000,  # ms
            "efficiency_score": 0.80,
            "consistency_score": 0.85
        }
        
        # Evaluation history for trend analysis
        self.evaluation_history: List[Dict[str, Any]] = []
        
        # Weights for composite scores
        self.metric_weights = {
            "accuracy": 0.35,
            "performance": 0.25,
            "reliability": 0.25,
            "quality": 0.15
        }
        
    async def evaluate_task_execution(self, 
                                    task: QATask, 
                                    test_results: List[TestResult],
                                    agent_decisions: List[AgentDecision]) -> Dict[str, Any]:
        """Evaluate a complete task execution"""
        self.logger.info(f"Evaluating task execution: {task.name}")
        
        # Calculate metrics
        metrics = EvaluationMetrics()
        await self._calculate_accuracy_metrics(metrics, task, test_results)
        await self._calculate_performance_metrics(metrics, task, test_results, agent_decisions)
        await self._calculate_reliability_metrics(metrics, task, test_results)
        await self._calculate_quality_metrics(metrics, task, test_results, agent_decisions)
        
        # Generate evaluation report
        evaluation_report = await self._generate_evaluation_report(task, metrics, test_results, agent_decisions)
        
        # Store in history
        self.evaluation_history.append({
            "timestamp": datetime.now().isoformat(),
            "task_id": task.id,
            "task_name": task.name,
            "metrics": self._metrics_to_dict(metrics),
            "overall_score": evaluation_report["overall_score"]
        })
        
        return evaluation_report
    
    async def _calculate_accuracy_metrics(self, 
                                        metrics: EvaluationMetrics, 
                                        task: QATask, 
                                        test_results: List[TestResult]):
        """Calculate accuracy-related metrics"""
        total_subgoals = 0
        successful_subgoals = 0
        total_bugs_detected = 0
        false_positives = 0
        false_negatives = 0
        
        for result in test_results:
            total_subgoals += result.total_subgoals
            successful_subgoals += result.successful_subgoals
            total_bugs_detected += len(result.bugs_detected)
            false_positives += result.false_positives
            false_negatives += result.false_negatives
        
        # Success rate
        metrics.success_rate = successful_subgoals / max(total_subgoals, 1)
        
        # Bug detection accuracy
        if total_bugs_detected > 0:
            true_positives = total_bugs_detected - false_positives
            metrics.bug_detection_accuracy = true_positives / (true_positives + false_positives + false_negatives)
            metrics.false_positive_rate = false_positives / total_bugs_detected
            metrics.false_negative_rate = false_negatives / (true_positives + false_negatives)
        else:
            metrics.bug_detection_accuracy = 1.0
            metrics.false_positive_rate = 0.0
            metrics.false_negative_rate = 0.0
    
    async def _calculate_performance_metrics(self, 
                                           metrics: EvaluationMetrics, 
                                           task: QATask, 
                                           test_results: List[TestResult],
                                           agent_decisions: List[AgentDecision]):
        """Calculate performance-related metrics"""
        total_execution_time = 0
        total_subgoals = 0
        total_api_calls = 0
        total_tokens_used = 0
        
        for result in test_results:
            total_execution_time += result.total_duration_ms
            total_subgoals += result.total_subgoals
            total_api_calls += result.total_api_calls
            total_tokens_used += result.total_tokens_used
        
        # Average execution time per subgoal
        metrics.average_execution_time = total_execution_time / max(total_subgoals, 1)
        
        # Throughput (subgoals per minute)
        if total_execution_time > 0:
            metrics.throughput = (total_subgoals * 60 * 1000) / total_execution_time
        
        # Efficiency score (consider API usage, tokens, time)
        api_efficiency = total_subgoals / max(total_api_calls, 1)
        token_efficiency = total_subgoals / max(total_tokens_used / 1000, 1)  # per 1k tokens
        time_efficiency = max(0, 1 - (metrics.average_execution_time / 10000))  # normalized
        
        metrics.efficiency_score = (api_efficiency + token_efficiency + time_efficiency) / 3
    
    async def _calculate_reliability_metrics(self, 
                                           metrics: EvaluationMetrics, 
                                           task: QATask, 
                                           test_results: List[TestResult]):
        """Calculate reliability-related metrics"""
        execution_times = []
        success_rates = []
        total_errors = 0
        total_recoveries = 0
        total_replannings = 0
        total_subgoals = 0
        
        for result in test_results:
            if result.execution_results:
                times = [er.execution_time_ms for er in result.execution_results if er.execution_time_ms > 0]
                execution_times.extend(times)
            
            if result.total_subgoals > 0:
                success_rates.append(result.successful_subgoals / result.total_subgoals)
            
            total_errors += result.failed_subgoals
            total_recoveries += result.error_recovery_count
            total_replannings += result.replanning_count
            total_subgoals += result.total_subgoals
        
        # Consistency score based on execution time variance
        if len(execution_times) > 1:
            time_variance = statistics.variance(execution_times)
            time_mean = statistics.mean(execution_times)
            cv = (time_variance ** 0.5) / time_mean if time_mean > 0 else 1
            metrics.consistency_score = max(0, 1 - cv)
        else:
            metrics.consistency_score = 1.0
        
        # Error recovery rate
        metrics.error_recovery_rate = total_recoveries / max(total_errors, 1)
        
        # Replanning frequency
        metrics.replanning_frequency = total_replannings / max(total_subgoals, 1)
    
    async def _calculate_quality_metrics(self, 
                                       metrics: EvaluationMetrics, 
                                       task: QATask, 
                                       test_results: List[TestResult],
                                       agent_decisions: List[AgentDecision]):
        """Calculate quality-related metrics"""
        # Test coverage analysis
        action_types_covered = set()
        element_types_interacted = set()
        
        for result in test_results:
            for execution_result in result.execution_results:
                action_types_covered.add(execution_result.action_type)
        
        # Coverage score (simplified)
        total_action_types = 5  # touch, type, scroll, wait, verify
        metrics.test_coverage_score = len(action_types_covered) / total_action_types
        
        # Agent confidence average
        confidence_scores = [decision.confidence_score for decision in agent_decisions if decision.confidence_score > 0]
        metrics.agent_confidence_average = statistics.mean(confidence_scores) if confidence_scores else 0.0
        
        # Supervision quality score (based on feedback completeness and usefulness)
        supervision_decisions = [d for d in agent_decisions if d.agent_type.value == "supervisor"]
        if supervision_decisions:
            feedback_scores = []
            for decision in supervision_decisions:
                feedback = decision.output_data
                score = 0.0
                
                # Check completeness of feedback
                if feedback.get("improvement_suggestions"):
                    score += 0.3
                if feedback.get("prompt_improvements"):
                    score += 0.3
                if feedback.get("quality_score", 0) > 0:
                    score += 0.2
                if feedback.get("risk_assessment"):
                    score += 0.2
                
                feedback_scores.append(score)
            
            metrics.supervision_quality_score = statistics.mean(feedback_scores)
        else:
            metrics.supervision_quality_score = 0.0
    
    async def _generate_evaluation_report(self, 
                                        task: QATask, 
                                        metrics: EvaluationMetrics,
                                        test_results: List[TestResult],
                                        agent_decisions: List[AgentDecision]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        # Calculate category scores
        accuracy_score = self._calculate_accuracy_score(metrics)
        performance_score = self._calculate_performance_score(metrics)
        reliability_score = self._calculate_reliability_score(metrics)
        quality_score = self._calculate_quality_score(metrics)
        
        # Calculate overall score
        overall_score = (
            self.metric_weights["accuracy"] * accuracy_score +
            self.metric_weights["performance"] * performance_score +
            self.metric_weights["reliability"] * reliability_score +
            self.metric_weights["quality"] * quality_score
        )
        
        # Generate insights and recommendations
        insights = await self._generate_insights(metrics, test_results)
        recommendations = await self._generate_recommendations(metrics, test_results)
        
        # Baseline comparison
        baseline_comparison = self._compare_to_baseline(metrics)
        
        # Trend analysis
        trend_analysis = self._analyze_trends(task.id)
        
        report = {
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "task_id": task.id,
                "task_name": task.name,
                "evaluator_version": "1.0.0"
            },
            "overall_score": round(overall_score, 3),
            "category_scores": {
                "accuracy": round(accuracy_score, 3),
                "performance": round(performance_score, 3),
                "reliability": round(reliability_score, 3),
                "quality": round(quality_score, 3)
            },
            "detailed_metrics": self._metrics_to_dict(metrics),
            "baseline_comparison": baseline_comparison,
            "insights": insights,
            "recommendations": recommendations,
            "trend_analysis": trend_analysis,
            "task_summary": {
                "total_goals": len(task.test_goals),
                "completed_goals": len([g for g in task.test_goals if g.status == TaskStatus.COMPLETED]),
                "total_execution_time": sum(r.total_duration_ms for r in test_results),
                "total_subgoals": sum(r.total_subgoals for r in test_results),
                "bugs_detected": sum(len(r.bugs_detected) for r in test_results)
            },
            "agent_performance": self._analyze_agent_performance(agent_decisions)
        }
        
        return report
    
    def _calculate_accuracy_score(self, metrics: EvaluationMetrics) -> float:
        """Calculate accuracy category score"""
        success_weight = 0.5
        bug_detection_weight = 0.3
        false_positive_weight = 0.2
        
        success_score = metrics.success_rate
        bug_score = metrics.bug_detection_accuracy
        fp_score = max(0, 1 - metrics.false_positive_rate)
        
        return (
            success_weight * success_score +
            bug_detection_weight * bug_score +
            false_positive_weight * fp_score
        )
    
    def _calculate_performance_score(self, metrics: EvaluationMetrics) -> float:
        """Calculate performance category score"""
        # Normalize metrics to 0-1 scale
        time_score = max(0, 1 - (metrics.average_execution_time / 10000))  # 10s max
        throughput_score = min(1, metrics.throughput / 10)  # 10 subgoals/min max
        efficiency_score = metrics.efficiency_score
        
        return (time_score + throughput_score + efficiency_score) / 3
    
    def _calculate_reliability_score(self, metrics: EvaluationMetrics) -> float:
        """Calculate reliability category score"""
        consistency_weight = 0.4
        recovery_weight = 0.3
        replanning_weight = 0.3
        
        consistency_score = metrics.consistency_score
        recovery_score = metrics.error_recovery_rate
        replanning_score = max(0, 1 - metrics.replanning_frequency)  # Lower is better
        
        return (
            consistency_weight * consistency_score +
            recovery_weight * recovery_score +
            replanning_weight * replanning_score
        )
    
    def _calculate_quality_score(self, metrics: EvaluationMetrics) -> float:
        """Calculate quality category score"""
        coverage_weight = 0.4
        confidence_weight = 0.3
        supervision_weight = 0.3
        
        coverage_score = metrics.test_coverage_score
        confidence_score = metrics.agent_confidence_average
        supervision_score = metrics.supervision_quality_score
        
        return (
            coverage_weight * coverage_score +
            confidence_weight * confidence_score +
            supervision_weight * supervision_score
        )
    
    async def _generate_insights(self, 
                               metrics: EvaluationMetrics, 
                               test_results: List[TestResult]) -> List[str]:
        """Generate insights based on evaluation results"""
        insights = []
        
        # Success rate insights
        if metrics.success_rate < 0.8:
            insights.append(f"Success rate ({metrics.success_rate:.1%}) is below recommended threshold")
        elif metrics.success_rate > 0.95:
            insights.append("Excellent success rate indicates robust test execution")
        
        # Performance insights
        if metrics.average_execution_time > 5000:
            insights.append("Execution time is higher than optimal, consider optimization")
        
        if metrics.efficiency_score < 0.7:
            insights.append("Low efficiency score suggests room for resource optimization")
        
        # Reliability insights
        if metrics.consistency_score < 0.8:
            insights.append("High variance in execution times indicates inconsistent performance")
        
        if metrics.replanning_frequency > 0.2:
            insights.append("High replanning frequency suggests initial planning could be improved")
        
        # Bug detection insights
        if metrics.false_positive_rate > 0.15:
            insights.append("High false positive rate in bug detection needs attention")
        
        # Quality insights
        if metrics.test_coverage_score < 0.8:
            insights.append("Test coverage could be expanded to include more action types")
        
        if metrics.agent_confidence_average < 0.7:
            insights.append("Low average agent confidence suggests prompt improvements needed")
        
        return insights
    
    async def _generate_recommendations(self, 
                                      metrics: EvaluationMetrics, 
                                      test_results: List[TestResult]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Success rate recommendations
        if metrics.success_rate < 0.85:
            recommendations.append("Improve element locating strategies and retry mechanisms")
            recommendations.append("Review and enhance agent prompts for better decision making")
        
        # Performance recommendations
        if metrics.average_execution_time > 4000:
            recommendations.append("Optimize wait times and action execution speed")
            recommendations.append("Implement parallel processing where possible")
        
        if metrics.efficiency_score < 0.75:
            recommendations.append("Reduce API calls through better caching and batching")
            recommendations.append("Optimize token usage in LLM interactions")
        
        # Reliability recommendations
        if metrics.consistency_score < 0.8:
            recommendations.append("Implement more deterministic timing mechanisms")
            recommendations.append("Add better synchronization points")
        
        if metrics.error_recovery_rate < 0.7:
            recommendations.append("Enhance error recovery strategies")
            recommendations.append("Implement more robust fallback mechanisms")
        
        # Quality recommendations
        if metrics.test_coverage_score < 0.8:
            recommendations.append("Expand test scenarios to cover more interaction types")
            recommendations.append("Add comprehensive edge case testing")
        
        if metrics.bug_detection_accuracy < 0.9:
            recommendations.append("Refine bug detection algorithms")
            recommendations.append("Improve verification heuristics")
        
        # Supervision recommendations
        if metrics.supervision_quality_score < 0.7:
            recommendations.append("Enhance supervision prompts for more detailed feedback")
            recommendations.append("Implement more comprehensive analysis frameworks")
        
        return recommendations
    
    def _compare_to_baseline(self, metrics: EvaluationMetrics) -> Dict[str, Dict[str, float]]:
        """Compare metrics to baseline"""
        comparison = {}
        
        metric_mapping = {
            "success_rate": metrics.success_rate,
            "bug_detection_accuracy": metrics.bug_detection_accuracy,
            "false_positive_rate": metrics.false_positive_rate,
            "average_execution_time": metrics.average_execution_time,
            "efficiency_score": metrics.efficiency_score,
            "consistency_score": metrics.consistency_score
        }
        
        for metric_name, current_value in metric_mapping.items():
            baseline_value = self.baseline_metrics.get(metric_name, 0)
            
            if baseline_value > 0:
                change_percent = ((current_value - baseline_value) / baseline_value) * 100
                
                # For metrics where lower is better (execution time, false positive rate)
                if metric_name in ["average_execution_time", "false_positive_rate"]:
                    improvement = change_percent < 0
                else:
                    improvement = change_percent > 0
                
                comparison[metric_name] = {
                    "current": current_value,
                    "baseline": baseline_value,
                    "change_percent": round(change_percent, 2),
                    "improved": improvement
                }
        
        return comparison
    
    def _analyze_trends(self, task_id: str) -> Dict[str, Any]:
        """Analyze trends from evaluation history"""
        if len(self.evaluation_history) < 2:
            return {"message": "Insufficient data for trend analysis"}
        
        recent_evaluations = [
            eval_data for eval_data in self.evaluation_history[-10:]  # Last 10 evaluations
        ]
        
        # Extract trends for key metrics
        overall_scores = [eval_data["overall_score"] for eval_data in recent_evaluations]
        success_rates = [eval_data["metrics"]["success_rate"] for eval_data in recent_evaluations]
        
        trend_analysis = {
            "evaluation_count": len(recent_evaluations),
            "overall_score_trend": {
                "current": overall_scores[-1],
                "average": statistics.mean(overall_scores),
                "trend": "improving" if overall_scores[-1] > overall_scores[0] else "declining",
                "variance": statistics.variance(overall_scores) if len(overall_scores) > 1 else 0
            },
            "success_rate_trend": {
                "current": success_rates[-1],
                "average": statistics.mean(success_rates),
                "trend": "improving" if success_rates[-1] > success_rates[0] else "declining"
            }
        }
        
        return trend_analysis
    
    def _analyze_agent_performance(self, agent_decisions: List[AgentDecision]) -> Dict[str, Dict[str, Any]]:
        """Analyze individual agent performance"""
        agent_stats = {}
        
        for decision in agent_decisions:
            agent_name = decision.agent_type.value
            
            if agent_name not in agent_stats:
                agent_stats[agent_name] = {
                    "total_decisions": 0,
                    "confidence_scores": [],
                    "execution_times": [],
                    "decision_types": {}
                }
            
            stats = agent_stats[agent_name]
            stats["total_decisions"] += 1
            stats["confidence_scores"].append(decision.confidence_score)
            stats["execution_times"].append(decision.execution_time_ms)
            
            decision_type = decision.decision_type.value
            stats["decision_types"][decision_type] = stats["decision_types"].get(decision_type, 0) + 1
        
        # Calculate summary statistics
        for agent_name, stats in agent_stats.items():
            stats["average_confidence"] = statistics.mean(stats["confidence_scores"])
            stats["average_execution_time"] = statistics.mean(stats["execution_times"])
            stats["confidence_consistency"] = 1 - (statistics.stdev(stats["confidence_scores"]) if len(stats["confidence_scores"]) > 1 else 0)
        
        return agent_stats
    
    def _metrics_to_dict(self, metrics: EvaluationMetrics) -> Dict[str, float]:
        """Convert metrics object to dictionary"""
        return {
            "success_rate": round(metrics.success_rate, 4),
            "bug_detection_accuracy": round(metrics.bug_detection_accuracy, 4),
            "false_positive_rate": round(metrics.false_positive_rate, 4),
            "false_negative_rate": round(metrics.false_negative_rate, 4),
            "average_execution_time": round(metrics.average_execution_time, 2),
            "throughput": round(metrics.throughput, 4),
            "efficiency_score": round(metrics.efficiency_score, 4),
            "consistency_score": round(metrics.consistency_score, 4),
            "error_recovery_rate": round(metrics.error_recovery_rate, 4),
            "replanning_frequency": round(metrics.replanning_frequency, 4),
            "test_coverage_score": round(metrics.test_coverage_score, 4),
            "agent_confidence_average": round(metrics.agent_confidence_average, 4),
            "supervision_quality_score": round(metrics.supervision_quality_score, 4)
        }
    
    def export_evaluation_report(self, evaluation_report: Dict[str, Any], output_file: str):
        """Export evaluation report to file"""
        with open(output_file, 'w') as f:
            json.dump(evaluation_report, f, indent=2)
        
        self.logger.info(f"Evaluation report exported to: {output_file}")
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations"""
        if not self.evaluation_history:
            return {"message": "No evaluations performed yet"}
        
        overall_scores = [eval_data["overall_score"] for eval_data in self.evaluation_history]
        
        return {
            "total_evaluations": len(self.evaluation_history),
            "latest_score": overall_scores[-1] if overall_scores else 0,
            "average_score": statistics.mean(overall_scores) if overall_scores else 0,
            "best_score": max(overall_scores) if overall_scores else 0,
            "worst_score": min(overall_scores) if overall_scores else 0,
            "score_trend": "improving" if len(overall_scores) > 1 and overall_scores[-1] > overall_scores[0] else "stable_or_declining"
        } 