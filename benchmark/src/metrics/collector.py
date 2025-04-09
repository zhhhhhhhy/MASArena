"""
Centralized Metrics Collection

This module provides a centralized metrics collection system that unifies
how metrics are recorded across the benchmark.
"""

import time
from typing import Dict, Any, List, Optional


class MetricsCollector:
    """
    Centralized metrics collection system that handles all metric recording.
    
    This class wraps various metric collectors (system, agent, inter_agent)
    and provides a unified interface for recording metrics.
    """
    
    def __init__(self, metrics_registry=None):
        """
        Initialize the metrics collector.
        
        Args:
            metrics_registry: The metrics registry to use
        """
        self.metrics_registry = metrics_registry
        self.timers = {}
    
    def set_metrics_registry(self, metrics_registry):
        """Set the metrics registry"""
        self.metrics_registry = metrics_registry
        return self
    
    def record_metric(self, metric_name: str, value: Any, tags: Dict[str, str] = None):
        """
        Record a generic metric.
        
        Args:
            metric_name: Name of the metric
            value: Value to record
            tags: Additional tags for the metric
        """
        if self.metrics_registry:
            collector = self.metrics_registry.get_collector("system")
            if collector:
                collector.collect_point(metric_name, value, tags or {})
    
    def start_timer(self, timer_name: str, tags: Dict[str, str] = None):
        """
        Start a timer for measuring durations.
        
        Args:
            timer_name: Name of the timer
            tags: Additional tags for the timer
        """
        self.timers[timer_name] = {
            "start_time": time.time(),
            "tags": tags or {}
        }
    
    def stop_timer(self, timer_name: str) -> float:
        """
        Stop a timer and record the duration.
        
        Args:
            timer_name: Name of the timer
            
        Returns:
            Duration in milliseconds
        """
        if timer_name not in self.timers:
            return 0.0
            
        timer_info = self.timers.pop(timer_name)
        duration_ms = (time.time() - timer_info["start_time"]) * 1000
        
        self.record_metric(
            f"{timer_name}.duration_ms", 
            duration_ms,
            timer_info["tags"]
        )
        
        return duration_ms
    
    def record_llm_usage(
        self, 
        agent_id: str, 
        model_name: str, 
        prompt_tokens: int = 0,
        completion_tokens: int = 0, 
        total_tokens: int = 0,
        latency_ms: float = 0.0,
        tags: Dict[str, str] = None
    ):
        """
        Record LLM usage metrics.
        
        Args:
            agent_id: ID of the agent
            model_name: Name of the LLM model
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            total_tokens: Total number of tokens (if not calculated from prompt+completion)
            latency_ms: Latency in milliseconds
            tags: Additional tags
        """
        if not self.metrics_registry:
            return
            
        # Get the agent collector
        agent_collector = self.metrics_registry.get_collector("agent")
        if not agent_collector:
            return
            
        # Calculate total tokens if not provided
        if total_tokens == 0 and (prompt_tokens > 0 or completion_tokens > 0):
            total_tokens = prompt_tokens + completion_tokens
            
        # Record detailed metrics
        agent_collector.record_llm_usage(
            agent_id=agent_id,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            tags=tags or {}
        )
        
        # Also record summary metrics
        self.record_metric(
            f"llm.tokens.{agent_id}", 
            total_tokens,
            {
                "agent_id": agent_id,
                "model": model_name,
                **(tags or {})
            }
        )
    
    def record_agent_interaction(self, from_agent: str, to_agent: str, message_type: str, tags: Dict[str, str] = None):
        """
        Record an interaction between agents.
        
        Args:
            from_agent: ID of the sending agent
            to_agent: ID of the receiving agent
            message_type: Type of message
            tags: Additional tags
        """
        if not self.metrics_registry:
            return
            
        inter_agent_collector = self.metrics_registry.get_collector("inter_agent")
        if inter_agent_collector:
            inter_agent_collector.record_interaction(
                from_agent=from_agent,
                to_agent=to_agent,
                message_type=message_type,
                tags=tags or {}
            )
    
    def record_evaluation_result(
        self, 
        problem_id: str, 
        score: float, 
        duration_ms: float, 
        tags: Dict[str, str] = None
    ):
        """
        Record an evaluation result.
        
        Args:
            problem_id: ID of the problem
            score: Evaluation score
            duration_ms: Duration in milliseconds
            tags: Additional tags
        """
        if not self.metrics_registry:
            return
            
        system_collector = self.metrics_registry.get_collector("system")
        if system_collector:
            # Record the score
            system_collector.collect_point(
                "evaluation.score", 
                score,
                {
                    "problem_id": problem_id,
                    **(tags or {})
                }
            )
            
            # Record the duration
            system_collector.record_latency(
                "evaluation.duration", 
                duration_ms,
                {
                    "problem_id": problem_id, 
                    **(tags or {})
                }
            )
            
            # Record pass/fail
            system_collector.collect_point(
                "evaluation.passed", 
                1.0 if score == 1 else 0.0,
                {
                    "problem_id": problem_id, 
                    **(tags or {})
                }
            ) 