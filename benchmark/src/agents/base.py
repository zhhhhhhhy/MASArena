"""
Base Agent System Interface

This module provides the base classes and interfaces for agent systems.
"""

import abc
from typing import Dict, Any, Optional, Type, Callable
import time
import uuid
import os
from pathlib import Path


class AgentSystem(abc.ABC):
    """Base class for all agent systems in the benchmark framework"""

    def __init__(self, name: str = None, config: Dict[str, Any] = None):
        """
        Initialize the agent system.

        Args:
            name: Name of the agent system
            config: Configuration parameters for the agent system
        """
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.metrics_registry = None
        self.evaluator = None
        self.metrics_collector = None
        
    def _initialize_evaluator(self, evaluator_type: Type = None):
        """
        Initialize the appropriate evaluator based on configuration.
        
        Args:
            evaluator_type: Optional evaluator class to use
        """
        if self.evaluator is not None:
            return
            
        # Get evaluator name from config
        evaluator_name = self.config.get("evaluator", "math")
        
        if evaluator_type is None:
            # Import here to avoid circular imports
            try:
                from benchmark.src.evaluators import MathEvaluator
                evaluator_type = MathEvaluator
            except ImportError:
                raise ImportError("Could not import evaluator. Please provide evaluator_type.")
        
        # Create evaluator instance
        self.evaluator = evaluator_type(
            name=evaluator_name,
            config={
                "data_path": self.config.get("data_path", f"benchmark/data/{evaluator_name}_test.jsonl"),
                "log_path": self.config.get("log_path", f"benchmark/data/results/{evaluator_name.upper()}")
            }
        )

    def _initialize_metrics_collector(self):
        """Initialize the metrics collector"""
        if self.metrics_collector is not None:
            return
            
        try:
            from benchmark.src.metrics.collector import MetricsCollector
            self.metrics_collector = MetricsCollector()
            if self.metrics_registry:
                self.metrics_collector.set_metrics_registry(self.metrics_registry)
        except ImportError:
            pass  # Metrics collector is optional

    @abc.abstractmethod
    def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the agent system on a given problem.
        
        This method should be implemented by subclasses to run the actual agent logic
        without handling evaluation or metrics.
        
        Args:
            problem: Dictionary containing the problem data
            
        Returns:
            Dictionary of run results (e.g., messages)
        """
        pass

    def _record_token_usage(self, problem_id: str, execution_time_ms: float, messages: list):
        """
        Record token usage metrics from AI messages with usage_metadata.
        
        Args:
            problem_id: ID of the problem
            execution_time_ms: Execution time in milliseconds
            messages: List of messages from the run
            
        Returns:
            Dictionary with collected LLM usage metrics
        """
        if not self.metrics_collector:
            return {}
            
        try:
            from langchain_core.messages import AIMessage
        except ImportError:
            return {}
        
        # Track metrics from AIMessages with usage_metadata
        total_tokens = 0
        message_count = 0
        usage_metrics = []
        
        for message in messages:
            if isinstance(message, AIMessage) and hasattr(message, 'usage_metadata') and message.usage_metadata:
                message_count += 1
                usage_metadata = message.usage_metadata
                agent_id = message.name if hasattr(message, 'name') and message.name else message.id if hasattr(message, 'id') and message.id else f"agent_{hash(message)}"
                
                # Extract metrics from usage_metadata
                input_tokens = usage_metadata.get('input_tokens', 0)
                output_tokens = usage_metadata.get('output_tokens', 0)
                reasoning_tokens = usage_metadata["output_token_details"].get("reasoning", 0)
                total_tokens_msg = usage_metadata.get('total_tokens', input_tokens + output_tokens)
                total_tokens += total_tokens_msg
                
                input_token_details = usage_metadata.get('input_token_details', {})
                output_token_details = usage_metadata.get('output_token_details', {})
                
                # Record detailed token metrics directly from the message's usage_metadata
                self.metrics_collector.record_llm_usage(
                    agent_id=agent_id,
                    model_name=os.getenv("MODEL_NAME", ""),
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    total_tokens=total_tokens_msg,
                    reasoning_tokens=reasoning_tokens,
                    input_token_details=input_token_details,
                    output_token_details=output_token_details,
                    latency_ms=execution_time_ms / message_count if message_count > 0 else 0,
                    tags={"agent_system": self.name, "problem_id": problem_id}
                )
                
                # Collect usage metrics
                usage_metrics.append({
                    "agent_id": agent_id,
                    "model_name": os.getenv("MODEL_NAME", "gpt-4o-mini"),
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": total_tokens_msg,
                    "latency_ms": execution_time_ms / message_count if message_count > 0 else 0
                })
            
            # Record agent interactions from tuple-style messages
            elif isinstance(message, tuple) and len(message) > 1:
                agent_id, content = message
                
                if agent_id != "user" and self.metrics_collector:  # Skip user messages
                    self.metrics_collector.record_agent_interaction(
                        from_agent="system",
                        to_agent=agent_id,
                        message_type="response",
                        content=content,
                        tags={"agent_system": self.name, "problem_id": problem_id}
                    )
        
        # Record total tokens for this problem
        if message_count > 0:
            self.metrics_collector.record_metric(
                "problem.total_tokens",
                total_tokens,
                {
                    "problem_id": problem_id,
                    "agent_system": self.name
                }
            )
        
        # Return collected metrics
        return {
            "total_tokens": total_tokens,
            "message_count": message_count,
            "agent_usage": usage_metrics
        }

    def evaluate(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Evaluate the agent system on a given problem.
        
        This method handles running the agent, evaluating the results,
        and collecting metrics.

        Args:
            problem: Dictionary containing the problem data

        Returns:
            Dictionary of evaluation results including any metrics
        """
        # Initialize components if needed
        metrics_registry = kwargs.get("metrics_registry", self.metrics_registry)
        if metrics_registry:
            self.metrics_registry = metrics_registry
            
        self._initialize_evaluator()
        self._initialize_metrics_collector()
        
        # Set up metrics collection
        if self.metrics_collector:
            self.metrics_collector.set_metrics_registry(self.metrics_registry)
            
            # Generate a stable problem ID if not present
            problem_id = problem.get("id", f"problem_{hash(problem['problem'])}")
            
            # Record problem metadata
            self.metrics_collector.record_metric(
                "problem.process", 
                1.0, 
                {
                    "problem_id": problem_id,
                    "agent_system": self.name,
                    "evaluator": self.evaluator.name if self.evaluator else "unknown"
                }
            )
            
            # Start problem timer
            self.metrics_collector.start_timer(
                "problem_evaluation", 
                {
                    "problem_id": problem_id, 
                    "agent_system": self.name,
                    "evaluator": self.evaluator.name if self.evaluator else "unknown"
                }
            )
        
        try:
            # Run the agent system
            run_result = self.run_agent(problem, **kwargs)
            
            # Record execution time
            execution_time_ms = 0
            if self.metrics_collector:
                execution_time_ms = self.metrics_collector.stop_timer("problem_evaluation")
                
                # Extract and record metrics from AI message metadata
                messages = run_result.get("messages", [])
                usage_metrics = self._record_token_usage(problem_id, execution_time_ms, messages)
            
            # Evaluate results
            evaluation_results = {}
            if self.evaluator:
                evaluation_results = self.evaluator.evaluate(problem, run_result)
                
                # Record evaluation metrics
                if self.metrics_collector:
                    score = evaluation_results.get("math_score", 0)
                    self.metrics_collector.record_evaluation_result(
                        problem_id=problem_id,
                        score=score,
                        duration_ms=execution_time_ms,
                        metrics={
                            "passed": score == 1,
                            "agent_system": self.name,
                        },
                        tags={
                            "agent_system": self.name,
                            "evaluator": self.evaluator.name,
                            "problem_type": self.evaluator.name
                        }
                    )
            
            # Return final results
            return {
                **evaluation_results,  # Include all evaluation results
                "messages": messages,  # Include messages for token analysis in benchmark_runner
                "execution_time_ms": execution_time_ms,
                "llm_usage": usage_metrics  # Include the collected LLM usage metrics
            }
            
        except Exception as e:
            # Record error
            if self.metrics_collector:
                self.metrics_collector.stop_timer("problem_evaluation")
                self.metrics_collector.record_error(
                    "evaluation_error",
                    str(e),
                    {
                        "problem_id": problem_id,
                        "agent_system": self.name,
                        "error_type": type(e).__name__
                    }
                )
            raise  # Re-raise the exception
    
    def with_timing(self, func_name: str, tags: Dict[str, str] = None) -> Callable:
        """
        Create a decorator for timing function execution.
        
        This is a convenience method for creating timing decorators.
        
        Args:
            func_name: Name to use for the timer
            tags: Additional tags for the timer
            
        Returns:
            A decorator function
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.metrics_collector:
                    return func(*args, **kwargs)
                    
                self.metrics_collector.start_timer(func_name, tags)
                try:
                    return func(*args, **kwargs)
                finally:
                    self.metrics_collector.stop_timer(func_name)
            return wrapper
        return decorator
    
    def set_metrics_registry(self, metrics_registry):
        """Set the metrics registry for this agent system"""
        self.metrics_registry = metrics_registry
        if self.metrics_collector:
            self.metrics_collector.set_metrics_registry(metrics_registry)
        return self

    def record_metric(self, metric_name: str, value: Any, tags: Dict[str, str] = None):
        """Record a metric if metrics registry is available"""
        if self.metrics_collector:
            self.metrics_collector.record_metric(metric_name, value, tags or {})
        elif self.metrics_registry:
            collector = self.metrics_registry.get_collector("system")
            if collector:
                collector.collect_point(metric_name, value, tags or {})

    def generate_run_id(self) -> str:
        """Generate a unique run ID"""
        return str(uuid.uuid4())

    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent system"""
        return {
            "name": self.name, 
            "type": self.__class__.__name__, 
            "config": self.config,
            "evaluator": self.evaluator.name if self.evaluator else None
        }

    def get_collected_metrics(self, problem_id=None, metric_type=None):
        """
        Retrieve metrics collected by the metrics_collector.
        
        Args:
            problem_id: Optional problem ID to filter metrics by
            metric_type: Optional metric type to filter by (e.g., 'llm_usage', 'evaluation')
            
        Returns:
            Dictionary of collected metrics
        """
        if not self.metrics_collector or not self.metrics_registry:
            return {}
            
        # Get all registered collectors
        collectors = {
            "system": self.metrics_registry.get_collector("system"),
            "agent": self.metrics_registry.get_collector("agent"),
            "inter_agent": self.metrics_registry.get_collector("inter_agent")
        }
        
        # Filter out None collectors
        collectors = {k: v for k, v in collectors.items() if v is not None}
        
        if not collectors:
            return {}
            
        # Collect metrics from each collector
        all_metrics = {}
        for collector_name, collector in collectors.items():
            metrics = collector.get_metrics()
            
            # Filter by problem_id if provided
            if problem_id:
                metrics = [m for m in metrics if m.get("tags", {}).get("problem_id") == problem_id]
                
            # Filter by metric_type if provided
            if metric_type:
                if metric_type == "llm_usage":
                    metrics = [m for m in metrics if m.get("name", "").startswith("llm.")]
                elif metric_type == "evaluation":
                    metrics = [m for m in metrics if m.get("name", "").startswith("evaluation.")]
            
            all_metrics[collector_name] = metrics
            
        return all_metrics

    def get_llm_usage_metrics(self, problem_id=None):
        """
        Retrieve only LLM usage metrics collected by the metrics_collector.
        
        Args:
            problem_id: Optional problem ID to filter metrics by
            
        Returns:
            List of LLM usage metrics
        """
        all_metrics = self.get_collected_metrics(problem_id=problem_id, metric_type="llm_usage")
        
        # Extract agent metrics which contain LLM usage
        agent_metrics = all_metrics.get("agent", [])
        llm_metrics = [m for m in agent_metrics if m.get("name", "").startswith("llm.")]
        
        return llm_metrics


class AgentSystemRegistry:
    """Registry for agent systems available in the benchmark"""

    _registry = {}

    @classmethod
    def register(cls, name: str, agent_class, **default_config):
        """
        Register an agent system class with the registry.

        Args:
            name: Name to register the agent system under
            agent_class: The agent system class
            default_config: Default configuration parameters
        """
        cls._registry[name] = {"class": agent_class, "default_config": default_config}

    @classmethod
    def get(cls, name: str, config: Dict[str, Any] = None) -> Optional[AgentSystem]:
        """
        Get an instance of the specified agent system.

        Args:
            name: Name of the agent system
            config: Configuration parameters (overrides defaults)

        Returns:
            An instance of the requested agent system or None if not found
        """
        if name not in cls._registry:
            return None

        agent_info = cls._registry[name]
        agent_config = dict(agent_info["default_config"])
        if config:
            agent_config.update(config)

        return agent_info["class"](name=name, config=agent_config)

    @classmethod
    def list_available(cls) -> Dict[str, Any]:
        """List all available agent systems"""
        return {
            name: {"class": info["class"].__name__, "default_config": info["default_config"]}
            for name, info in cls._registry.items()
        }


# Factory function for creating agent systems
def create_agent_system(name: str, config: Dict[str, Any] = None) -> Optional[AgentSystem]:
    """
    Create an agent system by name.

    Args:
        name: Name of the agent system
        config: Configuration parameters

    Returns:
        An instance of the requested agent system
    """
    return AgentSystemRegistry.get(name, config)
