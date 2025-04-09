"""
Base Agent System Interface

This module provides the base classes and interfaces for agent systems.
"""

import abc
from typing import Dict, Any, Optional, Type
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
            Dictionary of run results (e.g., messages, token usage)
        """
        pass

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
            
            # Start problem timer
            problem_id = problem.get("id", str(uuid.uuid4()))
            self.metrics_collector.start_timer(
                "problem_evaluation", 
                {"problem_id": problem_id, "agent_system": self.name}
            )
        
        # Run the agent system
        run_result = self.run_agent(problem, **kwargs)
        
        # Record execution time
        execution_time_ms = 0
        if self.metrics_collector:
            execution_time_ms = self.metrics_collector.stop_timer("problem_evaluation")
            
            # Record token usage if available
            token_usage = run_result.get("token_usage", {})
            self._record_token_usage(problem_id, token_usage, execution_time_ms, run_result.get("messages", []))
        
        # Evaluate results
        evaluation_results = {}
        if self.evaluator:
            evaluation_results = self.evaluator.evaluate(problem, run_result)
            
            # Record evaluation metrics
            if self.metrics_collector:
                self.metrics_collector.record_evaluation_result(
                    problem_id=problem_id,
                    score=evaluation_results.get("math_score", 0),
                    duration_ms=execution_time_ms,
                    tags={
                        "agent_system": self.name,
                        "evaluator": self.evaluator.name,
                        "problem_type": self.evaluator.name
                    }
                )
        
        # Return final results
        return {
            **evaluation_results,  # Include all evaluation results
            "token_usage": run_result.get("token_usage", {}),
            "execution_time_ms": execution_time_ms,
        }
    
    def _record_token_usage(self, problem_id: str, token_usage: Dict, execution_time_ms: float, messages: list):
        """
        Record token usage metrics from the run result.
        
        Args:
            problem_id: ID of the problem
            token_usage: Token usage dictionary
            execution_time_ms: Execution time in milliseconds
            messages: List of messages from the run
        """
        if not self.metrics_collector:
            return
            
        try:
            from langchain_core.messages import AIMessage
        except ImportError:
            return
            
        # Extract token usage from AIMessages
        for message in messages:
            if isinstance(message, AIMessage) and hasattr(message, 'usage_metadata'):
                usage_metadata = message.usage_metadata
                if usage_metadata:
                    input_tokens = usage_metadata.get('input_tokens', 0)
                    output_tokens = usage_metadata.get('output_tokens', 0)
                    total_tokens = usage_metadata.get('total_tokens', 0)
                    
                    # Record token usage with centralized metrics collector
                    self.metrics_collector.record_llm_usage(
                        agent_id=message.id,
                        model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                        total_tokens=total_tokens,
                        latency_ms=execution_time_ms / len(messages) if messages else 0,
                        tags={"agent_system": self.name, "problem_id": problem_id}
                    )

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
