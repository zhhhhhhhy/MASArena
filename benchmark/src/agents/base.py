"""
Base Agent System Interface

This module provides the base classes and interfaces for agent systems.
"""

import abc
from typing import Dict, Any, Optional
import time
import uuid

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
    
    @abc.abstractmethod
    def evaluate(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Evaluate the agent system on a given problem.
        
        Args:
            problem: Dictionary containing the problem data
            
        Returns:
            Dictionary of evaluation results including any metrics
        """
        pass
    
    def set_metrics_registry(self, metrics_registry):
        """Set the metrics registry for this agent system"""
        self.metrics_registry = metrics_registry
        return self
    
    def record_metric(self, metric_name: str, value: Any, tags: Dict[str, str] = None):
        """Record a metric if metrics registry is available"""
        if self.metrics_registry:
            collector = self.metrics_registry.get_collector("system")
            if collector:
                collector.collect_point(metric_name, value, tags or {})
    
    def record_timing(self, fn_name: str, start_time: float, tags: Dict[str, str] = None):
        """Record timing information for a function"""
        duration_ms = (time.time() - start_time) * 1000
        self.record_metric(f"{self.name}.{fn_name}.duration", duration_ms, tags)
        return duration_ms
    
    def generate_run_id(self) -> str:
        """Generate a unique run ID"""
        return str(uuid.uuid4())
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent system"""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "config": self.config
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
        cls._registry[name] = {
            "class": agent_class,
            "default_config": default_config
        }
    
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
            name: {
                "class": info["class"].__name__,
                "default_config": info["default_config"]
            }
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