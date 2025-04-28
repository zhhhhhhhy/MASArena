"""
Model parameter and activation data for memory estimation.

This module contains synthetic data about model sizes and their typical
activated parameters during inference, including parameter format and size information.
"""

MODEL_DATA = {
    "gpt-3.5-turbo": {
        "parameter_size_b": 6.7,
        "activated_size_b": 2.01,  # 30% activation
        "parameter_format": "FP16",
        "bytes_per_parameter": 2
    },
    "gpt-4o-mini": {
        "parameter_size_b": 1.3,
        "activated_size_b": 0.39,  # 30% activation
        "parameter_format": "FP16",
        "bytes_per_parameter": 2
    },
    "gpt-4": {
        "parameter_size_b": 175.0,
        "activated_size_b": 61.25,  # 35% activation
        "parameter_format": "FP16",
        "bytes_per_parameter": 2
    },
    "gpt-4-turbo": {
        "parameter_size_b": 230.0,
        "activated_size_b": 92.0,  # 40% activation
        "parameter_format": "FP16",
        "bytes_per_parameter": 2
    },
    "claude-3-opus": {
        "parameter_size_b": 150.0,
        "activated_size_b": 60.0,  # 40% activation
        "parameter_format": "FP16",
        "bytes_per_parameter": 2
    },
    "claude-3-sonnet": {
        "parameter_size_b": 70.0,
        "activated_size_b": 24.5,  # 35% activation
        "parameter_format": "FP16",
        "bytes_per_parameter": 2
    },
    "claude-3-haiku": {
        "parameter_size_b": 20.0,
        "activated_size_b": 6.0,  # 30% activation
        "parameter_format": "FP16",
        "bytes_per_parameter": 2
    },
    "llama-2-7b": {
        "parameter_size_b": 7.0,
        "activated_size_b": 1.75,  # 25% activation
        "parameter_format": "FP16",
        "bytes_per_parameter": 2
    },
    "llama-2-13b": {
        "parameter_size_b": 13.0,
        "activated_size_b": 3.9,  # 30% activation
        "parameter_format": "FP16",
        "bytes_per_parameter": 2
    },
    "llama-2-70b": {
        "parameter_size_b": 70.0,
        "activated_size_b": 24.5,  # 35% activation
        "parameter_format": "FP16",
        "bytes_per_parameter": 2
    },
    "mistral-7b": {
        "parameter_size_b": 7.0,
        "activated_size_b": 1.75,  # 25% activation
        "parameter_format": "FP16",
        "bytes_per_parameter": 2
    },
    "mixtral-8x7b": {
        "parameter_size_b": 56.0,
        "activated_size_b": 16.8,  # 30% activation
        "parameter_format": "FP16",
        "bytes_per_parameter": 2
    },
    "gemma-7b": {
        "parameter_size_b": 7.0,
        "activated_size_b": 1.75,  # 25% activation
        "parameter_format": "FP16",
        "bytes_per_parameter": 2
    },
    "gemma-2b": {
        "parameter_size_b": 2.0,
        "activated_size_b": 0.4,  # 20% activation
        "parameter_format": "FP16",
        "bytes_per_parameter": 2
    }
} 