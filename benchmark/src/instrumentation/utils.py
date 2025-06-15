"""
Utility functions for instrumentation and metrics collection.
"""

from data.model_data import MODEL_DATA

DTY_TYPE_SIZES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1
}
def calculate_kv_cache_size(model_name: str, num_tokens: int, dtype: str = "float16") -> float:
    """
    Calculate the KV cache size in bytes for a given model and number of tokens.
    
    Args:
        model_name (str): Name of the model (e.g., "meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen-7B")
        num_tokens (int): Number of tokens in the context
        dtype (str): Data type used for KV cache ("float16", "bfloat16", "float32", "int8")
        
    Returns:
        float: KV cache size in bytes
        
    Formula:
        KV cache size = 2 * num_layers * num_tokens * num_kv_heads * head_size * dtype_size
        where:
        - 2 represents key and value matrices
        - num_layers is the number of transformer layers
        - num_kv_heads is the number of key-value heads (may differ from attention heads)
        - head_size is the dimension of each attention head
        - dtype_size is the size of the data type in bytes
    """
    try:
        if not model_name or not isinstance(num_tokens, (int, float)) or num_tokens <= 0:
            return 0.0
            
        if model_name not in MODEL_DATA:
            print(f"Model {model_name} not found in configurations")
            return 0.0
            
        config = MODEL_DATA[model_name]
        
        # Verify required configuration parameters
        required_params = ["hidden_size", "num_attention_heads", "num_hidden_layers", "num_key_value_heads"]
        if not all(param in config for param in required_params):
            print(f"Missing required parameters for model {model_name}")
            return 0.0
        
        # Calculate head size
        head_size = config["hidden_size"] // config["num_attention_heads"]
        
        # Determine dtype size in bytes
        if dtype not in DTY_TYPE_SIZES:
            print(f"Unsupported data type: {dtype}, using float16")
            dtype = "float16"
            
        dtype_size = DTY_TYPE_SIZES[dtype]
        
        # Calculate total elements in KV cache
        total_elements = (
            2 *  # Key and value matrices
            config["num_hidden_layers"] *
            num_tokens *
            config["num_key_value_heads"] *
            head_size
        )
        
        # Calculate total size in bytes and convert to GB
        total_bytes = total_elements * dtype_size
        
        return total_bytes
        
    except Exception as e:
        print(f"Error calculating KV cache size for model {model_name}: {str(e)}")
        return 0.0
