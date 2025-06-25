# Failure Attribution Module

This module provides automated failure attribution capabilities for analyzing multi-agent system responses and identifying failure causes.

## Overview

The failure attribution module analyzes agent conversation histories to identify:
- Which agent made an error
- At which step the error occurred
- What type of error it was
- The specific reason for the failure

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables for GPT models (if using):
```bash
# Create a .env file in the failure directory

# Option 1: Standard OpenAI API (recommended)
OPENAI_API_KEY=your_openai_api_key_here
# Optional: Custom base URL
# OPENAI_API_BASE=https://api.chatanywhere.tech

# Option 2: Azure OpenAI
# AZURE_OPENAI_API_KEY=your_azure_api_key_here
# AZURE_OPENAI_ENDPOINT=your_azure_endpoint_here
```

## Usage

### Basic Usage

The module can be run using the `inference.py` script with various analysis methods:

```bash
python inference.py --method all_at_once --model gpt-4o --directory_path ../results/agent_responses
```

### Analysis Methods

1. **All-at-once Analysis**: Analyzes the entire conversation history at once
```bash
python inference.py --method all_at_once --model gpt-4o
```

2. **Step-by-step Analysis**: Analyzes the conversation incrementally, step by step
```bash
python inference.py --method step_by_step --model gpt-4o
```

3. **Binary Search Analysis**: Uses binary search to efficiently locate errors
```bash
python inference.py --method binary_search --model gpt-4o
```

### Supported Models

#### GPT Models (Azure OpenAI)
- `gpt-4o`
- `gpt4`
- `gpt4o-mini`

Example with Standard OpenAI:
```bash
python inference.py --method all_at_once --model gpt-4o \
    --api_key your_api_key \
    --directory_path ../results/agent_responses
```

Example with Azure OpenAI:
```bash
python inference.py --method all_at_once --model gpt-4o \
    --api_key your_azure_api_key \
    --azure_endpoint your_azure_endpoint \
    --directory_path ../results/agent_responses
```

Example with Custom Base URL:
```bash
python inference.py --method all_at_once --model gpt-4o \
    --api_key your_api_key \
    --openai_base_url https://api.chatanywhere.tech \
    --directory_path ../results/agent_responses
```

#### Local Models
- `llama-8b` (meta-llama/Llama-3.1-8B-Instruct)
- `llama-70b` (meta-llama/Llama-3.1-70B-Instruct)
- `qwen-7b` (Qwen/Qwen2.5-7B-Instruct)
- `qwen-72b` (Qwen/Qwen2.5-72B-Instruct)

Example with local model:
```bash
python inference.py --method all_at_once --model llama-8b \
    --device cuda:0 \
    --directory_path ../results/agent_responses
```

### Command Line Arguments

#### Required Arguments
- `--method`: Analysis method (`all_at_once`, `step_by_step`, `binary_search`)
- `--model`: Model to use (see supported models above)

#### Optional Arguments
- `--directory_path`: Path to agent response JSON files (default: `../results/agent_responses`)
- `--api_key`: OpenAI API key (for GPT models, supports both standard and Azure)
- `--azure_endpoint`: Azure OpenAI endpoint (if using Azure OpenAI)
- `--openai_base_url`: Custom OpenAI base URL (for third-party providers)
- `--api_version`: Azure OpenAI API version (default: `2024-08-01-preview`)
- `--max_tokens`: Maximum tokens for response (default: 1024)
- `--device`: Device for local models (default: `cuda:0` if available, else `cpu`)

### Input Data Format

The module expects JSON files in the following format:

```json
{
    "problem_id": "problem_1",
    "agent_system": "multi_agent",
    "run_id": "run_123",
    "timestamp": "2024-06-24T16:11:44",
    "responses": [
        {
            "timestamp": "2024-06-24T16:11:44.123",
            "problem_id": "problem_1",
            "message_index": 0,
            "agent_id": "agent_1",
            "content": "Agent response content here...",
            "role": "assistant",
            "message_type": "response",
            "usage_metadata": {}
        }
    ]
}
```

### Output

The analysis results are saved to the `outputs/` directory with filenames in the format:
`{method}_{model}_agent_responses.txt`

Example output format:
```
Error Agent: agent_2
Error Step: 3
Error Type: Calculation Error
Reason: The agent made an arithmetic error in the calculation step.
```

## Evaluation

To evaluate the accuracy of failure attribution predictions:

```bash
python evaluate.py --data_path /path/to/annotated/data --evaluation_file outputs/all_at_once_gpt-4o_agent_responses.txt
```

The evaluation script compares predictions against ground truth annotations and reports:
- Agent identification accuracy
- Error step identification accuracy

## Directory Structure

```
failure/
├── __init__.py
├── inference.py          # Main inference script
├── evaluate.py           # Evaluation script
├── requirements.txt      # Dependencies
├── README.md            # This file
├── lib/
│   ├── __init__.py
│   ├── utils.py         # GPT-based analysis functions
│   └── local_model.py   # Local model analysis functions
└── outputs/             # Generated analysis results
```

## Examples

### Example 1: Quick Analysis with GPT-4o
```bash
# Set environment variables (Option 1: Standard OpenAI)
export OPENAI_API_KEY="your_key"

# Or (Option 2: Azure OpenAI)
# export AZURE_OPENAI_API_KEY="your_azure_key"
# export AZURE_OPENAI_ENDPOINT="your_azure_endpoint"

# Run analysis
python inference.py --method all_at_once --model gpt-4o
```

### Example 2: Local Model Analysis
```bash
# Run with local Llama model
python inference.py --method step_by_step --model llama-8b --device cuda:0
```

### Example 3: Binary Search for Large Conversations
```bash
# Efficient analysis for long conversations
python inference.py --method binary_search --model gpt-4o --max_tokens 2048
```

### Example 4: Custom Data Path
```bash
# Analyze data from custom directory
python inference.py --method all_at_once --model gpt-4o \
    --directory_path /path/to/your/agent/responses
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Use smaller models or CPU inference
```bash
python inference.py --method all_at_once --model llama-8b --device cpu
```

2. **API Rate Limits**: The script includes automatic rate limiting, but you may need to adjust delays

3. **Missing Dependencies**: Install all requirements
```bash
pip install -r requirements.txt
```

4. **Model Loading Issues**: Ensure you have sufficient disk space and memory for local models

### Performance Tips

- Use `all_at_once` for comprehensive analysis
- Use `binary_search` for efficient error localization in long conversations
- Use `step_by_step` for detailed incremental analysis
- For local models, ensure adequate GPU memory or use CPU inference

## Contributing

To extend the module:
1. Add new analysis methods in `lib/utils.py` or `lib/local_model.py`
2. Update the model mappings in `inference.py`
3. Add corresponding command line options
4. Update this README with new features

## License

This module is part of the Multi-Agent Benchmark project.