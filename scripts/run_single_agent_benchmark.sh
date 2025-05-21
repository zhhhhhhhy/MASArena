#!/bin/bash
# Single Agent Benchmark Runner Script
# Usage: ./scripts/run_single_agent_benchmark.sh [limit]

# Default limit
LIMIT=${1:-2}

# Define task categories
TASK_CATEGORIES=("code_generation" "instruction_following" "math" "mmlu" "qa")

# Define models (reasoning and non-reasoning)
REASONING_MODELS=(
  "o4-mini"
  "claude-3-7-sonnet-20250219-thinking"
  "Qwen/QwQ-32B-Preview"
  "deepseek-ai/DeepSeek-R1"
  "gemini-2.5-flash-preview-04-17-thinking"
  "grok-3-think"
)

NON_REASONING_MODELS=(
  "gpt-4o-mini"
  "claude-3-7-sonnet-latest"
  "Qwen/Qwen3-32B"
  "deepseek-ai/DeepSeek-V3"
  "gemini-2.5-flash-preview-04-17-nothinking"
  "grok-3"
)

# Function to get tasks for a category
get_tasks() {
  local category=$1
  case $category in
    "code_generation")
      echo "humaneval mbpp"
      ;;
    "instruction_following")
      echo "ifeval"
      ;;
    "math")
      echo "aime math"
      ;;
    "mmlu")
      echo "mmlu-pro"
      ;;
    "qa")
      echo "drop"
      ;;
  esac
}

# Create necessary directories
mkdir -p results metrics logs

# Create log file
LOG_FILE="logs/single_agent_benchmark_$(date +%Y%m%d_%H%M%S).log"
touch $LOG_FILE

echo "Starting Single Agent Benchmark" | tee -a $LOG_FILE
echo "====================================================" | tee -a $LOG_FILE

# Activate virtual environment if exists
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# Loop through task categories and tasks
for TASK_CATEGORY in "${TASK_CATEGORIES[@]}"; do
  # Get tasks for this category
  TASKS=($(get_tasks "$TASK_CATEGORY"))
  
  for TASK_NAME in "${TASKS[@]}"; do
    
    echo "Task Category: $TASK_CATEGORY, Task: $TASK_NAME" | tee -a $LOG_FILE
    
    # Loop through reasoning models
    for MODEL_NAME in "${REASONING_MODELS[@]}"; do
      # Determine model provider
      MODEL_PREFIX=$(echo $MODEL_NAME | cut -d'-' -f1 | cut -d'/' -f1)
      
      # Get provider name from mapping
      if grep -q "\"$MODEL_PREFIX\":" scripts/model_providers.json; then
        MODEL_PROVIDER=$(grep "\"$MODEL_PREFIX\":" scripts/model_providers.json | cut -d':' -f2 | tr -d ' ",' | head -n 1)
      else
        MODEL_PROVIDER=$MODEL_PREFIX
      fi
      
      echo "  Running with model: $MODEL_NAME (Provider: $MODEL_PROVIDER, Reasoning: true)" | tee -a $LOG_FILE
      
      # Set the model in .env file
      sed -i.bak "s/^MODEL_NAME=.*/MODEL_NAME=$MODEL_NAME/" .env
      
      # Define results directory
      RESULTS_DIR="results/${TASK_CATEGORY}_${TASK_NAME}/single_agent/${MODEL_PROVIDER}_${MODEL_NAME}/reasoning"
      mkdir -p "$RESULTS_DIR"
      
      # Run the benchmark with reasoning
      python main.py \
        --benchmark "$TASK_NAME" \
        --agent-system "single_agent" \
        --limit "$LIMIT" \
        --output-dir "$RESULTS_DIR" >> $LOG_FILE 2>&1
    done
    
    # Loop through non-reasoning models
    for MODEL_NAME in "${NON_REASONING_MODELS[@]}"; do
      # Determine model provider
      MODEL_PREFIX=$(echo $MODEL_NAME | cut -d'-' -f1 | cut -d'/' -f1)
      
      # Get provider name from mapping
      if grep -q "\"$MODEL_PREFIX\":" scripts/model_providers.json; then
        MODEL_PROVIDER=$(grep "\"$MODEL_PREFIX\":" scripts/model_providers.json | cut -d':' -f2 | tr -d ' ",' | head -n 1)
      else
        MODEL_PROVIDER=$MODEL_PREFIX
      fi
      
      echo "  Running with model: $MODEL_NAME (Provider: $MODEL_PROVIDER, Reasoning: false)" | tee -a $LOG_FILE
      
      # Set the model in .env file
      sed -i.bak "s/^MODEL_NAME=.*/MODEL_NAME=$MODEL_NAME/" .env
      
      # Define results directory
      RESULTS_DIR="results/${TASK_CATEGORY}_${TASK_NAME}/single_agent/${MODEL_PROVIDER}_${MODEL_NAME}/no_reasoning"
      mkdir -p "$RESULTS_DIR"
      
      # Run the benchmark without reasoning
      python main.py \
        --benchmark "$TASK_NAME" \
        --agent-system "single_agent" \
        --limit "$LIMIT" \
        --output-dir "$RESULTS_DIR" >> $LOG_FILE 2>&1
    done
  done
done

echo "====================================================" | tee -a $LOG_FILE
echo "Benchmark completed! See $LOG_FILE for details." | tee -a $LOG_FILE

# Exit with success
exit 0 