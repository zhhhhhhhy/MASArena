#!/bin/bash
# run_aflow_optimize.sh - Run AFlow optimization pipeline with configurable parameters

# Usage:
#   ./run_aflow_optimize.sh [benchmark] [graph_path] [optimized_path] [validation_rounds] [eval_rounds] [max_rounds]

# Default values
BENCHMARK=${1:-humaneval}
GRAPH_PATH=${2:-"mas_arena/configs/aflow"}
OPTIMIZED_PATH=${3:-"example/aflow/humaneval/optimization2"}
VALIDATION_ROUNDS=${4:-1}
EVAL_ROUNDS=${5:-1}
MAX_ROUNDS=${6:-3}

# Set project root (assuming script is in project root or scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT" || exit 1

# Load environment variables from .env file
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found. Using system environment variables."
fi

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "../.venv" ]; then
    source ../.venv/bin/activate
fi

# Print header
echo ""
echo "====================================================="
echo "🚀 Running AFlow Optimization Pipeline"
echo "====================================================="
echo "Benchmark: $BENCHMARK"
echo "Graph Path: $GRAPH_PATH"
echo "Optimized Path: $OPTIMIZED_PATH"
echo "Validation Rounds: $VALIDATION_ROUNDS"
echo "Evaluation Rounds: $EVAL_ROUNDS"
echo "Max Optimization Rounds: $MAX_ROUNDS"
echo "====================================================="
echo ""

# Run the Python main script
python run_aflow_optimize.py \
    --benchmark "$BENCHMARK" \
    "$GRAPH_PATH" \
    "$OPTIMIZED_PATH" \
    --validation_rounds "$VALIDATION_ROUNDS" \
    --eval_rounds "$EVAL_ROUNDS" \
    --max_rounds "$MAX_ROUNDS"

# Exit with the same status as the Python script
exit $?