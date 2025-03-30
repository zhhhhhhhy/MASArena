#!/bin/bash
# Benchmark Runner Script
# Usage: ./run_benchmark.sh [benchmark_name] [agent_system] [limit]

# Default values
BENCHMARK=${1:-math}
AGENT_SYSTEM=${2:-swarm} # single_agent, supervisor_mas, swarm
LIMIT=${3:-10}

# Create necessary directories
mkdir -p results metrics

# Print header
echo "====================================================="
echo "Running Multi-Agent Benchmark"
echo "====================================================="
echo "Benchmark: $BENCHMARK"
echo "Agent System: $AGENT_SYSTEM"
echo "Limit: $LIMIT"
echo "====================================================="

# Run the benchmark
# venv
source .venv/bin/activate
python main.py --benchmark $BENCHMARK --agent-system $AGENT_SYSTEM --limit $LIMIT

# Exit with the same status as the Python script
exit $? 