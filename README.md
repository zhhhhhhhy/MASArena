# <div align="center">

  <h1 align="center">MASArena 🏟️</h1>
  <!-- <p align="center"><i>Multi-Agent Systems Arena</i></p> -->
  
  <p align="center">
    <b>Layered Architecture</b> • <b>Stack</b> • <b>Swap</b> • <b>Built for Scale</b>
  </p>
  <img src="assets/intro.svg" alt="MASArena Architecture" width="800"/>
</div>

## 🌟 Core Features

* **🧱 Modular Design**: Swap agents, tools, datasets, prompts, and evaluators with ease.
* **📦 Built-in Benchmarks**: Single/multi-agent datasets for direct comparison.
* **📊 Visual Debugging**: Inspect interactions, accuracy, and tool use.
* **🔧 Tool Support**:  Manage tool selection via pluggable wrappers.
* **🧩 Easy Extensions**: Add agents via subclassing—no core changes.
* **📂 Paired Datasets & Evaluators**: Add new benchmarks with minimal effort.

## 🚀 Quick Start

### 1. Setup

We recommend using [uv](https://docs.astral.sh/uv/) for dependency and virtual environment management.

```bash
# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

### 2. Configure Environment Variables

Create a `.env` file in the project root and set the following:

```bash
OPENAI_API_KEY=your_openai_api_key
MODEL_NAME=gpt-4o-mini
OPENAI_API_BASE=https://api.openai.com/v1
```

### 3. Running Benchmarks

```bash
./run_benchmark.sh
```
* Supported benchmarks: 
  * Math: `math`, `aime`
  * Code: `humaneval`, `mbpp`
  * Reasoning: `drop`, `bbh`, `mmlu_pro`, `ifeval`
* Supported agent systems: 
  * Single Agent: `single_agent`
  * Multi-Agent: `supervisor_mas`, `swarm`, `agentverse`, `chateval`, `evoagent`, `jarvis`, `metagpt`

## Additional Resources 
- [Quick Start](docs/quick/quick_start.md): A quick start guide to get you started with MAS Arena.
- [System Overview](docs/architecture/system_overview.md): A detailed overview of the system architecture.
- [Tool Integration](docs/tools/tool_integration.md): A guide to integrate tools into your agent system.
- [Extending](docs/extending/extending.md): A guide to extend the framework with your own agent system.


## ✅ TODOs

* [ ] Add asynchronous support for model calls
* [ ] Implement failure detection in MAS workflows
* [ ] Add more benchmarks emphasizing tool usage
* [ ] Improve configuration for MAS and tool integration

## 🙌 Contributing

We warmly welcome contributions from the community!

You can contribute in many ways:

* 🧠 **New Agent Systems (MAS):**
  Add novel single- or multi-agent systems to expand the diversity of strategies and coordination models.

* 📊 **New Benchmark Datasets:**
  Bring in domain-specific or task-specific datasets (e.g., reasoning, planning, tool-use, collaboration) to broaden the scope of evaluation.

* 🛠 **New Tools & Toolkits:**
  Extend the framework's tool ecosystem by integrating domain tools (e.g., search, calculators, code editors) and improving tool selection strategies.

* ⚙️ **Improvements & Utilities:**
  Help with performance optimization, failure handling, asynchronous processing, or new visualizations.
