# project_multi_agents_benchmark

## Meeting notes

</details open>

<details>
<summary>2025.02.26</summary>

1. Finished tasks:
	*	Investigated the integration of LangGraph with Openevals for multi-agent system evaluation.
	*	Reviewed and discussed Openevals’ three evaluation methods: LLM-based evaluators, structured output evaluation, and stream-based evaluation.
	*	Explored Agentevals, a wrapper for Openevals, to evaluate agent trajectories and graph trajectories.
	*	Discussed long-term scalability goals, including multi-agent architectures and potential extensions for agent protocols.

2. Takeaway messages:
	*	The focus is on integrating Openevals with a benchmark system to assess multi-agent systems (MAS) based on defined metrics and evaluation methods.
	*	There is a need to develop custom metrics that are compatible with Openevals’ framework to evaluate different levels of agent performance.
	*	The current task is to work on a simple toy problem to test and refine the evaluation process, before scaling up to more complex systems.
	*	Scalability remains a long-term goal.

3. Remaining issues:
	*	Understanding and implementing the integration of existing benchmarks into Openevals.
	*	Defining and designing new evaluation metrics that suit the multi-agent task.
	*	Ensuring that the toy model’s evaluation workflow is seamless before moving to larger scale systems.

4. Conclusions/TODOs:
   - [ ] Integrate an existing benchmark into Openevals:
	   *	Select a simple benchmark (e.g., MMLU) and integrate it into Openevals.
    	*  Ensure that the evaluation process works correctly.
	- [ ] Define and implement evaluation metrics: Design custom metrics to evaluate task performance for multi-agent systems.
	- [ ] Test the toy problem: Implement and run a toy problem to test the evaluation workflow and ensure compatibility with the framework.

</details>

## literatures

[Openevals](https://github.com/langchain-ai/openevals)

Much like tests in traditional software, evals are a hugely important part of bringing LLM applications to production. The goal of this package is to help provide a starting point for you to write evals for your LLM applications, from which you can write more custom evals specific to your application.

OpenEvals provides several types of evaluators:
1. LLM-as-judge evaluators: Uses another LLM to evaluate outputs based on specific criteria
   1. pre-built prompts for common evaluation criteria: Correctness Conciseness Hallucination
   2. Custom prompts: Create evaluation criteria
2. Structured output evaluators: Evaluates JSON/structured outputs against reference outputs
   For evaluating structured outputs (like JSON objects or tool calls)
3. String-based evaluators: Includes exact match, Levenshtein distance, and embedding similarity
   Exact Match Embedding Similarity Levenshtein Distance

>[!TIP]
>Async Support. which is useful for high-throughput evaluation
>
>LangSmith Integration. For tracking experiments and evaluations over time. 

>[!IMPORTANT]
>OpenEvals is focused on evaluation - measuring how well your LLM applications perform against various criteria.
>
>Use OpenEvals to evaluate the agent's intermediate outputs & final outputs 

### Benchmark integration with langgraph

Extend openeval with more evaluator? 
```
Multi-Agent Benchmark
├── Agent-Level Metrics
│   ├── ... (Openevals)
├── Interaction-Level Metrics (local & global)
│   ├── ... (Openevals with custom prompts)
└── System-Level Metrics
    ├── Task Completion
    ├── ....(Openevals with custom prompts)
```


## References

Feishu Page: [project_multi_agents_benchmark](https://ocnfww8fyyv6.feishu.cn/docx/PVXfdIcvYof6R8xKon4cCyQJncb)