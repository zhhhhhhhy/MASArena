# Tasks
## Code Generation
HumanEval
MBPP	

## Instruction Following
IFEval	

## Math
AIME 2025
MATH	

## MMLU(Massive Multitask Language Understanding)
MMLU-Pro	

## BIG-Bench Hard

## QA
DROP

# Qwen
For single and MAS:
- no reasoning: `Qwen/Qwen3-32B`
  - Prompt $0.21 / 1M tokens
  - Completion $0.62 / 1M tokens
- reasoning: `Qwen/QwQ-32B-Preview`
  - Prompt $0.50 / 1M tokens
  - Completion $0.67 / 1M tokens


Smaller model for MAS specific. 
- `Qwen/Qwen3-14B`
  - Prompt $0.29 / 1M tokens
  - Completion $1.02 / 1M tokens
- `Qwen/Qwen3-8B`
  - Prompt $0.17 / 1M tokens
  - Completion $0.50 / 1M tokens

Note: for open source models, we won't use the price.

# DeepSeek

For single and MAS:
- no reasoning: `deepseek-ai/DeepSeek-V3`
  - Prompt $0.11 / 1M tokens
  - Completion $0.45 / 1M tokens
- reasoning: `deepseek-ai/DeepSeek-R1`
  - Prompt $0.23 / 1M tokens
  - Completion $0.91 / 1M tokens

Smaller model for MAS specific. 
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
  - Prompt $1.82 / 1M tokens
  - Completion $7.29 / 1M tokens
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
  - Prompt $0.91 / 1M tokens
  - Completion $3.65 / 1M tokens
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
  - Prompt $0.46 / 1M tokens
  - Completion $1.82 / 1M tokens

Note: for open source models, we won't use the price.

# Claude

For single and MAS:
- no reasoning: `claude-3-7-sonnet-latest`
  - Prompt $3 / 1M tokens
  - Completion $15 / 1M tokens
  - ttft: 2 
  - tokens/s: 75
- reasoning: `claude-3-7-sonnet-20250219-thinking`
  - Prompt $3 / 1M tokens
  - Completion $15 / 1M tokens
  - ttft: 14.2 
  - tokens/s: 88

Note: the price is from the offical website, not from the third party's price.

# OpenAI

For single and MAS:
- no reasoning: `gpt-4.1-mini`
  - Prompt $0.4 / 1M tokens
  - Completion $1.6 / 1M tokens
  - ttft: 0.4
  - tokens/s: 71
- reasoning: `o4-mini`
  - Prompt $ 4 / 1M tokens
  - Completion $16 / 1M tokens
  - ttft: 48.6
  - tokens/s: 152


(Optional) weaker model for MAS specific. 
- `gpt-3.5-turbo`
  - Prompt $0.50 / 1M tokens
  - Completion $1.50 / 1M tokens


# Gemini

For single and MAS:
- no reasoning: `gemini-2.5-flash-preview-04-17-nothinking`
  - Prompt $0.15 / 1M tokens
  - Completion $0.6 / 1M tokens
  - ttft: 0.4
  - tokens/s: 266
- reasoning: `gemini-2.5-flash-preview-04-17-thinking`
  - Prompt $0.15 / 1M tokens
  - Completion $3.5 / 1M tokens
  - ttft: 15.4
  - tokens/s: 350


(Optional) weaker model for MAS specific. 
- `gemini-2.0-flash`
  - Prompt $0.1 / 1M tokens
  - Completion $0.4 / 1M tokens
  - ttft: 0.38 
  - tokens/s: 240

# Gork

For single and MAS:
- no reasoning: `grok-3`
  - Prompt $3.00 / 1M tokens
  - Completion $15.00 / 1M tokens
  - ttft: 0.5
  - tokens/s: 59
- reasoning: `grok-3-think` (grok-3-mini)
  - Prompt $0.3 / 1M tokens
  - Completion $0.5 / 1M tokens
  - ttft: 26.5
  - tokens/s: 76

Reasoning is only supported by `grok-3-mini` and `grok-3-mini-fast`.
The Grok 3 models `grok-3` and `grok-3-fast` do not support reasoning.

Remark: seems unfair to compare reasoning with non-reasoning model of grok. 


# Notes: 

- For close source models, we refer the price from the official website. 
- For open source models, we won't use the price.
- For ttft and tokens/s, we refer to [this](https://artificialanalysis.ai/models/?models=gpt-4-1-mini%2Co4-mini%2Cgemini-2-0-flash%2Cgemini-2-5-flash%2Cgemini-2-5-flash-reasoning-04-2025%2Cclaude-3-7-sonnet%2Cclaude-3-7-sonnet-thinking%2Cgrok-3-mini-reasoning%2Cgrok-3).
- For ttft of reasoning mode, it is the time of the first answer token received; Accounts for Reasoning Model 'Thinking' time.


# Results output folder structure
```
results/
  ├── {task_category}_{task_name}/
      ├── {agent_system}/
          ├── {model_provider}_{model_name}/
              ├── reasoning/
              └── no_reasoning/

```