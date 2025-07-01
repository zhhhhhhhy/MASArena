"""
LLM Debate Multi-Agent System

This module implements a multi-agent debate system where multiple LLM agents
engage in multi-round discussions to collaboratively solve problems through debate.
"""

import os
from typing import Dict, Any, List
import contextlib
from openai import OpenAI
from dotenv import load_dotenv
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

# Load environment variables
load_dotenv()


class LLMDebate(AgentSystem):
    """
    LLM Debate Multi-Agent System
    
    This system implements a multi-round debate mechanism where multiple LLM agents
    discuss and refine their answers through iterative rounds of debate.
    """

    def __init__(self, name: str = "llm_debate", config: Dict[str, Any] = None):
        """Initialize the LLM Debate System"""
        super().__init__(name, config)
        self.config = config or {}
        
        # Configuration parameters
        self.agents_num = self.config.get("agents_num", 3)  # Number of debate agents
        self.rounds_num = self.config.get("rounds_num", 2)  # Number of debate rounds
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "qwen-plus")
        
        # System prompt with format requirements
        self.base_system_prompt = self.config.get("system_prompt", "You are a helpful AI assistant.")
        self.system_prompt = self.base_system_prompt + "\n" + self.format_prompt
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), 
            base_url=os.getenv("OPENAI_API_BASE")
        )

    def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the LLM Debate system on a given problem.
        
        Args:
            problem: Dictionary containing the problem data
            
        Returns:
            Dictionary of run results including all agent messages and final answer
        """
        query = problem["problem"]
        
        # Initialize agent contexts - each agent starts with the same query
        initial_message = {
            "role": "user", 
            "content": f"{query} Make sure to state your answer at the end of the response."
        }
        agent_contexts = [[initial_message] for _ in range(self.agents_num)]
        all_messages = []
        
        # Multi-round debate
        for round_idx in range(self.rounds_num):
            round_messages = []
            
            for agent_idx, agent_context in enumerate(agent_contexts):
                # For rounds after the first, add other agents' opinions
                if round_idx > 0:
                    # Get contexts from other agents (excluding current agent)
                    other_contexts = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                    debate_message = self._construct_debate_message(other_contexts, query, round_idx)
                    agent_context.append(debate_message)
                
                # Get response from current agent
                response = self._call_llm(agent_context)
                
                # Create response message with usage metadata
                ai_message = {
                    'content': response['content'],
                    'name': f'debate_agent_{agent_idx+1}',
                    'role': 'assistant',
                    'message_type': 'ai_response',
                    'round': round_idx + 1,
                    'agent_id': agent_idx + 1,
                    'usage_metadata': response['usage']
                }
                
                # Add response to agent's context
                agent_context.append({"role": "assistant", "content": response['content']})
                round_messages.append(ai_message)
            
            all_messages.extend(round_messages)
        
        # Extract final answers from each agent
        final_answers = [context[-1]['content'] for context in agent_contexts]
        
        # Aggregate all answers into final result
        aggregated_answer = self._aggregate_answers(query, final_answers)
        
        # Create aggregation message
        aggregation_message = {
            'content': aggregated_answer['content'],
            'name': 'debate_aggregator',
            'role': 'assistant',
            'message_type': 'aggregation',
            'usage_metadata': aggregated_answer['usage']
        }
        all_messages.append(aggregation_message)
        
        return {
            "messages": all_messages,
            "final_answer": aggregated_answer['content'],
            "agent_responses": final_answers,
            "rounds_completed": self.rounds_num,
            "agents_participated": self.agents_num
        }

    def _call_llm(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Call the LLM with given messages and return response with usage metadata.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Dictionary containing response content and usage metadata
        """
        # Prepare messages for API call
        api_messages = [{"role": "system", "content": self.system_prompt}] + messages
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=api_messages
            )
            
            # Extract and clean content
            content = response.choices[0].message.content
            content = content.replace('\r\n', '\n').replace('\r', '\n').strip()
            with contextlib.suppress(UnicodeDecodeError):
                content = content.encode('utf-8').decode('utf-8-sig')  # Remove BOM
            
            return {
                'content': content,
                'usage': response.usage
            }
            
        except Exception as e:
            # Fallback response in case of API error
            return {
                'content': f"Error calling LLM: {str(e)}",
                'usage': None
            }

    def _construct_debate_message(self, other_agent_contexts: List[List[Dict]], 
                                query: str, round_idx: int) -> Dict[str, str]:
        """
        Construct a message containing other agents' opinions for the current agent.
        
        Args:
            other_agent_contexts: List of other agents' conversation contexts
            query: Original query/problem
            round_idx: Current round index
            
        Returns:
            Message dictionary containing other agents' opinions
        """
        # Handle case with no other agents (single agent introspection)
        if len(other_agent_contexts) == 0:
            return {
                "role": "user", 
                "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."
            }
        
        # Build message with other agents' recent responses
        prefix_string = "These are the recent/updated opinions from other agents: "
        
        for agent_idx, agent_context in enumerate(other_agent_contexts):
            # Get the most recent assistant response (last response in context)
            if len(agent_context) > 1:  # Ensure there's at least one response
                # Find the latest assistant message
                latest_response = None
                for msg in reversed(agent_context):
                    if msg.get("role") == "assistant":
                        latest_response = msg["content"]
                        break
                
                if latest_response:
                    response_text = f"\n\nAgent {agent_idx + 1} response: ```{latest_response}```"
                    prefix_string += response_text
        
        # Add instruction for using the opinions
        suffix_string = (
            f"\n\nUse these opinions carefully as additional advice, can you provide an updated answer? "
            f"Make sure to state your answer at the end of the response. "
            f"\nThe original problem is: {query}"
        )
        
        return {
            "role": "user", 
            "content": prefix_string + suffix_string
        }

    def _aggregate_answers(self, query: str, answers: List[str]) -> Dict[str, Any]:
        """
        Aggregate all agents' final answers into a single result.
        
        Args:
            query: Original query/problem
            answers: List of final answers from all agents
            
        Returns:
            Dictionary containing aggregated answer and usage metadata
        """
        # Build aggregation prompt
        aggregate_instruction = f"Task:\n{query}\n\n"
        
        for i, answer in enumerate(answers):
            aggregate_instruction += f"Solution {i+1}:\n{answer}\n\n"
        
        aggregate_instruction += (
            "Given all the above solutions, reason over them carefully and provide a final answer to the task. "
            f"Make sure to follow these requirements:\n{self.format_prompt}"
        )
        
        # Call LLM for aggregation
        messages = [{"role": "user", "content": aggregate_instruction}]
        return self._call_llm(messages)


# Register the agent system
AgentSystemRegistry.register("llm_debate", LLMDebate, 
                           agents_num=3, rounds_num=2) 