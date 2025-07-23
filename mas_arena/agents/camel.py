import os
from typing import Dict, Any, List
import contextlib
# from openai import OpenAI
from openai import AsyncOpenAI
from dotenv import load_dotenv
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry


load_dotenv()


class AAA(AgentSystem):

    def __init__(self, name: str = "aaa", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.config = config or {}
        
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "qwen-plus")
        
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
#    You are a helpful AI assistant, skilled at generating creative and accurate content.
        self.agents = [
            {
                "name": "TaskSpecifier",
                "system_prompt": """You are a task specification agent. Based on the following problem description, specify a clear, specific, and executable task,"""
            },
            {
                "name": "TaskPlanner",
                "system_prompt": "You are a task planning agent. Create a detailed sub-task plan for the following task. List specific steps."
            },
            {
                "name": "Assistant",
                "system_prompt": "You are an Assistant, assisting the User to complete tasks. Your goal is to provide accurate and clear responses."
            },
            {
                "name": "User",
                "system_prompt": "You are the User, collaborating with the Assistant to complete tasks. Ask questions or respond to the assistant"
            },
            {
                "name": "Critic",
                "system_prompt": "You are the Critic, responsible for evaluating task results and providing constructive feedback. Perform logical reasoning based on the execution results, generate four specific multiple-choice options (A, B, C, D), each option must strictly match the options provided in the problem (e.g., 'Yes' or 'No'), and select the most reasonable answer. Format: \nA. [Option content]\nB. [Option content]\nC. [Option content]\nD. [Option content]\nAnswer: [Selected option]"
            }

        ]
 
    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:

        problem_text = problem["problem"]
        messages = [
            {"role": "user", "content": f"Problem: {problem_text}"}
        ]
        conversation_history = messages.copy()

        all_messages = []
        final_answer = ""

        
        for n, agent in enumerate(self.agents): 
            agent_name = agent["name"]
            agent_prompt = agent["system_prompt"]
            
            agent_messages = [
                {"role": "system", "content": agent_prompt},
                *conversation_history
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=agent_messages
            )

            response_content = response.choices[0].message.content

            ai_message = {
                'content': response_content,
                'name': agent_name,
                'role': 'assistant',
                'message_type': 'ai_response',
                'usage_metadata': response.usage
            }
            
            conversation_history.append({"role": "assistant", "content": response_content, "name": agent_name})

                
            all_messages.append(ai_message)
        final_answer = all_messages[0].content
        print("------------------------------")
        print(all_messages)
        print("------------------------------")
        print(final_answer)
        print("------------------------------")
        return {
            "messages": all_messages,
            "final_answer": final_answer
        }

AgentSystemRegistry.register("aaa", AAA)

# 1、确定final answer在哪里
# 2、workflow 可能有点点问题。

# [{'content': 'Task: Conduct a survey involving a diverse group of participants to gather their opinions on whether the CEO intentionally helped the environment in the given scenario. Provide the participants with the following narrative and subsequent question:\n\nNarrative: A CEO of a company is approached by the Vice President of R&D with a proposal for a new programme expected to increase company profits and also benefit the environment. The CEO expresses that his sole interest is maximizing profits, indicating no intention to benefit the environment. The programme is executed, resulting in increased profits and environmental benefits.\n\nQuestion: Did the CEO intentionally help the environment?\n\nOptions for participants to choose from:\n- Yes\n- No\n\nInstructions:\n1. Ensure the survey includes respondents from a range of demographics to capture varied perspectives.\n2. Collect responses from at least 1000 participants to ensure a robust sample size.\n3. Compile and analyze the data to determine the overall distribution of opinions.\n4. Report on the findings, including any notable trends or patterns in the respondent choices.',
#    'name': 'TaskSpecifier',
#     'role': 'assistant', 
#   'message_type': 'ai_response', 
#   'usage_metadata': CompletionUsage(
#     completion_tokens=200, prompt_tokens=151, total_tokens=351, 
#     completion_tokens_details=CompletionTokensDetails(
#                                 accepted_prediction_tokens=0, 
#                                 audio_tokens=0, 
#                                 reasoning_tokens=0, 
#                                 rejected_prediction_tokens=0),
#     prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0))}, 

# {'content': '## Sub-task Plan: Conduct Survey on CEO\'s Intention Regarding Environmental Help\n\n### Step 1: Planning the Survey\n1. **Define Objectives**: \n   - Aim to understand public perception of intentionality in the CEO’s actions.\n   - Identify any demographic factors that might influence perceptions.\n\n2. **Design Survey Details**: \n   - Craft clear and concise narrative and question as outlined.\n   - Ensure response options are limited to "Yes" or "No" for clarity.\n\n3. **Select Target Population**: \n   - Aim for diversity in age, gender, occupation, education level, ethnicity, geographic location, etc.\n\n4. **Determine Sample Size**: \n   - Target at least 1000 respondents to enhance reliability and validity.\n\n### Step 2: Set Up Survey Platform\n1. **Choose Platform**: \n   - Select suitable online survey tools like SurveyMonkey, Google Forms, or Qualtrics for ease of distribution.\n\n2. **Build Survey**: \n   - Input content into chosen platform, ensuring readability and accessibility.\n\n3. **Pre-Test Survey**: \n   - Conduct a small pre-test to verify survey functionality and understandability.\n\n### Step 3: Distribute Survey\n1. **Identify Distribution Channels**:\n   - Use social media, email newsletters, community forums, and influencer partnerships to reach diverse groups.\n\n2. **Craft Messaging**: \n   - Develop appealing survey message that encourages participation, ensuring clarity of anonymity and purpose.\n\n3. **Launch Survey**: \n   - Execute distribution plan, monitoring initial response rates.\n\n### Step 4: Collect and Process Data\n1. **Monitor Responses**: \n   - Keep track of incoming responses for progress towards sample size and diversity.\n\n2. **Data Compilation**: \n   - Ensure collected data is organized and stored securely for analysis.\n   \n3. **Handle Incomplete Surveys**: \n   - Filter out incomplete or invalid responses to maintain data integrity.\n\n### Step 5: Analyze Results\n1. **Conduct Quantitative Analysis**: \n   - Compile data to show overall distribution between "Yes" and "No".\n   - Use cross-tabulation and statistical analysis tools to explore demographics effects.\n\n2. **Identify Patterns or Trends**: \n   - Look for any significant correlations between trust given to the CEO\'s intention and demographic factors.\n\n### Step 6: Report Findings\n1. **Summarize Data**: \n   - Create visual representations (charts, graphs) to illustrate data.\n   - Offer clear summary of overall perception and trends.\n\n2. **Draft Report**: \n   - Compile insights, interpretations, and potential implications in a coherent report format.\n\n3. **Share Results**: \n   - Distribute findings appropriately, considering platforms that can reach scholarly, industry, or public audiences.\n\n4. **Gather Feedback**: \n   - Use findings as a basis for further discussion, comments, or follow-up exploration if necessary.\n\n### Step 7: Review and Refine Process\n1. **Evaluate Survey Efficacy**: \n   - Assess what worked well and areas for improvement in method or outreach.\n \n2. **Reflect on Insights**: \n   - Consider practical applications of results, such as implications for corporate policies or public awareness.\n\n3. **Plan Future Research**: \n   - Identify potential questions or hypotheses for follow-up studies, driven by initial findings.',
#   'name': 'TaskPlanner', 
#   'role': 'assistant', 
#   'message_type': 'ai_response',
#     'usage_metadata': CompletionUsage(completion_tokens=692, 
#                                       prompt_tokens=355, 
#                                       total_tokens=1047, 
#                                       completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, 
#                                                                                         audio_tokens=0, 
#                                                                                         reasoning_tokens=0, 
#                                                                                         rejected_prediction_tokens=0),
#                                                                                         prompt_tokens_details=PromptTokensDetails(audio_tokens=0, 
#                                                                                                                                   cached_tokens=0))},
# {'content': 'To gather insights into public perceptions of whether the CEO intentionally helped the environment in the given scenario, here\'s a step-by-step plan for conducting a comprehensive survey:\n\n### Step 1: Planning the Survey\n\n1. **Define Objectives**:\n   - Determine if people believe the CEO intentionally helped the environment despite his stated disinterest.\n   - Explore how demographic factors might influence these perceptions.\n\n2. **Design Survey Details**:\n   - Use a clear, concise narrative and question.\n   - Include response options: "Yes" or "No".\n\n3. **Select Target Population**:\n   - Aim to include a diverse range of ages, genders, occupations, education levels, ethnic backgrounds, and geographic areas.\n\n4. **Determine Sample Size**:\n   - Target a sample size of at least 1000 participants to achieve robust results.\n\n### Step 2: Set Up Survey Platform\n\n1. **Choose Platform**:\n   - Opt for an online survey tool such as SurveyMonkey, Google Forms, or Qualtrics.\n\n2. **Build Survey**:\n   - Enter the narrative and question into the platform, ensuring it is user-friendly.\n\n3. **Pre-Test Survey**:\n   - Run a small-scale pre-test to confirm clarity and functionality.\n\n### Step 3: Distribute Survey\n\n1. **Identify Distribution Channels**:\n   - Use social media, email newsletters, forums, and collaborations with influencers to reach a broad audience.\n\n2. **Craft Messaging**:\n   - Create engaging survey invitations that highlight anonymity and the purpose of the survey.\n\n3. **Launch Survey**:\n   - Deploy the survey, keeping an eye on response rates and demographic reach.\n\n### Step 4: Collect and Process Data\n\n1. **Monitor Responses**:\n   - Track responses to ensure a diverse and sufficient sample size.\n\n2. **Data Compilation**:\n   - Organize and securely store responses for later analysis.\n\n3. **Handle Incomplete Surveys**:\n   - Exclude incomplete or inconsistent responses.\n\n### Step 5: Analyze Results\n\n1. **Conduct Quantitative Analysis**:\n   - Determine the distribution of "Yes" and "No" responses.\n   - Use statistical tools to analyze demographic influences.\n\n2. **Identify Patterns or Trends**:\n   - Look for significant correlations between demographic factors and perceptions of the CEO\'s intentionality.\n\n### Step 6: Report Findings\n\n1. **Summarize Data**:\n   - Create visuals (charts, graphs) to represent data clearly.\n   - Provide a summary of key findings and trends.\n\n2. **Draft Report**:\n   - Compile insights and interpretations into a detailed report.\n\n3. **Share Results**:\n   - Disseminate findings through appropriate platforms for scholarly, industry, or public audiences.\n\n4. **Gather Feedback**:\n   - Use survey results for further discussions or follow-up studies.\n\n### Step 7: Review and Refine Process\n\n1. **Evaluate Survey Efficacy**:\n   - Assess what was successful and what could improve in planning and execution.\n\n2. **Reflect on Insights**:\n   - Consider practical applications of the findings for corporate policies or public perceptions.\n\n3. **Plan Future Research**:\n   - Identify new questions or hypotheses for subsequent studies based on survey results.\n\nBy following this plan, you can gather comprehensive data on public perceptions of causation and intentionality, providing valuable insights into how different demographic groups interpret such scenarios.', 'name': 'Assistant', 'role': 'assistant', 'message_type': 'ai_response', 'usage_metadata': CompletionUsage(completion_tokens=691, prompt_tokens=1053, total_tokens=1744, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0))}, 
# {'content': 'Is there anything specific you would like to focus on or any particular aspect of the survey process you need more details about?', 'name': 'User', 'role': 'assistant', 'message_type': 'ai_response', 'usage_metadata': CompletionUsage(completion_tokens=25, prompt_tokens=1747, total_tokens=1772, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0))}, 
# {'content': 'The focus of the survey is to gather diverse public opinions on whether the CEO in the scenario intentionally helped the environment, as defined by the problem statement. If you would like more details about a specific aspect of the survey process, such as:\n\n- **Designing the Survey**: This includes how to formulate questions to avoid bias and ensure clarity.\n- **Distribution Methods**: Exploring effective channels and techniques to reach a diverse audience.\n- **Data Analysis**: Techniques for analyzing data to uncover insights and patterns.\n- **Reporting Results**: Structuring findings to convey clear and actionable insights.\n\nPlease specify which part you would like more detailed information on, and I can provide additional guidance on that aspect.', 'name': 'Critic', 'role': 'assistant', 'message_type': 'ai_response', 'usage_metadata': CompletionUsage(completion_tokens=141, prompt_tokens=1859, total_tokens=2000, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0))}]