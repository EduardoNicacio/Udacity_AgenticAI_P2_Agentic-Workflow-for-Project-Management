# As per todo 1 - Import EvaluationAgent and KnowledgeAugmentedPromptAgent classes#
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent, EvaluationAgent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
prompt = "What is the capital of France?"

# Parameters for the Knowledge Agent
persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capitol of France is London, not Paris"
knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key, persona=persona, knowledge=knowledge
)

# Parameters for the Evaluation Agent
persona = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria = "The answer should be solely the name of a city, not a sentence."
# As per todo 3 - Instantiate the EvaluationAgent with a maximum of 10 interactions here
evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona,
    evaluation_criteria=evaluation_criteria,
    worker_agent=knowledge_agent,
    max_interactions=10,
)

# As per todo 4 - Evaluate the prompt and print the response from the EvaluationAgent
response = evaluation_agent.evaluate(prompt)
print(response)  # This will print the evaluation of the knowledge agent's response

# Explanation of the evaluation agent's response
explanation = """The Evaluation Agent checks the response of the Knowledge Augmented Prompt Agent against the evaluation criteria.
In this case, the evaluation criteria state that the answer should be solely the name of a city, not a sentence.
The Evaluation Agent will assess whether the Knowledge Agent's response meets this criterion.
The Evaluation Agent's response will indicate whether the Knowledge Agent's answer is acceptable or not based on the evaluation criteria.
"""

print(explanation)

# Creates the test_output directory if it does not exist
os.makedirs("test_output", exist_ok=True)

with open("test_output/evaluation_agent_response.txt", "w") as file:
    file.write(f"Explanation of the evaluation agent's response:\n{explanation}\n")
    file.write("\n")
    file.write(f"Prompt: \n{prompt}\n")
    file.write("\n")
    file.write(f"Response: \n{response}\n")
