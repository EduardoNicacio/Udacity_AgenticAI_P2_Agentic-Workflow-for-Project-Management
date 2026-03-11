from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"

persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"
knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key, persona=persona, knowledge=knowledge
)

# TODO: 3 - Write a print statement that demonstrates the agent using the provided knowledge rather than its own inherent knowledge.
knowledge_agent_response = knowledge_agent.respond(input_text=prompt)

print(knowledge_agent_response)

# Explanation of the agent's response
explanation = """The agent used the provided knowledge about the capital of France, which is stated as London, not Paris.
This indicates that the agent is following the instructions to use only the provided knowledge and not its own inherent knowledge, which would typically state that Paris is the capital of France.
The system prompt specifying the persona as a college professor influenced the agent's response to start with "Dear students," which adds a formal tone to the answer.
"""

print(explanation)

with open("test_output/knowledge_augmented_prompt_agent_response.txt", "w") as file:
    file.write(f"Explanation of the agent's response:\n{explanation}\n")
    file.write("\n")
    file.write(f"Prompt: \n{prompt}\n")
    file.write("\n")
    file.write(f"Response: \n{knowledge_agent_response}\n")
