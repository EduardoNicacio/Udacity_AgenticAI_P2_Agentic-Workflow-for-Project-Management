# As per todo 1 - Import all required libraries, including the ActionPlanningAgent
from workflow_agents.base_agents import ActionPlanningAgent
import os

# As per todo 2 - Load environment variables and define the openai_api_key variable with your OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

knowledge = """
# Fried Egg
1. Heat pan with oil or butter
2. Crack egg into pan
3. Cook until white is set (2-3 minutes)
4. Season with salt and pepper
5. Serve

# Scrambled Eggs
1. Crack eggs into a bowl
2. Beat eggs with a fork until mixed
3. Heat pan with butter or oil over medium heat
4. Pour egg mixture into pan
5. Stir gently as eggs cook
6. Remove from heat when eggs are just set but still moist
7. Season with salt and pepper
8. Serve immediately

# Boiled Eggs
1. Place eggs in a pot
2. Cover with cold water (about 1 inch above eggs)
3. Bring water to a boil
4. Remove from heat and cover pot
5. Let sit: 4-6 minutes for soft-boiled or 10-12 minutes for hard-boiled
6. Transfer eggs to ice water to stop cooking
7. Peel and serve
"""

# As per todo 3 - Instantiate the ActionPlanningAgent, passing the openai_api_key and the knowledge variable
agent = ActionPlanningAgent(openai_api_key=OPENAI_API_KEY, knowledge=knowledge)

# As per todo 4 - Print the agent's response to the following prompt: "One morning I wanted to have scrambled eggs"
prompt = "One morning I wanted to have scrambled eggs"
print(f"Prompt: {prompt}")

action_planning_response = agent.extract_steps_from_prompt(prompt)

# Print the agent's response
print("Agent's response to the prompt:")
print("\n".join(action_planning_response))

# Creates the test_output directory if it does not exist
os.makedirs("test_output", exist_ok=True)

with open("test_output/action_planning_agent_response.txt", "w") as file:
    file.write(f"Prompt: \n{prompt}\n")
    file.write("\n")
    file.write("Response: \n")
    file.write("\n".join(action_planning_response) + "\n")
