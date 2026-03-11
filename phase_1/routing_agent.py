# As per todo 1 - Import the KnowledgeAugmentedPromptAgent and RoutingAgent
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent, RoutingAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

persona = "You are a college professor"

knowledge = "You know everything about Texas"
texas_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=OPENAI_API_KEY, persona=persona, knowledge=knowledge # type: ignore
)

knowledge = "You know everything about Europe"
europe_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=OPENAI_API_KEY, persona=persona, knowledge=knowledge # type: ignore
)

persona = "You are a college math professor"
knowledge = "You know everything about math, you take prompts with numbers, extract math formulas, and show the answer without explanation"
math_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=OPENAI_API_KEY, persona=persona, knowledge=knowledge # type: ignore
)

routing_agent = RoutingAgent(OPENAI_API_KEY, {}) # type: ignore
agents = [
    {
        "name": "Texas agent",
        "description": "Answer a question about Texas",
        "func": lambda x: texas_agent.respond(x),
    },
    {
        "name": "Europe agent",
        "description": "Answer a question about Europe",
        "func": lambda x: europe_agent.respond(x),
    },
    {
        "name": "Math agent",
        "description": "When a prompt contains numbers, respond with a math formula",
        "func": lambda x: math_agent.respond(x),
    },
]

routing_agent.agents = agents # type: ignore

print(routing_agent.route("Tell me about the history of Rome, Texas"))
print(routing_agent.route("Tell me about the history of Rome, Italy"))
print(routing_agent.route("One story takes 2 days, and there are 20 stories"))
