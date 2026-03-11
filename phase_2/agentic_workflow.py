"""
Product Specification Workflow Script
------------------------------------

This script orchestrates a multi-agent workflow that transforms a product specification into concrete development tasks.
It uses four types of agents:

1. **ActionPlanningAgent** - parses the user prompt and produces high-level steps.
2. **KnowledgeAugmentedPromptAgent** - generates content (user stories, features, or tasks) based on a persona and knowledge base.
3. **EvaluationAgent** - validates the output of a Knowledge agent against predefined criteria.
4. **RoutingAgent** - routes each step to the appropriate support function.

The workflow proceeds as follows:

* The user supplies a prompt describing what they want (e.g., “What would the development tasks for this product be?”).
* `ActionPlanningAgent` extracts actionable steps from that prompt.
* For each step, `RoutingAgent` determines which role should handle it and calls the corresponding support function.
* Each support function obtains a response from its Knowledge agent and then validates it with its Evaluation agent.
* The final validated output is printed.

The script is intended to be run as a standalone module. All configuration (API keys, file paths) is loaded from environment variables or local files.
"""

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------

from workflow_agents.base_agents import (
    ActionPlanningAgent,
    KnowledgeAugmentedPromptAgent,
    EvaluationAgent,
    RoutingAgent,
)

import os
import sys
import logging
from dotenv import load_dotenv

# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,  # INFO is sufficient; DEBUG can be enabled if needed
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/agentic_workflow.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Environment / constants
# ------------------------------------------------------------------

load_dotenv()

# Retrieve OpenAI API key from environment variables; sets constants for Vocareum Uri and Model name
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logger.error("Missing required environment variable: OPENAI_API_KEY")
    sys.exit(1)

# ------------------------------------------------------------------
# Load product spec document
# ------------------------------------------------------------------

try:
    with open("Product-Spec-Email-Router.txt", "r") as file:
        product_spec = file.read()
except FileNotFoundError:
    logger.exception(
        "Product specification file 'Product-Spec-Email-Router.txt' not found."
    )
    sys.exit(1)
except PermissionError:
    logger.exception("Permission denied while reading 'Product-Spec-Email-Router.txt'.")
    sys.exit(1)

# ------------------------------------------------------------------
# Instantiate all the agents
# ------------------------------------------------------------------

try:
    # Action Planning Agent
    knowledge_action_planning = (
        "Stories are defined from a product spec by identifying a "
        "persona, an action, and a desired outcome for each story. "
        "Each story represents a specific functionality of the product "
        "described in the specification. \n"
        "Features are defined by grouping related user stories. \n"
        "Tasks are defined for each story and represent the engineering "
        "work required to develop the product. \n"
        "A development Plan for a product contains all these components"
    )
    action_planning_agent = ActionPlanningAgent(
        openai_api_key=OPENAI_API_KEY, knowledge=knowledge_action_planning
    )

    # Product Manager - Knowledge Augmented Prompt Agent
    persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
    knowledge_product_manager = (
        "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
        "The sentences always start with: As a "
        "Write several stories for the product spec below, where the personas are the different users of the product. "
        f"{product_spec}"
    )
    product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
        openai_api_key=OPENAI_API_KEY,
        persona=persona_product_manager,
        knowledge=knowledge_product_manager,
    )

    # Product Manager - Evaluation Agent
    persona_product_manager_eval = (
        "You are an evaluation agent that checks the answers of other worker agents."
    )
    evaluation_criteria_product_manager = """
        The answer should be user stories that follow this structure:
        As a [type of user], I want [an action or feature] so that [benefit/value].
        Each story should be clear, concise, and focused on a specific user need or functionality.
    """
    product_manager_evaluation_agent = EvaluationAgent(
        openai_api_key=OPENAI_API_KEY,
        persona=persona_product_manager_eval,
        evaluation_criteria=evaluation_criteria_product_manager,
        worker_agent=product_manager_knowledge_agent,
        max_interactions=10,
    )

    # Program Manager - Knowledge Augmented Prompt Agent
    persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
    knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
    program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
        openai_api_key=OPENAI_API_KEY,
        persona=persona_program_manager,
        knowledge=knowledge_program_manager,
    )

    # Program Manager - Evaluation Agent
    persona_program_manager_eval = (
        "You are an evaluation agent that checks the answers of other worker agents."
    )
    program_manager_evaluation_agent = EvaluationAgent(
        openai_api_key=OPENAI_API_KEY,
        persona=persona_program_manager_eval,
        evaluation_criteria=(
            "The answer should be product features that follow the following structure: "
            "Feature Name: A clear, concise title that identifies the capability\n"
            "Description: A brief explanation of what the feature does and its purpose\n"
            "Key Functionality: The specific capabilities or actions the feature provides\n"
            "User Benefit: How this feature creates value for the user"
        ),
        worker_agent=program_manager_knowledge_agent,
        max_interactions=10,
    )

    # Development Engineer - Knowledge Augmented Prompt Agent
    persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
    knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."
    development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
        openai_api_key=OPENAI_API_KEY,
        persona=persona_dev_engineer,
        knowledge=knowledge_dev_engineer,
    )

    # Development Engineer - Evaluation Agent
    persona_dev_engineer_eval = (
        "You are an evaluation agent that checks the answers of other worker agents."
    )
    development_engineer_evaluation_agent = EvaluationAgent(
        openai_api_key=OPENAI_API_KEY,
        persona=persona_dev_engineer_eval,
        evaluation_criteria=(
            "The answer should be tasks following this exact structure: "
            "Task ID: A unique identifier for tracking purposes\n"
            "Task Title: Brief description of the specific development work\n"
            "Related User Story: Reference to the parent user story\n"
            "Description: Detailed explanation of the technical work required\n"
            "Acceptance Criteria: Specific requirements that must be met for completion\n"
            "Estimated Effort: Time or complexity estimation\n"
            "Dependencies: Any tasks that must be completed first"
        ),
        worker_agent=development_engineer_knowledge_agent,
        max_interactions=10,
    )
except Exception as exc:
    logger.exception("Failed to instantiate one or more agents.")
    sys.exit(1)

# ------------------------------------------------------------------
# Support functions for routing
# ------------------------------------------------------------------


def product_manager_support_function(query: str) -> str:
    """
    Generate and validate user stories for the given query.

    Parameters
    ----------
    query : str
        A step extracted by the ActionPlanningAgent that requires user story generation.

    Returns
    -------
    str
        The validated user stories produced by the Product Manager Knowledge agent.
    """
    print(f"Product Manager support function called with query: {query}")
    try:
        response = product_manager_knowledge_agent.respond(input_text=query)
        print(f"Knowledge agent response (PM): {response[:200]}...")  # type: ignore
    except Exception as exc:
        logger.exception("Error while generating user stories.")
        return f"[ERROR] Failed to generate user stories for query: {query}"

    try:
        evaluation = product_manager_evaluation_agent.evaluate(response)
        print(f"Evaluation result (PM): {evaluation[:200]}...")  # type: ignore
    except Exception as exc:
        logger.exception("Error while evaluating user stories.")
        return f"[ERROR] Failed to evaluate user stories for query: {query}"
    return evaluation  # type: ignore


def program_manager_support_function(query: str) -> str:
    """
    Generate and validate feature descriptions for the given query.

    Parameters
    ----------
    query : str
        A step extracted by the ActionPlanningAgent that requires feature extraction.

    Returns
    -------
    str
        The validated feature set produced by the Program Manager Knowledge agent.
    """
    print(f"Program Manager support function called with query: {query}")
    try:
        response = program_manager_knowledge_agent.respond(input_text=query)
        print(f"Knowledge agent response (ProgMgr): {response[:200]}...")  # type: ignore
    except Exception as exc:
        logger.exception("Error while generating features.")
        return f"[ERROR] Failed to generate features for query: {query}"

    try:
        evaluation = program_manager_evaluation_agent.evaluate(response)
        print(f"Evaluation result (ProgMgr): {evaluation[:200]}...")  # type: ignore
    except Exception as exc:
        logger.exception("Error while evaluating features.")
        return f"[ERROR] Failed to evaluate features for query: {query}"
    return evaluation  # type: ignore


def development_engineer_support_function(query: str) -> str:
    """
    Generate and validate development tasks for the given query.

    Parameters
    ----------
    query : str
        A step extracted by the ActionPlanningAgent that requires task extraction.

    Returns
    -------
    str
        The validated list of development tasks produced by the Development Engineer Knowledge agent.
    """
    print(f"Development Engineer support function called with query: {query}")
    try:
        response = development_engineer_knowledge_agent.respond(input_text=query)
        print(f"Knowledge agent response (DevEng): {response[:200]}...")  # type: ignore
    except Exception as exc:
        logger.exception("Error while generating tasks.")
        return f"[ERROR] Failed to generate tasks for query: {query}"

    try:
        evaluation = development_engineer_evaluation_agent.evaluate(response)
        print(f"Evaluation result (DevEng): {evaluation[:200]}...")  # type: ignore
    except Exception as exc:
        logger.exception("Error while evaluating tasks.")
        return f"[ERROR] Failed to evaluate tasks for query: {query}"
    return evaluation  # type: ignore


# ------------------------------------------------------------------
# Routing Agent
# ------------------------------------------------------------------

try:
    routing_agent = RoutingAgent(
        openai_api_key=OPENAI_API_KEY,
        agents=[
            {
                "name": "Product Manager",
                "description": "Routes to the Product Manager support function for user story extraction.",
                "func": product_manager_support_function,
            },
            {
                "name": "Program Manager",
                "description": "Routes to the Program Manager support function for feature extraction.",
                "func": program_manager_support_function,
            },
            {
                "name": "Development Engineer",
                "description": "Routes to the Development Engineer support function for task extraction.",
                "func": development_engineer_support_function,
            },
        ],
    )
except Exception as exc:
    logger.exception("Failed to instantiate RoutingAgent.")
    sys.exit(1)

# ------------------------------------------------------------------
# Run the workflow
# ------------------------------------------------------------------

print("Workflow execution started.")

workflow_prompt = "What would the development tasks for this product be?"
print(f"Workflow prompt: {workflow_prompt}")

try:
    # Extract steps from the prompt
    print("Extracting workflow steps using ActionPlanningAgent.")
    workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)

    if not workflow_steps:
        print("No workflow steps were extracted. Exiting.")
        sys.exit(0)

    completed_steps = []
    for step in workflow_steps:
        print(f"Executing step: {step}")
        try:
            result = routing_agent.route(step)
        except Exception as exc:
            logger.exception(f"Routing failed for step: {step}")
            result = f"[ERROR] Routing failed for step: {step}"
        completed_steps.append(result)
        print(f"Result of step '{step}': {result}")

    if completed_steps:
        final_output = completed_steps[-1]
        print(f"Final output: {final_output}")
        print("\n*** Workflow execution completed ***\n")
except Exception as exc:
    logger.exception("An unexpected error occurred during workflow execution.")
    sys.exit(1)
