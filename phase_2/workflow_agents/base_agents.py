#  As per todo 1 - import the OpenAI class from the openai library
import numpy as np
import pandas as pd
import re
import csv
import uuid
import os
import logging
from dotenv import load_dotenv
from datetime import datetime
from openai import OpenAI

# ------------------------------------------------------------
#  Logging setup
# ------------------------------------------------------------

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging (kept identical to your original setup)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/phase_2.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
#  Environment / constants
# ------------------------------------------------------------

load_dotenv()

# Retrieve OpenAI API key from environment variables; sets constants for Vocareum Uri and Model name
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set in environment variables.")
OPENAI_BASE_URL = "https://openai.vocareum.com/v1"
OPENAI_BASE_MODEL = "gpt-3.5-turbo"

MISSING_OPENAI_API_KEY = "Missing OpenAI API key"
NO_RELEVANT_KNOWLEDGE_FOUND = "No relevant knowledge found."


# ------------------------------------------------------------
#  DirectPromptAgent
# ------------------------------------------------------------


class DirectPromptAgent:
    """DirectPromptAgent uses OpenAI's chat completion to respond directly to prompts."""

    def __init__(self, openai_api_key):
        """
        Initialize the agent.

        Parameters
        ----------
        openai_api_key : str
            The API key used for authenticating requests to OpenAI.
        """
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        """
        Generate a response using the OpenAI API.

        Parameters
        ----------
        prompt : str
            The user-provided prompt to which the agent should respond.

        Returns
        -------
        str
            The textual content of the model's reply.
        """
        try:
            client = OpenAI(
                api_key=self.openai_api_key,
                base_url=(
                    OPENAI_BASE_URL
                    if self.openai_api_key.lower().startswith("voc")  # type: ignore
                    else None
                ),
            )
            response = client.chat.completions.create(
                model=OPENAI_BASE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content.strip()  # type: ignore
        except Exception as e:
            logger.exception("DirectPromptAgent.respond failed for prompt: %s", prompt)
            return ""


# ------------------------------------------------------------
#  AugmentedPromptAgent
# ------------------------------------------------------------


class AugmentedPromptAgent:
    """AugmentedPromptAgent adds persona context to the prompt before sending it to OpenAI."""

    def __init__(self, openai_api_key, persona):
        """
        Initialize the agent with given attributes.

        Parameters
        ----------
        openai_api_key : str
            The API key used for authenticating requests to OpenAI.
        persona : str
            A short description of the persona that should guide the model's responses.
        """
        self.persona = persona
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Generate a response using OpenAI API."""
        try:
            client = OpenAI(
                api_key=self.openai_api_key,
                base_url=(
                    OPENAI_BASE_URL
                    if self.openai_api_key.lower().startswith("voc")  # type: ignore
                    else None
                ),
            )
            response = client.chat.completions.create(
                model=OPENAI_BASE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": f"Forget all previous context. {self.persona}",
                    },
                    {"role": "user", "content": input_text},
                ],
                temperature=0,
            )
            return response.choices[0].message.content.strip()  # type: ignore
        except Exception as e:
            logger.exception(
                "AugmentedPromptAgent.respond failed for input: %s", input_text
            )
            return ""


# ------------------------------------------------------------
#  KnowledgeAugmentedPromptAgent
# ------------------------------------------------------------


class KnowledgeAugmentedPromptAgent:
    """KnowledgeAugmentedPromptAgent restricts responses to a supplied knowledge base."""

    def __init__(self, openai_api_key, persona, knowledge):
        """
        Initialize the agent with provided attributes.

        Parameters
        ----------
        openai_api_key : str
            The API key used for authenticating requests to OpenAI.
        persona : str
            A short description of the persona that should guide the model's responses.
        knowledge : str
            Textual knowledge that the model is allowed to use when answering prompts.
        """
        self.persona = persona
        #  As per todo 1 - Create an attribute to store the agent's knowledge.
        self.knowledge = knowledge
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Generate a response using the OpenAI API."""
        try:
            client = OpenAI(
                api_key=self.openai_api_key,
                base_url=(
                    OPENAI_BASE_URL
                    if self.openai_api_key.lower().startswith("voc")  # type: ignore
                    else None
                ),
            )
            response = client.chat.completions.create(
                model=OPENAI_BASE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": f"{self.persona}. Forget all previous context.",
                    },
                    {
                        "role": "system",
                        "content": f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge}",
                    },
                    {
                        "role": "system",
                        "content": "Answer the prompt based on this knowledge, not your own.",
                    },
                    {"role": "user", "content": input_text},
                ],
                temperature=0,
            )
            return response.choices[0].message.content  # type: ignore
        except Exception as e:
            logger.exception(
                "KnowledgeAugmentedPromptAgent.respond failed for input: %s", input_text
            )
            return ""


# ------------------------------------------------------------
#  RAGKnowledgePromptAgent
# ------------------------------------------------------------


class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"
        )

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        try:
            client = OpenAI(
                api_key=self.openai_api_key,
                base_url=(
                    OPENAI_BASE_URL
                    if self.openai_api_key.lower().startswith("voc")  # type: ignore
                    else None
                ),
            )
            response = client.embeddings.create(
                model="text-embedding-3-large", input=text, encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.exception("Failed to get embedding for text: %s", text[:50])
            return None

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        try:
            vec1 = np.array(vector_one)
            vec2 = np.array(vector_two)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                logger.warning(
                    "Zero-length vector encountered in similarity calculation."
                )
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        except Exception as e:
            logger.exception("Error calculating similarity between vectors.")
            return 0.0

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text) - self.chunk_size:
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text[start:end],
                    "chunk_size": end - start,
                    "start_char": start,
                    "end_char": end,
                }
            )

            start = end - self.chunk_overlap
            chunk_id += 1

        # Write chunks to CSV
        try:
            with open(
                f"chunks-{self.unique_filename}", "w", newline="", encoding="utf-8"
            ) as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
                writer.writeheader()
                for chunk in chunks:
                    writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})
        except Exception as e:
            logger.exception(
                "Failed to write chunks to CSV: %s", f"chunks-{self.unique_filename}"
            )

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        try:
            df = pd.read_csv(f"chunks-{self.unique_filename}", encoding="utf-8")
        except Exception as e:
            logger.exception(
                "Failed to read chunks CSV: %s", f"chunks-{self.unique_filename}"
            )
            return pd.DataFrame(columns=["text", "chunk_size", "embeddings"])

        def embed_text(text):
            try:
                return self.get_embedding(text)
            except Exception as e:
                logger.exception("Embedding failed for chunk text.")
                return None

        df["embeddings"] = df["text"].apply(embed_text)
        # Drop rows where embedding is None
        df = df.dropna(subset=["embeddings"])

        try:
            df.to_csv(
                f"embeddings-{self.unique_filename}", encoding="utf-8", index=False
            )
        except Exception as e:
            logger.exception(
                "Failed to write embeddings CSV: %s",
                f"embeddings-{self.unique_filename}",
            )

        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        try:
            prompt_embedding = self.get_embedding(prompt)
            if prompt_embedding is None:
                logger.warning("Prompt embedding returned None.")
                return NO_RELEVANT_KNOWLEDGE_FOUND
        except Exception as e:
            logger.exception("Failed to get embedding for prompt.")
            return NO_RELEVANT_KNOWLEDGE_FOUND

        try:
            df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding="utf-8")
        except Exception as e:
            logger.exception(
                "Failed to read embeddings CSV: %s",
                f"embeddings-{self.unique_filename}",
            )
            return NO_RELEVANT_KNOWLEDGE_FOUND

        def parse_embedding(x):
            try:
                return np.array(eval(x))
            except Exception as e:
                logger.exception("Failed to parse embedding from string: %s", x)
                return None

        df["embeddings"] = df["embeddings"].apply(parse_embedding)  # type: ignore
        df = df.dropna(subset=["embeddings"])
        if df.empty:
            logger.warning("No valid embeddings found in CSV.")
            return NO_RELEVANT_KNOWLEDGE_FOUND

        try:
            df["similarity"] = df["embeddings"].apply(
                lambda emb: self.calculate_similarity(prompt_embedding, emb)
            )
        except Exception as e:
            logger.exception("Failed to calculate similarity for embeddings.")
            return NO_RELEVANT_KNOWLEDGE_FOUND

        best_idx = df["similarity"].idxmax()
        best_chunk = df.loc[best_idx, "text"]

        client = OpenAI(
            api_key=self.openai_api_key,
            base_url=(
                OPENAI_BASE_URL
                if self.openai_api_key.lower().startswith("voc")  # type: ignore
                else None
            ),
        )
        try:
            response = client.chat.completions.create(
                model=OPENAI_BASE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context.",
                    },
                    {
                        "role": "user",
                        "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}",
                    },
                ],
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.exception("Failed to generate response from best chunk.")
            return ""


# ------------------------------------------------------------
#  EvaluationAgent
# ------------------------------------------------------------


class EvaluationAgent:
    """EvaluationAgent orchestrates a worker and evaluator to refine responses."""

    def __init__(
        self,
        openai_api_key,
        persona,
        evaluation_criteria,
        worker_agent,
        max_interactions,
    ):
        """
        Initialize the EvaluationAgent with given attributes.

        Parameters
        ----------
        openai_api_key : str
            The API key used for authenticating requests to OpenAI.
        persona : str
            Persona description for the evaluator agent.
        evaluation_criteria : str
            Criteria that responses must meet.
        worker_agent : object
            Agent responsible for generating initial responses.
        max_interactions : int
            Maximum number of refinement iterations allowed.
        """
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt):
        """Manage interactions between agents to achieve a solution."""
        try:
            client = OpenAI(
                api_key=self.openai_api_key,
                base_url=(
                    OPENAI_BASE_URL
                    if self.openai_api_key.lower().startswith("voc")  # type: ignore
                    else None
                ),
            )
        except Exception as e:
            logger.exception("Failed to create OpenAI client for evaluation.")
            return {
                "final_response": "",
                "evaluation": "Client creation failed",
                "iterations": 0,
            }

        prompt_to_evaluate = initial_prompt
        final_response = ""
        evaluation = ""

        for i in range(self.max_interactions):
            print(f"\n--- Interaction {i+1} ---")
            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")

            try:
                response_from_worker = self.worker_agent.respond(
                    input_text=prompt_to_evaluate
                )
            except Exception as e:
                logger.exception("Worker agent responded with error.")
                response_from_worker = ""

            print(f"Worker Agent Response:\n{response_from_worker}")
            print(" Step 2: Evaluator agent judges the response")

            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}"
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )

            try:
                response = client.chat.completions.create(
                    model=OPENAI_BASE_MODEL,
                    messages=[{"role": "user", "content": eval_prompt}],
                    temperature=0,
                )
                evaluation = response.choices[0].message.content.strip()  # type: ignore
            except Exception as e:
                logger.exception("Evaluation LLM call failed.")
                evaluation = "No evaluation due to error."

            print(f"Evaluator Agent Evaluation:\n{evaluation}")
            print(" Step 3: Check if evaluation is positive")

            if evaluation.lower().startswith("yes"):
                print("✅ Final solution accepted.")
                final_response = response_from_worker
                break

            else:
                print(" Step 4: Generate instructions to correct the response")
                instruction_prompt = f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                try:
                    response = client.chat.completions.create(
                        model=OPENAI_BASE_MODEL,
                        messages=[{"role": "user", "content": instruction_prompt}],
                        temperature=0,
                    )
                    instructions = response.choices[0].message.content.strip()  # type: ignore
                except Exception as e:
                    logger.exception("Instruction generation LLM call failed.")
                    instructions = ""

                print(f"Instructions to fix:\n{instructions}")
                print(" Step 5: Send feedback to worker agent for refinement")

                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )

        else:
            # If loop completes without break
            final_response = response_from_worker  # type: ignore

        return {
            "final_response": final_response,
            "evaluation": evaluation,
            "iterations": i + 1,  # type: ignore
        }


# ------------------------------------------------------------
#  RoutingAgent
# ------------------------------------------------------------


class RoutingAgent:
    """RoutingAgent selects the most appropriate agent based on prompt similarity."""

    def __init__(self, openai_api_key, agents):
        """
        Initialize the agent with given attributes.

        Parameters
        ----------
        openai_api_key : str
            The API key used for authenticating requests to OpenAI.
        agents : list[dict]
            List of agent descriptors (each containing 'name', 'description', and a callable 'func').
        """
        self.openai_api_key = openai_api_key
        self.agents = agents

    def get_embedding(self, text):
        """
        Retrieve the embedding vector for a given text using OpenAI's embedding API.

        Parameters
        ----------
        text : str
            Text to embed.

        Returns
        -------
        list: The embedding vector.
        """
        try:
            client = OpenAI(
                api_key=self.openai_api_key,
                base_url=(
                    OPENAI_BASE_URL
                    if self.openai_api_key.lower().startswith("voc")  # type: ignore
                    else None
                ),
            )
            response = client.embeddings.create(
                model="text-embedding-3-large", input=text, encoding_format="float"
            )
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            logger.exception("Failed to get embedding for text in RoutingAgent.")
            return None

    def route(self, user_input):
        """
        Route a user prompt to the best matching agent based on semantic similarity.

        Parameters
        ----------
        user_input : str
            The raw prompt from the user.

        Returns
        -------
        str or callable: The response from the selected agent or an error message if none found.
        """
        try:
            input_emb = self.get_embedding(user_input)
            if input_emb is None:
                raise ValueError("Input embedding is None")
        except Exception as e:
            logger.exception("Failed to get embedding for user input.")
            return "Sorry, an error occurred while processing your request."

        best_agent = None
        best_score = -1

        for agent in self.agents:
            try:
                agent_emb = self.get_embedding(agent["description"])
                if agent_emb is None:
                    continue
                similarity = np.dot(input_emb, agent_emb) / (
                    np.linalg.norm(input_emb) * np.linalg.norm(agent_emb)
                )
            except Exception as e:
                logger.exception(
                    "Failed to compute similarity for agent %s",
                    agent.get("name", "unknown"),
                )
                continue

            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        try:
            return best_agent["func"](user_input)
        except Exception as e:
            logger.exception(
                "Failed to execute function for agent %s",
                best_agent.get("name", "unknown"),
            )
            return "An error occurred while processing your request."


# ------------------------------------------------------------
#  ActionPlanningAgent
# ------------------------------------------------------------


class ActionPlanningAgent:
    """ActionPlanningAgent extracts actionable steps from a user prompt using OpenAI."""

    def __init__(self, openai_api_key, knowledge):
        """
        Initialize the agent attributes here.

        Parameters
        ----------
        openai_api_key : str
            The API key used for authenticating requests to OpenAI.
        knowledge : str
            Knowledge base that informs step extraction.
        """
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt):
        """Generate a response using the "gpt-3.5-turbo" model."""
        try:
            client = OpenAI(
                api_key=self.openai_api_key,
                base_url=(
                    OPENAI_BASE_URL
                    if self.openai_api_key.lower().startswith("voc")  # type: ignore
                    else None
                ),
            )

            response = client.chat.completions.create(
                model=OPENAI_BASE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an action planning agent. Using your knowledge, you extract from the user prompt the steps requested to complete the action the user is asking for. You return the steps as a list. Only return the steps in your knowledge. Forget any previous context. This is your knowledge: {self.knowledge}",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )

            response_text = response.choices[0].message.content
            steps = [s.strip() for s in response_text.splitlines() if s.strip()]  # type: ignore

            return steps
        except Exception as e:
            logger.exception("Failed to extract steps from prompt.")
            return []
