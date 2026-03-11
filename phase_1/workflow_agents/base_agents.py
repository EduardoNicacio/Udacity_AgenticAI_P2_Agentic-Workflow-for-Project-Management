# As per todo 1 - import the OpenAI class from the openai library
import numpy as np
import pandas as pd
import re
import csv
import uuid
import os
import logging
from dotenv import load_dotenv
from datetime import datetime
from openai import OpenAI  # the client class
import openai  # for exception types
import ast  # safe eval of embeddings

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
        logging.FileHandler("logs/phase_1.log"),
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

# ------------------------------------------------------------
#  Helper utilities (used by several classes)
# ------------------------------------------------------------


def _safe_get_embedding(client: OpenAI, text: str):
    """
    Wrapper around the embeddings endpoint that logs and returns None on failure.
    """
    try:
        response = client.embeddings.create(
            model="text-embedding-3-large", input=text, encoding_format="float"
        )
        return response.data[0].embedding
    except openai.OpenAIError as e:
        logger.exception("OpenAI API error while fetching embedding")
    except Exception as e:
        logger.exception("Unexpected error while fetching embedding")
    return None


# ------------------------------------------------------------
#  DirectPromptAgent
# ------------------------------------------------------------


class DirectPromptAgent:
    """
    A simple agent that forwards prompts directly to OpenAI's chat completion endpoint.
    """

    def __init__(self, openai_api_key: str):
        # Prefer the passed key; fall back to env var if missing.
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        if not self.openai_api_key:
            logger.error("OpenAI API key is required for DirectPromptAgent.")
            raise ValueError(MISSING_OPENAI_API_KEY)

    def respond(self, prompt: str) -> str:
        """
        Generate a response to the given prompt using the OpenAI chat completion endpoint.
        Errors are logged and an empty string is returned on failure.
        """
        try:
            client = OpenAI(
                api_key=self.openai_api_key,
                base_url=OPENAI_BASE_URL if self.openai_api_key.lower().startswith("voc") else None,  # type: ignore
            )
            response = client.chat.completions.create(
                model=OPENAI_BASE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content.strip()  # type: ignore
        except openai.OpenAIError as e:
            logger.exception("OpenAI API error in DirectPromptAgent.respond")
        except Exception as e:
            logger.exception("Unexpected error in DirectPromptAgent.respond")
        return ""


# ------------------------------------------------------------
#  AugmentedPromptAgent
# ------------------------------------------------------------


class AugmentedPromptAgent:
    """
    An agent that augments the user prompt with a persona before sending it to OpenAI.
    """

    def __init__(self, openai_api_key: str, persona: str):
        self.persona = persona
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        if not self.openai_api_key:
            logger.error("OpenAI API key is required for AugmentedPromptAgent.")
            raise ValueError(MISSING_OPENAI_API_KEY)

    def respond(self, input_text: str) -> str:
        try:
            client = OpenAI(
                api_key=self.openai_api_key,
                base_url=OPENAI_BASE_URL if self.openai_api_key.lower().startswith("voc") else None,  # type: ignore
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
        except openai.OpenAIError as e:
            logger.exception("OpenAI API error in AugmentedPromptAgent.respond")
        except Exception as e:
            logger.exception("Unexpected error in AugmentedPromptAgent.respond")
        return ""


# ------------------------------------------------------------
#  KnowledgeAugmentedPromptAgent
# ------------------------------------------------------------


class KnowledgeAugmentedPromptAgent:
    """
    An agent that uses a fixed knowledge base to answer prompts.
    """

    def __init__(self, openai_api_key: str, persona: str, knowledge: str):
        self.persona = persona
        self.knowledge = knowledge
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        if not self.openai_api_key:
            logger.error(
                "OpenAI API key is required for KnowledgeAugmentedPromptAgent."
            )
            raise ValueError(MISSING_OPENAI_API_KEY)

    def respond(self, input_text: str) -> str:
        try:
            client = OpenAI(
                api_key=self.openai_api_key,
                base_url=OPENAI_BASE_URL if self.openai_api_key.lower().startswith("voc") else None,  # type: ignore
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
        except openai.OpenAIError as e:
            logger.exception(
                "OpenAI API error in KnowledgeAugmentedPromptAgent.respond"
            )
        except Exception as e:
            logger.exception(
                "Unexpected error in KnowledgeAugmentedPromptAgent.respond"
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

    def __init__(
        self,
        openai_api_key: str,
        persona: str,
        chunk_size: int = 2000,
        chunk_overlap: int = 100,
    ):
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        if not self.openai_api_key:
            logger.error("OpenAI API key is required for RAGKnowledgePromptAgent.")
            raise ValueError(MISSING_OPENAI_API_KEY)
        self.unique_filename = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"
        )

    # ------------------------------------------------------------------
    #  Embedding helpers
    # ------------------------------------------------------------------

    def get_embedding(self, text: str):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.
        Returns None on failure.
        """
        try:
            client = OpenAI(
                api_key=self.openai_api_key,
                base_url=OPENAI_BASE_URL if self.openai_api_key.lower().startswith("voc") else None,  # type: ignore
            )
            return _safe_get_embedding(client, text)
        except Exception as e:
            logger.exception("Error in RAGKnowledgePromptAgent.get_embedding")
            return None

    def calculate_similarity(self, vector_one: list, vector_two: list) -> float:
        """
        Calculates cosine similarity between two vectors.
        Returns 0.0 if either vector is zero or an error occurs.
        """
        try:
            vec1 = np.array(vector_one)
            vec2 = np.array(vector_two)
            denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            if denom == 0:
                logger.warning("Zero magnitude encountered in similarity calculation.")
                return 0.0
            return float(np.dot(vec1, vec2) / denom)
        except Exception as e:
            logger.exception("Error calculating cosine similarity")
            return 0.0

    # ------------------------------------------------------------------
    #  Chunking & persistence helpers
    # ------------------------------------------------------------------

    def chunk_text(self, text: str):
        """
        Splits text into manageable chunks, attempting natural breaks.
        Writes the chunks to a CSV file for later processing.
        Returns the list of chunk dictionaries.
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

        # Persist chunks to CSV – log any I/O errors but continue.
        try:
            with open(
                f"chunks-{self.unique_filename}", "w", newline="", encoding="utf-8"
            ) as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
                writer.writeheader()
                for chunk in chunks:
                    writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})
        except OSError as e:
            logger.exception("Failed to write chunks CSV")

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.
        Returns the DataFrame containing text, chunk_size, and embeddings.
        """
        try:
            df = pd.read_csv(f"chunks-{self.unique_filename}", encoding="utf-8")
        except FileNotFoundError as e:
            logger.exception("Chunks file not found for embedding calculation.")
            return pd.DataFrame()
        except Exception as e:
            logger.exception("Error reading chunks CSV.")
            return pd.DataFrame()

        # Compute embeddings safely
        def embed_text(text):
            try:
                return self.get_embedding(text)
            except Exception:
                return None

        df["embeddings"] = df["text"].apply(embed_text)

        # Drop rows where embedding failed
        df.dropna(subset=["embeddings"], inplace=True)

        try:
            df.to_csv(
                f"embeddings-{self.unique_filename}", encoding="utf-8", index=False
            )
        except Exception as e:
            logger.exception("Failed to write embeddings CSV.")

        return df

    # ------------------------------------------------------------------
    #  Prompt handling
    # ------------------------------------------------------------------

    def find_prompt_in_knowledge(self, prompt: str) -> str:
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.
        Returns an empty string if any step fails.
        """
        try:
            prompt_embedding = self.get_embedding(prompt)
            if prompt_embedding is None:
                logger.error("Failed to obtain embedding for the user prompt.")
                return ""

            df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding="utf-8")
        except Exception as e:
            logger.exception("Error loading embeddings or obtaining prompt embedding.")
            return ""

        # Convert string representation of array back to numpy array safely
        def parse_embedding(x):
            try:
                return np.array(ast.literal_eval(x))
            except Exception:
                logger.warning(f"Failed to parse embedding: {x}")
                return None

        df["embeddings"] = df["embeddings"].apply(parse_embedding)  # type: ignore
        df.dropna(subset=["embeddings"], inplace=True)

        if df.empty:
            logger.error("No valid embeddings available for similarity search.")
            return ""

        try:
            df["similarity"] = df["embeddings"].apply(
                lambda emb: self.calculate_similarity(prompt_embedding, emb)
            )
        except Exception as e:
            logger.exception("Error calculating similarities.")
            return ""

        best_chunk = df.loc[df["similarity"].idxmax(), "text"]

        # Ask the model to answer based on the best chunk
        try:
            client = OpenAI(
                api_key=self.openai_api_key,
                base_url=OPENAI_BASE_URL if self.openai_api_key.lower().startswith("voc") else None,  # type: ignore
            )
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
            return response.choices[0].message.content  # type: ignore
        except openai.OpenAIError as e:
            logger.exception("OpenAI API error in find_prompt_in_knowledge")
        except Exception as e:
            logger.exception("Unexpected error in find_prompt_in_knowledge")
        return ""


# ------------------------------------------------------------
#  EvaluationAgent
# ------------------------------------------------------------


class EvaluationAgent:
    """
    An agent that evaluates responses from a worker agent against specified criteria
    and iteratively refines them.
    """

    def __init__(
        self,
        openai_api_key: str,
        persona: str,
        evaluation_criteria: str,
        worker_agent,
        max_interactions: int,
    ):
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        if not self.openai_api_key:
            logger.error("OpenAI API key is required for EvaluationAgent.")
            raise ValueError(MISSING_OPENAI_API_KEY)
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt: str) -> dict:
        """
        Run the evaluation loop for a given prompt.
        Returns a dictionary with final response, evaluation text and iteration count.
        """
        try:
            client = OpenAI(
                api_key=self.openai_api_key,
                base_url=OPENAI_BASE_URL if self.openai_api_key.lower().startswith("voc") else None,  # type: ignore
            )
        except Exception as e:
            logger.exception(
                "Failed to create OpenAI client in EvaluationAgent.evaluate"
            )
            return {"final_response": "", "evaluation": "", "iterations": 0}

        prompt_to_evaluate = initial_prompt

        for i in range(self.max_interactions):
            print(f"\n--- Interaction {i+1} ---")

            # Step 1: Worker agent generates a response
            try:
                response_from_worker = self.worker_agent.respond(
                    input_text=prompt_to_evaluate
                )
            except Exception as e:
                logger.exception("Worker agent responded with an error.")
                response_from_worker = ""

            # Step 2: Evaluator judges the response
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}\n"
                "Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            try:
                evaluation_resp = client.chat.completions.create(
                    model=OPENAI_BASE_MODEL,
                    messages=[{"role": "user", "content": eval_prompt}],
                    temperature=0,
                )
                evaluation = evaluation_resp.choices[0].message.content.strip()
            except openai.OpenAIError as e:
                logger.exception("OpenAI API error during evaluation.")
                evaluation = ""
            except Exception as e:
                logger.exception("Unexpected error during evaluation.")
                evaluation = ""

            # Step 3: Check if evaluation is positive
            if evaluation.lower().startswith("yes"):
                logger.info("✅ Final solution accepted.")
                break

            # Step 4: Generate instructions to correct the response
            instruction_prompt = f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
            try:
                instr_resp = client.chat.completions.create(
                    model=OPENAI_BASE_MODEL,
                    temperature=0,
                    messages=[{"role": "user", "content": instruction_prompt}],
                )
                instructions = instr_resp.choices[0].message.content.strip()
            except openai.OpenAIError as e:
                logger.exception(
                    "OpenAI API error while generating correction instructions."
                )
                instructions = ""
            except Exception as e:
                logger.exception(
                    "Unexpected error while generating correction instructions."
                )
                instructions = ""

            # Step 5: Send feedback to worker agent for refinement
            prompt_to_evaluate = (
                f"The original prompt was: {initial_prompt}\n"
                f"The response to that prompt was: {response_from_worker}\n"
                f"It has been evaluated as incorrect.\n"
                f"Make only these corrections, do not alter content validity: {instructions}"
            )

        return {
            "final_response": response_from_worker,  # type: ignore
            "evaluation": evaluation,  # type: ignore
            "iterations": i + 1,  # type: ignore
        }


# ------------------------------------------------------------
#  RoutingAgent
# ------------------------------------------------------------


class RoutingAgent:
    """
    Routes user prompts to the most suitable agent based on semantic similarity.
    """

    def __init__(self, openai_api_key: str, agents: list[dict]):
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        if not self.openai_api_key:
            logger.error("OpenAI API key is required for RoutingAgent.")
            raise ValueError(MISSING_OPENAI_API_KEY)
        self.agents = agents

    def get_embedding(self, text: str):
        """
        Return the embedding vector for a given text using OpenAI's embeddings endpoint.
        Returns None on failure.
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
            return _safe_get_embedding(client, text)
        except Exception as e:
            logger.exception("Error in RoutingAgent.get_embedding")
            return None

    def route(self, user_input: str):
        """
        Determine the best agent for the given input and invoke it.
        Returns the result of calling the selected agent's function with `user_input`.
        """
        try:
            input_emb = self.get_embedding(user_input)
            if input_emb is None:
                logger.error("Failed to obtain embedding for user input.")
                return "Sorry, could not process your request."
        except Exception as e:
            logger.exception("Error obtaining embedding for routing.")
            return "Sorry, could not process your request."

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
                logger.debug(f"Similarity with {agent['name']}: {similarity:.3f}")
                if similarity > best_score:
                    best_score = similarity
                    best_agent = agent
            except Exception as e:
                logger.exception(
                    f"Error evaluating similarity for agent {agent.get('name', 'unknown')}."
                )
                continue

        if best_agent is None:
            logger.warning("No suitable agent found after routing.")
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")

        # Call the chosen agent's function safely
        try:
            result = best_agent["func"](user_input)
            return result
        except Exception as e:
            logger.exception("Error executing selected agent.")
            return "Sorry, an error occurred while processing your request."


# ------------------------------------------------------------
#  ActionPlanningAgent
# ------------------------------------------------------------


class ActionPlanningAgent:
    """
    Extracts a list of actionable steps from a prompt using a knowledge base.
    """

    def __init__(self, openai_api_key: str, knowledge: str):
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        if not self.openai_api_key:
            logger.error("OpenAI API key is required for ActionPlanningAgent.")
            raise ValueError(MISSING_OPENAI_API_KEY)
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt: str) -> list[str]:
        """
        Extract a list of steps from the given user prompt.
        Returns an empty list on failure.
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
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are an action planning agent. Using your knowledge, "
                            f"you extract from the user prompt the steps requested to complete the action the user is asking for. "
                            f"You return the steps as a list. Only return the steps in your knowledge. "
                            f"Forget any previous context. This is your knowledge: {self.knowledge}"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            response_text = response.choices[0].message.content
            steps = [s.strip() for s in response_text.splitlines() if s.strip()]  # type: ignore
            return steps
        except openai.OpenAIError as e:
            logger.exception(
                "OpenAI API error in ActionPlanningAgent.extract_steps_from_prompt"
            )
        except Exception as e:
            logger.exception(
                "Unexpected error in ActionPlanningAgent.extract_steps_from_prompt"
            )
        return []
