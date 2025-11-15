"""
semantic_evaluator.py

Purpose:
- Compare criteria written by an AI model with criteria written by a human expert.
- Measure how close they are in meaning (similarity from 0 to 1).
- Highlight which AI criteria are clearly not close enough and should be reviewed.

Main function to use:
    evaluate_llm_vs_user_semantic_only(llm_output, user_input)

Inputs:
- llm_output: JSON string from the AI, e.g.
      '[{"id": 1, "criteria": "..."}, {"id": 2, "criteria": "..."}]'
- user_input: Python dict with fields:
      control_id, control, subcontrol, framework,
      compare_content.expected_content = list of {"id", "criteria"}

Output:
{
  "overall_metrics": {
    "control_id": "...",
    "control": "...",
    "subcontrol": "...",
    "framework": "...",
    "average_similarity": 0.87,
    "low_similarity_count": 1
  },
  "per_criterion_results": [
    {
      "criteria_id": 1,
      "human_criteria": "...",
      "llm_criteria": "...",
      "framework": "...",
      "control_id": "...",
      "control": "...",
      "subcontrol": "...",
      "similarity": 0.91,
      "is_low_similarity": false
    },
    ...
  ]
}
"""

# ---------------------------------------------------------
# Imports – standard Python, math, and sentence embeddings
# ---------------------------------------------------------

import asyncio              # To run some work asynchronously (in parallel style)
import json                 # To read and write JSON text
import logging              # To print logs and errors
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor  # To run heavy tasks in background threads

import numpy as np          # For number crunching (vectors, averages, etc.)
from sentence_transformers import SentenceTransformer  # For turning sentences into vectors


# ---------------------------------------------------------
# Global setup – logging, model, and similarity limit
# ---------------------------------------------------------

# Set up a basic logger so we can see warnings or errors
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Name of the sentence embedding model.
# This model turns each sentence into a numeric vector representing its meaning.
MODEL_NAME = "all-MiniLM-L6-v2"   # You can later try "all-mpnet-base-v2" for higher accuracy.

# Load the embedding model once (so we don't reload it every time).
MODEL = SentenceTransformer(MODEL_NAME)

# Thread pool so embedding work can run in the background without blocking everything.
EXECUTOR = ThreadPoolExecutor(max_workers=4)

# If similarity between AI and human text is below this value,
# we will call it "low similarity" and mark it for review.
LOW_SIMILARITY_LIMIT = 0.70


# ---------------------------------------------------------
# Core utilities
# ---------------------------------------------------------

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Measure how similar two numeric vectors are.
    Each vector represents one sentence.

    Result is between -1 and 1:
      - 1  : very similar meaning
      - 0  : unrelated
      - -1 : opposite
    """
    # Make sure both vectors are NumPy arrays of numbers.
    vec1 = np.asarray(vec1, dtype=np.float32)
    vec2 = np.asarray(vec2, dtype=np.float32)

    # Vector lengths (magnitudes)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # If either vector has zero length, we cannot compare them well.
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    # Standard cosine similarity formula: dot product divided by product of lengths.
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


async def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Turn a list of sentences into a matrix of vectors using the embedding model.

    Each sentence -> one vector (row in the matrix).
    """
    # If we got an empty list, return an empty matrix with the right width.
    if not texts:
        dim = MODEL.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)

    # Ask Python for the current asynchronous "event loop".
    loop = asyncio.get_event_loop()

    # Run the model encoding in a background thread.
    # This avoids blocking the main program while embeddings are computed.
    return await loop.run_in_executor(
        EXECUTOR,
        lambda: MODEL.encode(texts, convert_to_numpy=True)
    )


# ---------------------------------------------------------
# Helper functions – each does one small step
# ---------------------------------------------------------

def clean_and_parse_llm_output(llm_output: str) -> List[Dict[str, Any]]:
    """
    Clean the AI (LLM) output and parse it as JSON.

    Sometimes models wrap JSON in ``` or ```json ... ``` code blocks.
    This function removes those wrappers and returns the list of items.
    """
    stripped = llm_output.strip()

    # If it starts with ``` then it's probably inside a code block.
    if stripped.startswith("```"):
        # Find the first newline after ``` or ```json
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1:]
        # Remove closing ```
        if stripped.endswith("```"):
            stripped = stripped[:-3]

    # Now it should be plain JSON text.
    return json.loads(stripped)


def extract_criteria_texts(
    llm_json: List[Dict[str, Any]],
    user_input: Dict[str, Any]
) -> Tuple[List[str], List[str]]:
    """
    From the AI JSON and the human input, extract:
      - list of AI criteria sentences
      - list of human criteria sentences
    """
    expected_content = user_input.get("compare_content", {}).get("expected_content", [])

    if not llm_json or not expected_content:
        logger.warning("LLM output or expected content is empty")

    # Get just the 'criteria' strings from each item.
    llm_texts = [item.get("criteria", "") for item in llm_json]
    human_texts = [item.get("criteria", "") for item in expected_content]

    return llm_texts, human_texts


def compute_pair_similarities(
    llm_embeds: np.ndarray,
    human_embeds: np.ndarray
) -> Tuple[List[float], int]:
    """
    Compute similarity score for each aligned pair:
      (AI sentence 1 vs Human sentence 1, etc.)

    Returns:
      - list of similarity scores
      - number of pairs actually compared
    """
    # Compare only up to the shorter length of both lists.
    n_pairs = min(llm_embeds.shape[0], human_embeds.shape[0])

    pair_sims: List[float] = []
    for i in range(n_pairs):
        pair_sims.append(cosine_similarity(llm_embeds[i], human_embeds[i]))

    return pair_sims, n_pairs


def find_low_similarity_items(
    pair_sims: List[float],
    llm_count: int,
    human_count: int,
    n_pairs: int,
    low_limit: float
) -> Tuple[List[int], int]:
    """
    Find which AI criteria have low similarity compared to human criteria.

    - low similarity = similarity < low_limit
    - extra AI criteria (no human pair) are also treated as low similarity.

    Returns:
      - indices of items that have low similarity
      - how many such items there are
    """
    low_indices: List[int] = []

    if human_count == 0:
        # No human baseline: treat everything as low similarity
        low_indices = list(range(llm_count))
    else:
        # Check each pair score
        for i, sim in enumerate(pair_sims):
            if sim < low_limit:
                low_indices.append(i)

        # Extra AI-only items
        if llm_count > n_pairs:
            low_indices.extend(range(n_pairs, llm_count))

    count = len(low_indices)
    return low_indices, count


def build_per_criterion_results(
    llm_json: List[Dict[str, Any]],
    llm_texts: List[str],
    human_texts: List[str],
    user_input: Dict[str, Any],
    pair_sims: List[float],
    low_indices: List[int],
    n_pairs: int,
) -> List[Dict[str, Any]]:
    """
    Build one record per criterion pair with:
      - human text
      - AI text
      - similarity score
      - whether it has low similarity
    """
    framework = user_input.get("framework", "")
    control_id = user_input.get("control_id", "")
    control = user_input.get("control", "")
    subcontrol = user_input.get("subcontrol", "")

    results: List[Dict[str, Any]] = []

    # AI + human pairs
    for i in range(n_pairs):
        sim = pair_sims[i]
        results.append({
            "criteria_id": llm_json[i].get("id"),
            "human_criteria": human_texts[i],
            "llm_criteria": llm_texts[i],
            "framework": framework,
            "control_id": control_id,
            "control": control,
            "subcontrol": subcontrol,
            # Main number for each criterion
            "similarity": round(sim, 4),
            # True if this item should be considered weak / needs checking
            "is_low_similarity": i in low_indices
        })

    # Extra AI-only criteria (no human counterpart)
    for i in range(n_pairs, len(llm_texts)):
        results.append({
            "criteria_id": llm_json[i].get("id"),
            "human_criteria": "",
            "llm_criteria": llm_texts[i],
            "framework": framework,
            "control_id": control_id,
            "control": control,
            "subcontrol": subcontrol,
            "similarity": 0.0,
            "is_low_similarity": True
        })

    return results


def build_overall_metrics(
    user_input: Dict[str, Any],
    pair_sims: List[float],
    low_indices: List[int],
) -> Dict[str, Any]:
    """
    Build an easy-to-understand summary for this control / test case.

    Key number:
      - average_similarity: average score across all pairs (0 to 1).
    """
    average_similarity = float(np.mean(pair_sims)) if pair_sims else 0.0

    return {
        "control_id": user_input.get("control_id", ""),
        "control": user_input.get("control", ""),
        "subcontrol": user_input.get("subcontrol", ""),
        "framework": user_input.get("framework", ""),
        # Main KPI in plain language
        "average_similarity": round(average_similarity, 4),
        # How many AI criteria have clearly low similarity
        "low_similarity_count": len(low_indices)
    }


# ---------------------------------------------------------
# Public function – this is what you call from your code
# ---------------------------------------------------------

async def evaluate_llm_vs_user_semantic_only(
    llm_output: str,
    user_input: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare AI-generated criteria (llm_output) against human criteria (user_input).

    Expected inputs:
      - llm_output: JSON string produced by the AI,
                    e.g. '[{"id": 1, "criteria": "..."}, ...]'
      - user_input: dict with keys:
            control_id, control, subcontrol, framework,
            compare_content.expected_content (list of {"id", "criteria"})

    Returns:
      {
        "overall_metrics": { ... summary for this control ... },
        "per_criterion_results": [ ... one entry per criterion pair ... ]
      }
    """
    default_result = {"overall_metrics": {}, "per_criterion_results": []}

    try:
        # 1) Parse AI JSON (and remove any ``` wrappers if present)
        llm_json = clean_and_parse_llm_output(llm_output)

        # 2) Extract just the sentences from both sides
        llm_texts, human_texts = extract_criteria_texts(llm_json, user_input)

        # 3) Turn sentences into vectors using the embedding model
        llm_embeds, human_embeds = await asyncio.gather(
            embed_texts(llm_texts),
            embed_texts(human_texts)
        )

        # 4) Compute similarity for each pair
        pair_sims, n_pairs = compute_pair_similarities(llm_embeds, human_embeds)

        # 5) Find which ones have low similarity
        llm_count = llm_embeds.shape[0]
        human_count = human_embeds.shape[0]

        low_indices, _ = find_low_similarity_items(
            pair_sims=pair_sims,
            llm_count=llm_count,
            human_count=human_count,
            n_pairs=n_pairs,
            low_limit=LOW_SIMILARITY_LIMIT
        )

        # 6) Per-criterion result list
        per_criterion_results = build_per_criterion_results(
            llm_json=llm_json,
            llm_texts=llm_texts,
            human_texts=human_texts,
            user_input=user_input,
            pair_sims=pair_sims,
            low_indices=low_indices,
            n_pairs=n_pairs,
        )

        # 7) Overall metrics
        overall_metrics = build_overall_metrics(
            user_input=user_input,
            pair_sims=pair_sims,
            low_indices=low_indices,
        )

        return {
            "overall_metrics": overall_metrics,
            "per_criterion_results": per_criterion_results
        }

    except Exception:
        # If anything goes wrong, log the error and return an empty result.
        logger.exception("Evaluation failed")
        return default_result


# ---------------------------------------------------------
# Demo – quick example you can run directly
# ---------------------------------------------------------

async def main_demo() -> None:
    """
    Small example to show how the evaluator works.

    In your real project you will:
      - get user_input from your test_cases JSON,
      - get llm_output from your LLM call,
      - then call evaluate_llm_vs_user_semantic_only(...)
    """
    user_input = {
        "control_id": "M1.1.1",
        "control": "Understanding The Entity and its Context",
        "subcontrol": "The entity shall determine interested parties that are relevant to its information security.",
        "framework": "UAE Information Assurance Regulation",
        "compare_content": {
            "expected_content": [
                {"id": 1, "criteria": "Identify stakeholders"},
                {"id": 2, "criteria": "Conduct stakeholder analysis"},
                {"id": 3, "criteria": "Establish communication channels"}
            ]
        }
    }

    # Example AI output (normally this comes from your model)
    llm_output_json = json.dumps([
        {"id": 1, "criteria": "Identify all relevant stakeholders affecting information security."},
        {"id": 2, "criteria": "Perform stakeholder analysis on a regular basis."},
        {"id": 3, "criteria": "Define and maintain communication channels with stakeholders."}
    ])

    # Run the evaluator
    result = await evaluate_llm_vs_user_semantic_only(llm_output_json, user_input)

    # Pretty-print the JSON result
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    # If you run this file directly: python semantic_evaluator.py
    asyncio.run(main_demo())
