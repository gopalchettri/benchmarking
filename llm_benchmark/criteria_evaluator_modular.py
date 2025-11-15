"""
semantic_evaluator.py

Goal:
Compare AI-generated criteria with human-written criteria and calculate
how similar they are in meaning (semantic similarity).

Main public function:
    evaluate_llm_vs_user_semantic_only(llm_output, user_input)

It returns:
{
  "overall_metrics": { ... summary for one control ... },
  "per_criterion_results": [ ... one entry per criterion pair ... ]
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
# Global setup – logging, model, and thresholds
# ---------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Name of the sentence embedding model.
# This model turns each sentence into a numeric vector representing its meaning.
MODEL_NAME = "all-MiniLM-L6-v2"   # You can later try "all-mpnet-base-v2" for higher quality.

# Load the embedding model once (so we don't reload it every time).
MODEL = SentenceTransformer(MODEL_NAME)

# Thread pool so embedding work can run in the background without blocking everything.
EXECUTOR = ThreadPoolExecutor(max_workers=4)

# If similarity between AI and human text is below this value,
# we will treat it as "hallucinated" (too different from what was expected).
SIMILARITY_THRESHOLD = 0.65


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

    # Vector length (magnitude)
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
# Small helper functions – each does one simple thing
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


def detect_hallucinations(
    pair_sims: List[float],
    llm_count: int,
    human_count: int,
    n_pairs: int,
    threshold: float
) -> Tuple[List[int], int, float]:
    """
    Decide which AI criteria are "hallucinated":

    - too low similarity with matching human sentence (< threshold), OR
    - extra AI sentences that have no human reference.

    Returns:
      - list of indices that are hallucinated
      - how many there are
      - what fraction of all AI criteria that is
    """
    hallucinations: List[int] = []

    # If there are no human sentences at all, everything from the AI is "untrusted".
    if human_count == 0:
        hallucinations = list(range(llm_count))
    else:
        # Mark pairs with low similarity as hallucinated.
        for i, sim in enumerate(pair_sims):
            if sim < threshold:
                hallucinations.append(i)

        # Any extra AI sentences beyond the number of human sentences
        # are also treated as hallucinated (no reference to compare to).
        if llm_count > n_pairs:
            hallucinations.extend(range(n_pairs, llm_count))

    hallucination_count = len(hallucinations)
    hallucination_rate = (hallucination_count / llm_count) if llm_count > 0 else 0.0

    return hallucinations, hallucination_count, hallucination_rate


def build_per_criterion_results(
    llm_json: List[Dict[str, Any]],
    llm_texts: List[str],
    human_texts: List[str],
    user_input: Dict[str, Any],
    pair_sims: List[float],
    hallucinations: List[int],
    n_pairs: int,
    threshold: float
) -> List[Dict[str, Any]]:
    """
    Build a list with detailed information for each criterion pair.

    Each item in the list describes:
      - the human sentence,
      - the AI sentence,
      - how similar they are,
      - whether we flagged it as hallucinated.
    """
    framework = user_input.get("framework", "")
    control_id = user_input.get("control_id", "")
    control = user_input.get("control", "")
    subcontrol = user_input.get("subcontrol", "")

    results: List[Dict[str, Any]] = []

    # First, handle the pairs where both AI and human criteria exist.
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
            "semantic_similarity": round(sim, 4),
            "is_hallucinated": i in hallucinations,
            "semantic_similarity_threshold": threshold
        })

    # Then, if there are extra AI criteria with no human pair,
    # add them as fully hallucinated with similarity 0.
    for i in range(n_pairs, len(llm_texts)):
        results.append({
            "criteria_id": llm_json[i].get("id"),
            "human_criteria": "",
            "llm_criteria": llm_texts[i],
            "framework": framework,
            "control_id": control_id,
            "control": control,
            "subcontrol": subcontrol,
            "semantic_similarity": 0.0,
            "is_hallucinated": True,
            "semantic_similarity_threshold": threshold
        })

    return results


def build_overall_metrics(
    user_input: Dict[str, Any],
    pair_sims: List[float],
    hallucinations: List[int],
    hallucination_rate: float
) -> Dict[str, Any]:
    """
    Build a single summary dictionary for this control / test case.

    Key number here:
      - average_semantic_similarity: average score across all pairs.
    """
    average_sim = float(np.mean(pair_sims)) if pair_sims else 0.0

    return {
        "control_id": user_input.get("control_id", ""),
        "control": user_input.get("control", ""),
        "subcontrol": user_input.get("subcontrol", ""),
        "framework": user_input.get("framework", ""),
        "average_semantic_similarity": round(average_sim, 4),
        "hallucination_count": len(hallucinations),
        "hallucination_rate": round(hallucination_rate, 4),
        "hallucinations": hallucinations,
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

        # 4) Compute similarity for each aligned pair
        pair_sims, n_pairs = compute_pair_similarities(llm_embeds, human_embeds)

        # 5) Detect hallucinations (too different / no human reference)
        llm_count = llm_embeds.shape[0]
        human_count = human_embeds.shape[0]

        hallucinations, hallucination_count, hallucination_rate = detect_hallucinations(
            pair_sims=pair_sims,
            llm_count=llm_count,
            human_count=human_count,
            n_pairs=n_pairs,
            threshold=SIMILARITY_THRESHOLD
        )

        # 6) Build detailed report per criterion
        per_criterion_results = build_per_criterion_results(
            llm_json=llm_json,
            llm_texts=llm_texts,
            human_texts=human_texts,
            user_input=user_input,
            pair_sims=pair_sims,
            hallucinations=hallucinations,
            n_pairs=n_pairs,
            threshold=SIMILARITY_THRESHOLD
        )

        # 7) Build overall summary metrics
        overall_metrics = build_overall_metrics(
            user_input=user_input,
            pair_sims=pair_sims,
            hallucinations=hallucinations,
            hallucination_rate=hallucination_rate
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
