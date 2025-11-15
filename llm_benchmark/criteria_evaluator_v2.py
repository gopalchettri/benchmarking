
# updated
import asyncio
import json
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------
# Configuration & Logging
# -------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# SentenceTransformer model
# You can switch to "all-mpnet-base-v2" later if you want a stronger model.
MODEL_NAME = "all-MiniLM-L6-v2"
MODEL = SentenceTransformer(MODEL_NAME)
EXECUTOR = ThreadPoolExecutor(max_workers=4)

# Thresholds
SIMILARITY_THRESHOLD = 0.65   # hallucination detection
GOOD_THRESHOLD_85 = 0.85
GOOD_THRESHOLD_90 = 0.90


# -------------------------
# Utility functions
# -------------------------

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two vectors.
    Returns a value in [-1, 1].
    """
    vec1 = np.asarray(vec1, dtype=np.float32)
    vec2 = np.asarray(vec2, dtype=np.float32)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


async def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Convert a list of texts into numerical embeddings using SentenceTransformer.
    """
    if not texts:
        dim = MODEL.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        EXECUTOR,
        lambda: MODEL.encode(texts, convert_to_numpy=True)
    )


def detect_hallucinations(
    llm_embeds: np.ndarray,
    user_embeds: np.ndarray,
    threshold: float
) -> List[int]:
    """
    Mark outputs as hallucinated if similarity with corresponding reference
    is below `threshold`. Extra LLM outputs (without reference) are also
    considered hallucinated.
    """
    hallucinations: List[int] = []

    if user_embeds is None or user_embeds.shape[0] == 0:
        # No reference at all: everything is hallucinated
        return list(range(llm_embeds.shape[0])) if llm_embeds is not None else []

    n = min(llm_embeds.shape[0], user_embeds.shape[0])
    for i in range(n):
        sim = cosine_similarity(llm_embeds[i], user_embeds[i])
        if sim < threshold:
            hallucinations.append(i)

    # Any extra LLM outputs beyond reference count are hallucinated
    if llm_embeds.shape[0] > n:
        hallucinations.extend(range(n, llm_embeds.shape[0]))

    return hallucinations


# -------------------------
# Main evaluation function
# -------------------------

async def evaluate_llm_vs_user_semantic_only(
    llm_output: str,
    user_input: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate LLM-generated criteria against user/reference criteria using
    semantic similarity only.

    Returns:
      {
        "overall_metrics": {
            "control_id": ...,
            "semantic_similarity": ...,      # mean similarity (same as overall_mean)
            "overall_mean": ...,
            "coverage_85": ...,
            "coverage_90": ...,
            ...
        },
        "per_criterion_results": [
            {
                "semantic_similarity": ...,
                "meets_85": true/false,
                "meets_90": true/false,
                ...
            },
            ...
        ]
      }
    """
    default_result = {"overall_metrics": {}, "per_criterion_results": []}

    try:
        # -------------------------
        # Parse LLM output
        # -------------------------
        # Robustly strip code fences if present
        stripped = llm_output.strip()
        if stripped.startswith("```"):
            # remove starting fence like ``` or ```json
            first_newline = stripped.find("\n")
            if first_newline != -1:
                stripped = stripped[first_newline + 1:]
            if stripped.endswith("```"):
                stripped = stripped[:-3]
        llm_json = json.loads(stripped)

        expected_content = user_input.get("compare_content", {}).get("expected_content", [])

        if not llm_json or not expected_content:
            logger.warning("LLM output or expected content is empty")

        # Reference/user criteria and LLM criteria
        llm_texts = [item.get("criteria", "") for item in llm_json]
        user_texts = [item.get("criteria", "") for item in expected_content]

        # -------------------------
        # Generate embeddings
        # -------------------------
        llm_embeds, user_embeds = await asyncio.gather(
            embed_texts(llm_texts),
            embed_texts(user_texts)
        )

        n_pairs = min(llm_embeds.shape[0], user_embeds.shape[0])

        # -------------------------
        # Detect hallucinations
        # -------------------------
        hallucinations = detect_hallucinations(llm_embeds, user_embeds, SIMILARITY_THRESHOLD)
        hallucination_count = len(hallucinations)
        hallucination_rate = (
            hallucination_count / llm_embeds.shape[0]
            if llm_embeds is not None and llm_embeds.shape[0] > 0
            else 0.0
        )

        # -------------------------
        # Compute similarities once (vectorised loop)
        # -------------------------
        pair_sims: List[float] = []
        for i in range(n_pairs):
            pair_sims.append(cosine_similarity(llm_embeds[i], user_embeds[i]))

        # Coverage metrics
        overall_mean = float(np.mean(pair_sims)) if pair_sims else 0.0
        coverage_85 = float(np.mean([s >= GOOD_THRESHOLD_85 for s in pair_sims])) if pair_sims else 0.0
        coverage_90 = float(np.mean([s >= GOOD_THRESHOLD_90 for s in pair_sims])) if pair_sims else 0.0

        # -------------------------
        # Per-criterion evaluation
        # -------------------------
        per_criterion_results = []
        for i in range(n_pairs):
            sim = pair_sims[i]
            per_criterion_results.append({
                "criteria_id": llm_json[i].get("id"),
                "human_criteria": user_texts[i],
                "llm_criteria": llm_texts[i],
                "framework": user_input.get("framework", ""),
                "control_id": user_input.get("control_id", ""),
                "control": user_input.get("control", ""),
                "subcontrol": user_input.get("subcontrol", ""),
                "semantic_similarity": round(sim, 4),
                "is_hallucinated": i in hallucinations,
                "semantic_similarity_threshold": SIMILARITY_THRESHOLD,
                # extra flags for per-criteria evaluation
                "meets_85": sim >= GOOD_THRESHOLD_85,
                "meets_90": sim >= GOOD_THRESHOLD_90
            })

        # Add hallucinated outputs without reference
        for i in range(n_pairs, llm_embeds.shape[0]):
            per_criterion_results.append({
                "criteria_id": llm_json[i].get("id"),
                "human_criteria": "",
                "llm_criteria": llm_texts[i],
                "framework": user_input.get("framework", ""),
                "control_id": user_input.get("control_id", ""),
                "control": user_input.get("control", ""),
                "subcontrol": user_input.get("subcontrol", ""),
                "semantic_similarity": 0.0,
                "is_hallucinated": True,
                "semantic_similarity_threshold": SIMILARITY_THRESHOLD,
                "meets_85": False,
                "meets_90": False
            })

        # -------------------------
        # Overall metrics (per test_case / control_id)
        # -------------------------
        overall_metrics = {
            "control_id": user_input.get("control_id", ""),
            "control": user_input.get("control", ""),
            "subcontrol": user_input.get("subcontrol", ""),
            "framework": user_input.get("framework", ""),
            # backward-compatible overall similarity
            "semantic_similarity": round(overall_mean, 4),
            # new metrics
            "overall_mean": round(overall_mean, 4),
            "coverage_85": round(coverage_85, 4),
            "coverage_90": round(coverage_90, 4),
            "hallucination_count": hallucination_count,
            "hallucination_rate": round(hallucination_rate, 4),
            "hallucinations": hallucinations
        }

        return {
            "overall_metrics": overall_metrics,
            "per_criterion_results": per_criterion_results
        }

    except Exception:
        logger.exception("Evaluation failed")
        return default_result


# -------------------------
# Demo main
# -------------------------
async def main_demo():
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

    llm_output_json = json.dumps([
        {"id": 1, "criteria": "Identify all relevant stakeholders affecting information security."},
        {"id": 2, "criteria": "Perform stakeholder analysis on a regular basis."},
        {"id": 3, "criteria": "Define and maintain communication channels with stakeholders."}
    ])

    result = await evaluate_llm_vs_user_semantic_only(llm_output_json, user_input)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main_demo())
