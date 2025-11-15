import asyncio
import json
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------
# Setup
# -------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_NAME = "all-MiniLM-L6-v2"  # or "all-mpnet-base-v2"
MODEL = SentenceTransformer(MODEL_NAME)
EXECUTOR = ThreadPoolExecutor(max_workers=4)

SIMILARITY_THRESHOLD = 0.65  # below this we treat as hallucinated


# -------------------------
# Core utilities
# -------------------------

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    vec1 = np.asarray(vec1, dtype=np.float32)
    vec2 = np.asarray(vec2, dtype=np.float32)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


async def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        dim = MODEL.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        EXECUTOR,
        lambda: MODEL.encode(texts, convert_to_numpy=True)
    )


# -------------------------
# Small helper functions
# -------------------------

def clean_and_parse_llm_output(llm_output: str) -> List[Dict[str, Any]]:
    """Strip code fences if present and parse JSON."""
    stripped = llm_output.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return json.loads(stripped)


def extract_criteria_texts(
    llm_json: List[Dict[str, Any]],
    user_input: Dict[str, Any]
) -> (List[str], List[str], List[Dict[str, Any]]):
    """Return LLM texts, human texts, and expected_content list."""
    expected_content = user_input.get("compare_content", {}).get("expected_content", [])
    if not llm_json or not expected_content:
        logger.warning("LLM output or expected content is empty")

    llm_texts = [item.get("criteria", "") for item in llm_json]
    user_texts = [item.get("criteria", "") for item in expected_content]
    return llm_texts, user_texts, expected_content


def compute_pair_similarities(
    llm_embeds: np.ndarray,
    user_embeds: np.ndarray
) -> (List[float], int):
    """Compute similarity for aligned pairs and return (similarities, n_pairs)."""
    n_pairs = min(llm_embeds.shape[0], user_embeds.shape[0])
    pair_sims: List[float] = []
    for i in range(n_pairs):
        pair_sims.append(cosine_similarity(llm_embeds[i], user_embeds[i]))
    return pair_sims, n_pairs


def detect_hallucinations(
    pair_sims: List[float],
    llm_embeds: np.ndarray,
    user_embeds: np.ndarray,
    n_pairs: int,
    threshold: float
) -> (List[int], int, float):
    """Return hallucinated indices, count, and rate."""
    hallucinations: List[int] = []

    if user_embeds.shape[0] == 0:
        hallucinations = list(range(llm_embeds.shape[0]))
    else:
        for i, sim in enumerate(pair_sims):
            if sim < threshold:
                hallucinations.append(i)
        if llm_embeds.shape[0] > n_pairs:
            hallucinations.extend(range(n_pairs, llm_embeds.shape[0]))

    hallucination_count = len(hallucinations)
    hallucination_rate = (
        hallucination_count / llm_embeds.shape[0]
        if llm_embeds is not None and llm_embeds.shape[0] > 0
        else 0.0
    )
    return hallucinations, hallucination_count, hallucination_rate


def build_per_criterion_results(
    llm_json: List[Dict[str, Any]],
    llm_texts: List[str],
    user_texts: List[str],
    user_input: Dict[str, Any],
    pair_sims: List[float],
    hallucinations: List[int],
    n_pairs: int,
    threshold: float
) -> List[Dict[str, Any]]:
    """Build list of per-criterion result dicts."""
    framework = user_input.get("framework", "")
    control_id = user_input.get("control_id", "")
    control = user_input.get("control", "")
    subcontrol = user_input.get("subcontrol", "")

    results: List[Dict[str, Any]] = []

    # paired criteria
    for i in range(n_pairs):
        sim = pair_sims[i]
        results.append({
            "criteria_id": llm_json[i].get("id"),
            "human_criteria": user_texts[i],
            "llm_criteria": llm_texts[i],
            "framework": framework,
            "control_id": control_id,
            "control": control,
            "subcontrol": subcontrol,
            "semantic_similarity": round(sim, 4),
            "is_hallucinated": i in hallucinations,
            "semantic_similarity_threshold": threshold
        })

    # extra LLM-only criteria
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
    hallucination_count: int,
    hallucination_rate: float
) -> Dict[str, Any]:
    """Build the overall summary metrics dict."""
    average_semantic_similarity = float(np.mean(pair_sims)) if pair_sims else 0.0

    return {
        "control_id": user_input.get("control_id", ""),
        "control": user_input.get("control", ""),
        "subcontrol": user_input.get("subcontrol", ""),
        "framework": user_input.get("framework", ""),
        "average_semantic_similarity": round(average_semantic_similarity, 4),
        "hallucination_count": hallucination_count,
        "hallucination_rate": round(hallucination_rate, 4),
        "hallucinations": [] if hallucination_count == 0 else None  # optional: or pass indices
    }


# -------------------------
# Main evaluation function
# -------------------------

async def evaluate_llm_vs_user_semantic_only(
    llm_output: str,
    user_input: Dict[str, Any]
) -> Dict[str, Any]:
    default_result = {"overall_metrics": {}, "per_criterion_results": []}

    try:
        # 1) Parse and clean LLM JSON
        llm_json = clean_and_parse_llm_output(llm_output)

        # 2) Extract texts
        llm_texts, user_texts, _ = extract_criteria_texts(llm_json, user_input)

        # 3) Embed both sides
        llm_embeds, user_embeds = await asyncio.gather(
            embed_texts(llm_texts),
            embed_texts(user_texts)
        )

        # 4) Similarities
        pair_sims, n_pairs = compute_pair_similarities(llm_embeds, user_embeds)

        # 5) Hallucinations
        hallucinations, hallucination_count, hallucination_rate = detect_hallucinations(
            pair_sims, llm_embeds, user_embeds, n_pairs, SIMILARITY_THRESHOLD
        )

        # 6) Per-criterion details
        per_criterion_results = build_per_criterion_results(
            llm_json, llm_texts, user_texts, user_input,
            pair_sims, hallucinations, n_pairs, SIMILARITY_THRESHOLD
        )

        # 7) Overall metrics
        overall_metrics = build_overall_metrics(
            user_input, pair_sims, hallucination_count, hallucination_rate
        )

        return {
            "overall_metrics": overall_metrics,
            "per_criterion_results": per_criterion_results
        }

    except Exception:
        logger.exception("Evaluation failed")
        return default_result


# -------------------------
# Demo
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
