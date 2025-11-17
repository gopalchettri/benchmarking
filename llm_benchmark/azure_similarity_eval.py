"""
azure_similarity_eval.py

Lightweight helper to add Azure GPT-4o-mini SimilarityEvaluator
on top of your existing cosine / MPNet pipeline.

Usage pattern in your evaluator:
    from azure_similarity_eval import add_azure_similarity_to_results

    per_criterion_results = build_per_criterion_results(...)
    per_criterion_results = add_azure_similarity_to_results(user_input, per_criterion_results)
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

from azure.ai.evaluation import (
    SimilarityEvaluator,
    AzureOpenAIModelConfiguration,
)


# -----------------------------
# Global / singleton evaluator
# -----------------------------

_AZURE_SIMILARITY_EVALUATOR: Optional[SimilarityEvaluator] = None
_AZURE_SIMILARITY_THRESHOLD: int = 3  # Likert 1–5; 3 = pass/fail cutoff


def _build_model_config() -> AzureOpenAIModelConfiguration:
    """
    Build AzureOpenAIModelConfiguration from environment variables.

    Required env vars:
      - AZURE_OPENAI_ENDPOINT
      - AZURE_OPENAI_API_KEY
      - AZURE_OPENAI_DEPLOYMENT_NAME  (your gpt-4o-mini deployment name)

    Optional:
      - AZURE_OPENAI_API_VERSION  (default: 2024-08-01-preview)
    """
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    if not endpoint or not api_key or not deployment_name:
        raise RuntimeError(
            "Azure OpenAI configuration missing. Please set "
            "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
            "AZURE_OPENAI_DEPLOYMENT_NAME (gpt-4o-mini)."
        )

    return AzureOpenAIModelConfiguration(
        azure_endpoint=endpoint,
        api_key=api_key,
        deployment_name=deployment_name,
        api_version=api_version,
    )


def _get_evaluator() -> Optional[SimilarityEvaluator]:
    """
    Returns a singleton SimilarityEvaluator instance.

    If env vars are missing, returns None (so your pipeline keeps working
    without Azure similarity).
    """
    global _AZURE_SIMILARITY_EVALUATOR

    if _AZURE_SIMILARITY_EVALUATOR is not None:
        return _AZURE_SIMILARITY_EVALUATOR

    try:
        model_config = _build_model_config()
    except RuntimeError as e:
        # Don't blow up your main pipeline – just disable Azure similarity.
        # You can optionally log this where you import this module.
        print(f"[azure_similarity_eval] Azure similarity disabled: {e}")
        _AZURE_SIMILARITY_EVALUATOR = None
        return None

    _AZURE_SIMILARITY_EVALUATOR = SimilarityEvaluator(
        model_config=model_config,
        threshold=_AZURE_SIMILARITY_THRESHOLD,
    )
    return _AZURE_SIMILARITY_EVALUATOR


def _evaluate_pair(
    framework: str,
    control_id: str,
    control: str,
    subcontrol: str,
    llm_criteria: str,
    human_criteria: str,
) -> tuple[Optional[float], Optional[str]]:
    """
    Call Azure SimilarityEvaluator for a single LLM vs human criterion pair.

    Returns:
        (similarity_score, similarity_result)
        where similarity_score is 1–5 (Likert) or None,
        and similarity_result is "pass"/"fail" or None.
    """
    evaluator = _get_evaluator()
    if evaluator is None:
        return None, None

    query = f"{framework} | {control_id} | {control} | {subcontrol}"

    result = evaluator(
        query=query,
        response=llm_criteria,
        ground_truth=human_criteria,
    )

    score = result.get("similarity", None)           # float 1–5
    verdict = result.get("similarity_result", None)  # "pass"/"fail"
    return score, verdict


# -----------------------------
# Public helper entry point
# -----------------------------

def add_azure_similarity_to_results(
    user_input: Dict[str, Any],
    per_criterion_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Enrich your existing per_criterion_results with Azure similarity fields.

    Expected structure in per_criterion_results:
      - "llm_criteria": str
      - "human_criteria": str

    This function adds:
      - "azure_similarity_score": float | None  (1–5 Likert)
      - "azure_similarity_result": "pass"/"fail" | None
      - "azure_similarity_threshold": int

    If Azure is not configured, these will all be None and your pipeline
    still works as before.
    """
    framework = user_input.get("framework", "")
    control_id = user_input.get("control_id", "")
    control = user_input.get("control", "")
    subcontrol = user_input.get("subcontrol", "")

    evaluator = _get_evaluator()
    if evaluator is None:
        # Nothing to do; just attach None fields (optional).
        for item in per_criterion_results:
            item.setdefault("azure_similarity_score", None)
            item.setdefault("azure_similarity_result", None)
            item.setdefault("azure_similarity_threshold", _AZURE_SIMILARITY_THRESHOLD)
        return per_criterion_results

    # If evaluator exists, score each pair
    for item in per_criterion_results:
        llm_text = item.get("llm_criteria", "") or ""
        human_text = item.get("human_criteria", "") or ""

        score, verdict = _evaluate_pair(
            framework=framework,
            control_id=control_id,
            control=control,
            subcontrol=subcontrol,
            llm_criteria=llm_text,
            human_criteria=human_text,
        )

        item["azure_similarity_score"] = score
        item["azure_similarity_result"] = verdict
        item["azure_similarity_threshold"] = _AZURE_SIMILARITY_THRESHOLD

    return per_criterion_results
