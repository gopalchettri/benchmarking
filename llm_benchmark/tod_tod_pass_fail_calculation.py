from typing import Dict, Any, List

def summarize_semantic_results(
    per_criterion_results: List[Dict[str, Any]],
    threshold: float,
) -> Dict[str, Any]:
    """
    Compute summary statistics for a given TOD/TOE result set:

      - total_criteria    : total number of LLM criteria evaluated
      - passed_criteria   : similarity >= threshold
      - failed_criteria   : similarity < threshold

    Uses the 'is_low_similarity' flag that is already set for each item.
    """
    total = len(per_criterion_results)

    # Passed = NOT low similarity -> similarity >= threshold
    passed = sum(
        1
        for item in per_criterion_results
        if not item.get("is_low_similarity", False)
    )

    failed = total - passed

    return {
        "total_criteria": total,
        "passed_criteria": passed,
        "failed_criteria": failed,
        "threshold": threshold,
    }
async def evaluate_semantic_for_criteria_type(
    llm_case: Dict[str, Any],
    human_case: Dict[str, Any],
    criteria_type: Literal["tod", "toe"] = "tod",
) -> Dict[str, Any]:
    """
    Evaluate semantic similarity between human-written and LLM-generated
    criteria for a single type ("tod" or "toe") for one control.

    Returns:
    {
      "criteria": [ per-criterion records ... ],
      "summary": {
          "criteria_type": "tod" or "toe",
          "total_criteria": int,
          "passed_criteria": int,
          "failed_criteria": int,
          "threshold": float
      }
    }
    """
    default_result = {
        "criteria": [],
        "summary": {
            "criteria_type": criteria_type,
            "total_criteria": 0,
            "passed_criteria": 0,
            "failed_criteria": 0,
            "threshold": LOW_SIMILARITY_LIMIT,
        },
    }

    try:
        # 1) Extract the raw criteria lists for this type (TOD/TOE)
        llm_items, human_items = await extract_criteria_lists_for_type(
            llm_case=llm_case,
            human_case=human_case,
            criteria_type=criteria_type,
        )

        # If nothing, return early
        if not llm_items and not human_items:
            return default_result

        # 🔹 Ensure same-id items line up by index (extra safety)
        llm_items   = sorted(llm_items,   key=lambda x: int(x.get("id", 0)))
        human_items = sorted(human_items, key=lambda x: int(x.get("id", 0)))

        # 2) Get plain text lists
        llm_texts, human_texts = await extract_texts_from_items(llm_items, human_items)

        # 3) Turn sentences into vectors using the embedding model
        llm_embeds, human_embeds = await asyncio.gather(
            embed_texts(llm_texts),
            embed_texts(human_texts),
        )

        # 4) Compute similarity for each aligned pair (same sequence)
        pair_sims, n_pairs = await compute_pair_similarities(
            llm_embeds=llm_embeds,
            human_embeds=human_embeds,
        )

        # 5) Find which ones have low similarity
        llm_count = llm_embeds.shape[0]
        human_count = human_embeds.shape[0]

        low_indices, _ = await find_low_similarity_items(
            pair_sims=pair_sims,
            llm_count=llm_count,
            human_count=human_count,
            n_pairs=n_pairs,
            low_limit=LOW_SIMILARITY_LIMIT,
        )

        # 6) Build per-criterion results
        per_criterion_results = await build_per_criterion_results(
            criteria_type=criteria_type,
            llm_items=llm_items,
            llm_texts=llm_texts,
            human_texts=human_texts,
            pair_sims=pair_sims,
            low_indices=low_indices,
            n_pairs=n_pairs,
        )

        # 7) Summary: total / passed / failed
        summary = summarize_semantic_results(
            per_criterion_results=per_criterion_results,
            threshold=LOW_SIMILARITY_LIMIT,
        )
        summary["criteria_type"] = criteria_type

        return {
            "criteria": per_criterion_results,
            "summary": summary,
        }

    except Exception as e:
        logger.exception(f"Error while evaluating {criteria_type.upper()} semantics: {e}")
        return default_result
