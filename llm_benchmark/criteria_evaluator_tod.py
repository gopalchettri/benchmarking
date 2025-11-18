import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration & Logging
# ============================================================

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# IMPORTANT: Use all-mpnet-base-v2 for better semantic quality
MODEL_NAME = "all-mpnet-base-v2"
MODEL = SentenceTransformer(MODEL_NAME)

# Thread pool for running embedding in background
EXECUTOR = ThreadPoolExecutor(max_workers=4)

# Thresholds
SIMILARITY_THRESHOLD = 0.65   # used to flag "low similarity" items
GOOD_THRESHOLD_85 = 0.85      # computed internally (not exposed in JSON)
GOOD_THRESHOLD_90 = 0.90      # computed internally (not exposed in JSON)
LOW_SIMILARITY_LIMIT = SIMILARITY_THRESHOLD


# ============================================================
# Embedding utilities
# ============================================================

async def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Convert a list of texts into numerical embeddings using SentenceTransformer.
    Embeddings are L2-normalized so that dot product == cosine similarity.

    Returns:
        np.ndarray of shape (len(texts), dim)
    """
    if not texts:
        dim = MODEL.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        EXECUTOR,
        lambda: MODEL.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True  # important for cosine via dot product
        )
    )


# ============================================================
# Extraction utilities (TOD / TOE)
# ============================================================

def extract_criteria_lists_for_type(
    llm_case: Dict[str, Any],
    human_case: Dict[str, Any],
    criteria_type: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract LLM and Human criteria lists for a given criteria_type: 'tod' or 'toe'.

    Expected structure:
      llm_case["llm_content"]["llm_tod_criterias"] / ["llm_toe_criterias"]
      human_case["human_content"]["human_tod_criterias"] / ["human_toe_criterias"]
    """
    if criteria_type not in ("tod", "toe"):
        raise ValueError(f"criteria_type must be 'tod' or 'toe', got {criteria_type}")

    llm_key = f"llm_{criteria_type}_criterias"
    human_key = f"human_{criteria_type}_criterias"

    llm_content = llm_case.get("llm_content", {})
    human_content = human_case.get("human_content", {})

    llm_list = llm_content.get(llm_key, []) or []
    human_list = human_content.get(human_key, []) or []

    if not llm_list:
        logger.warning(f"No LLM {criteria_type.upper()} criterias found for control_id={llm_case.get('control_id')}.")
    if not human_list:
        logger.warning(f"No Human {criteria_type.upper()} criterias found for control_id={human_case.get('control_id')}.")

    return llm_list, human_list


def extract_texts_from_items(
    llm_items: List[Dict[str, Any]],
    human_items: List[Dict[str, Any]]
) -> Tuple[List[str], List[str]]:
    """
    From LLM & Human items, extract just the 'criteria' texts as lists of strings.
    """
    llm_texts = [item.get("criteria", "") for item in llm_items]
    human_texts = [item.get("criteria", "") for item in human_items]
    return llm_texts, human_texts


# ============================================================
# Similarity calculations
# ============================================================

def compute_pair_similarities(
    llm_embeds: np.ndarray,
    human_embeds: np.ndarray
) -> Tuple[List[float], int]:
    """
    Compute similarity score for each aligned pair (same index).

    Returns:
      - list of similarity scores
      - number of pairs actually compared
    """
    n_pairs = min(llm_embeds.shape[0], human_embeds.shape[0])

    if n_pairs == 0:
        return [], 0

    # Since embeddings are normalized, dot product == cosine similarity
    pair_sims = np.sum(
        llm_embeds[:n_pairs] * human_embeds[:n_pairs],
        axis=1
    ).astype(float).tolist()

    return pair_sims, n_pairs


def find_low_similarity_items(
    pair_sims: List[float],
    llm_count: int,
    human_count: int,
    n_pairs: int,
    low_limit: float
) -> Tuple[List[int], int]:
    """
    Find which LLM criteria have low similarity compared to human criteria.

    - low similarity = similarity < low_limit
    - extra LLM criteria (no human pair) are also treated as low similarity.

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

        # Extra LLM-only items
        if llm_count > n_pairs:
            low_indices.extend(range(n_pairs, llm_count))

    count = len(low_indices)
    return low_indices, count


# ============================================================
# Result building helpers
# ============================================================

def build_per_criterion_results(
    criteria_type: str,
    llm_items: List[Dict[str, Any]],
    llm_texts: List[str],
    human_texts: List[str],
    pair_sims: List[float],
    low_indices: List[int],
    n_pairs: int,
) -> List[Dict[str, Any]]:
    """
    Build one record per criterion pair with:
      - human text
      - LLM text
      - similarity score
      - whether it has low similarity

    Field names are renamed based on criteria_type:
      - TOD: human_tod_criteria, llm_tod_criteria
      - TOE: human_toe_criteria, llm_toe_criteria
    """
    results: List[Dict[str, Any]] = []

    # Determine field names based on type
    if criteria_type == "tod":
        human_field = "human_tod_criteria"
        llm_field = "llm_tod_criteria"
    else:  # "toe"
        human_field = "human_toe_criteria"
        llm_field = "llm_toe_criteria"

    # LLM + Human pairs
    for i in range(n_pairs):
        sim = pair_sims[i]
        results.append({
            "criteria_type": criteria_type,
            "criteria_id": llm_items[i].get("id"),
            human_field: human_texts[i],
            llm_field: llm_texts[i],
            "similarity": round(sim, 4),
            "is_low_similarity": i in low_indices,
        })

    # Extra LLM-only criteria (no human counterpart)
    for i in range(n_pairs, len(llm_texts)):
        results.append({
            "criteria_type": criteria_type,
            "criteria_id": llm_items[i].get("id"),
            human_field: "",
            llm_field: llm_texts[i],
            "similarity": 0.0,
            "is_low_similarity": True,
        })

    return results


def build_overall_metrics(
    criteria_type: str,
    pair_sims: List[float],
) -> Dict[str, Any]:
    """
    Build a summary for this criteria type.
    NOTE:
      - We DO compute average similarity, above_0_85, above_0_90 internally,
        but we DO NOT expose them in the JSON (as per your requirement).
      - Only criteria_type and total_pairs are returned.
    """
    # Internal calculations (not returned)
    average_similarity = float(np.mean(pair_sims)) if pair_sims else 0.0
    above_85 = sum(1 for s in pair_sims if s >= GOOD_THRESHOLD_85)
    above_90 = sum(1 for s in pair_sims if s >= GOOD_THRESHOLD_90)

    logger.info(
        f"[{criteria_type.upper()}] average_similarity={average_similarity:.4f}, "
        f">=0.85: {above_85}, >=0.90: {above_90}, total_pairs={len(pair_sims)}"
    )

    # Only exposing minimal info in JSON
    return {
        "criteria_type": criteria_type,
        "total_pairs": len(pair_sims),
    }


# ============================================================
# Core evaluation (per control, per criteria type)
# ============================================================

async def evaluate_semantic_for_criteria_type(
    llm_case: Dict[str, Any],
    human_case: Dict[str, Any],
    criteria_type: str = "tod",
) -> Dict[str, Any]:
    """
    Compare LLM-generated criteria vs human criteria for a single criteria type.

    criteria_type: "tod" or "toe"

    Expected inputs:
      - llm_case: dict with keys:
          control_id, control, subcontrol, framework,
          llm_content.{llm_tod_criterias / llm_toe_criterias}
      - human_case: dict with keys:
          control_id, control, subcontrol, framework,
          human_content.{human_tod_criterias / human_toe_criterias}

    Returns:
      {
        "overall_metrics": { "criteria_type": ..., "total_pairs": ... },
        "per_criterion_results": [ ... one entry per criterion pair ... ]
      }
    """
    default_result = {"overall_metrics": {}, "per_criterion_results": []}

    try:
        # 1) Extract the raw criteria lists for this type (TOD/TOE)
        llm_items, human_items = extract_criteria_lists_for_type(
            llm_case=llm_case,
            human_case=human_case,
            criteria_type=criteria_type,
        )

        # 2) Get plain text lists
        llm_texts, human_texts = extract_texts_from_items(llm_items, human_items)

        # 3) Turn sentences into vectors using the embedding model
        llm_embeds, human_embeds = await asyncio.gather(
            embed_texts(llm_texts),
            embed_texts(human_texts)
        )

        # 4) Compute similarity for each aligned pair
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

        # 6) Build per-criterion results
        per_criterion_results = build_per_criterion_results(
            criteria_type=criteria_type,
            llm_items=llm_items,
            llm_texts=llm_texts,
            human_texts=human_texts,
            pair_sims=pair_sims,
            low_indices=low_indices,
            n_pairs=n_pairs,
        )

        # 7) Overall metrics (minimal JSON, but logs full stats)
        overall_metrics = build_overall_metrics(
            criteria_type=criteria_type,
            pair_sims=pair_sims,
        )

        return {
            "overall_metrics": overall_metrics,
            "per_criterion_results": per_criterion_results
        }

    except Exception:
        logger.exception("Semantic evaluation failed")
        return default_result


# ============================================================
# Control-level evaluation: join TOD & TOE in a single JSON
# ============================================================

async def evaluate_control(
    llm_case: Dict[str, Any],
    human_case: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate both TOD and TOE for a single control, returning a combined JSON:

    {
      "control_id": ...,
      "control": ...,
      "subcontrol": ...,
      "framework": ...,
      "tod": { ... },
      "toe": { ... }
    }
    """
    # Make sure top-level metadata is taken from human_case if available
    control_id = human_case.get("control_id") or llm_case.get("control_id", "")
    control = human_case.get("control") or llm_case.get("control", "")
    subcontrol = human_case.get("subcontrol") or llm_case.get("subcontrol", "")
    framework = human_case.get("framework") or llm_case.get("framework", "")

    tod_result = await evaluate_semantic_for_criteria_type(llm_case, human_case, "tod")
    toe_result = await evaluate_semantic_for_criteria_type(llm_case, human_case, "toe")

    return {
        "control_id": control_id,
        "control": control,
        "subcontrol": subcontrol,
        "framework": framework,
        "tod": tod_result,
        "toe": toe_result,
    }


# ============================================================
# Excel export helpers
# ============================================================

def flatten_control_results_for_excel(control_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    control_results: list of per-control dicts in the agreed JSON structure:
      {
        "control_id": ...,
        "control": ...,
        "subcontrol": ...,
        "framework": ...,
        "tod": { ... },
        "toe": { ... }
      }

    Returns: pandas DataFrame ready for export to Excel.
    """
    rows: List[Dict[str, Any]] = []

    for ctrl in control_results:
        framework = ctrl.get("framework", "")
        control_id = ctrl.get("control_id", "")
        control = ctrl.get("control", "")
        subcontrol = ctrl.get("subcontrol", "")

        for criteria_type in ["tod", "toe"]:
            section = ctrl.get(criteria_type)
            if not section:
                continue

            for item in section.get("per_criterion_results", []):
                if criteria_type == "tod":
                    human_text = item.get("human_tod_criteria", "")
                    llm_text = item.get("llm_tod_criteria", "")
                else:  # "toe"
                    human_text = item.get("human_toe_criteria", "")
                    llm_text = item.get("llm_toe_criteria", "")

                sim = float(item.get("similarity", 0.0))
                status = "Good" if not item.get("is_low_similarity", False) else "Needs review"

                rows.append({
                    "Framework": framework,
                    "Control ID": control_id,
                    "Control": control,
                    "Subcontrol": subcontrol,
                    "Criteria Type": criteria_type.upper(),   # TOD / TOE
                    "Criteria ID": item.get("criteria_id"),
                    "Human Criteria": human_text,
                    "LLM Criteria": llm_text,
                    "Similarity (%)": round(sim * 100, 1),
                    "Status": status,
                })

    df = pd.DataFrame(rows)
    return df


def export_results_to_excel(control_results: List[Dict[str, Any]], excel_path: str) -> None:
    """
    Create a well-formatted Excel file for non-technical users.
    """
    df = flatten_control_results_for_excel(control_results)

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Benchmark Results")

        workbook = writer.book
        worksheet = writer.sheets["Benchmark Results"]

        # Header format: bold, wrapped
        header_format = workbook.add_format({"bold": True, "text_wrap": True, "valign": "top"})
        for col_num, col_name in enumerate(df.columns):
            worksheet.write(0, col_num, col_name, header_format)

        # Set column widths
        worksheet.set_column("A:A", 30)  # Framework
        worksheet.set_column("B:B", 12)  # Control ID
        worksheet.set_column("C:C", 35)  # Control
        worksheet.set_column("D:D", 60)  # Subcontrol
        worksheet.set_column("E:E", 14)  # Criteria Type
        worksheet.set_column("F:F", 11)  # Criteria ID
        worksheet.set_column("G:G", 80)  # Human Criteria
        worksheet.set_column("H:H", 80)  # LLM Criteria
        worksheet.set_column("I:I", 14)  # Similarity (%)
        worksheet.set_column("J:J", 16)  # Status

        # Wrap text for long columns
        wrap_format = workbook.add_format({"text_wrap": True, "valign": "top"})
        worksheet.set_column("D:D", 60, wrap_format)
        worksheet.set_column("G:H", 80, wrap_format)

        # Conditional formatting: color scale on Similarity (%)
        last_row = len(df) + 1  # +1 for header row
        worksheet.conditional_format(
            f"I2:I{last_row}",
            {
                "type": "3_color_scale",
                "min_color": "#F8696B",  # red
                "mid_color": "#FFEB84",  # yellow
                "max_color": "#63BE7B",  # green
            }
        )

        # Conditional formatting: highlight "Needs review"
        needs_review_format = workbook.add_format({"bg_color": "#F4CCCC"})
        worksheet.conditional_format(
            f"J2:J{last_row}",
            {
                "type": "formula",
                "criteria": '=$J2="Needs review"',
                "format": needs_review_format
            }
        )


# ============================================================
# Demo / example usage
# ============================================================

async def main_demo() -> None:
    """
    Small demo showing how to:
      - Evaluate a single control (TOD + TOE)
      - Print JSON
      - Export to Excel for that one control
    Replace the mock llm_case and human_case with your real JSONs.
    """

    # --- Mock example using your structure ---

    llm_case = {
        "control_id": "T1.2.1.2",
        "control": "Inventory of Information and other Associated Assets",
        "subcontrol": "11.2.1.2: The entity shall identify and maintain an up-to-date inventory of information assets within the entity.",
        "framework": "UAE Information Assurance Regulation",
        "llm_content": {
            "llm_tod_criterias": [
                {"id": 1, "criteria": "Existence of a formal policy on maintaining an updated inventory of information assets."},
                {"id": 2, "criteria": "Clear definition of information assets within the organization."},
                {"id": 3, "criteria": "Procedures for updating inventory items whenever changes occur."}
            ],
            "llm_toe_criterias": [
                {"id": 1, "criteria": "Extract required asset details including identifiers from inventory lists."},
                {"id": 2, "criteria": "Verify last updated dates for assets, prioritizing oldest entries."},
                {"id": 3, "criteria": "Identify asset categories and check their presence across inventories."}
            ]
        }
    }

    human_case = {
        "control_id": "T1.2.1.2",
        "control": "Inventory of Information and other Associated Assets",
        "subcontrol": "11.2.1.2: The entity shall identify and maintain an up-to-date inventory of information assets within the entity.",
        "framework": "UAE Information Assurance Regulation",
        "human_content": {
            "human_tod_criterias": [
                {"id": 1, "criteria": "Existence of a formal policy that mandates the identification and maintenance of an up-to-date inventory of information assets."},
                {"id": 2, "criteria": "Clear definition of ‘information assets’ within organizational documentation to ensure comprehensive inventory coverage."},
                {"id": 3, "criteria": "Procedures for regularly updating the inventory list to reflect changes in information assets, including additions, deletions, or modifications."}
            ],
            "human_toe_criterias": [
                {"id": 1, "criteria": "Extract asset details from each inventory and verify whether each information asset contains details such as identifier (e.g., hostname, serial number, asset tag, unique ID)."},
                {"id": 2, "criteria": "Extract the field that identifies the last update date. If there are multiple entries of the last update date, extract the oldest one from that, and cross verify to see if that is correct."},
                {"id": 3, "criteria": "Identify the information asset types defined in the organization's policy, and go through each asset entry from the uploaded inventories, and verify that each asset type mentioned is present."}
            ]
        }
    }

    # --- Evaluate one control ---
    control_result = await evaluate_control(llm_case, human_case)

    # Print JSON (for debugging)
    print(json.dumps(control_result, indent=2))

    # --- Export to Excel (single control example) ---
    export_results_to_excel([control_result], "llm_benchmark_results_demo.xlsx")
    print("Excel file 'llm_benchmark_results_demo.xlsx' generated.")


if __name__ == "__main__":
    asyncio.run(main_demo())
