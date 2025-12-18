"""
Semantic Similarity Evaluation & Excel Export
--------------------------------------------

• Uses analysis API response JSON (no DB dependency)
• Compares LLM Reason vs Expected Output (ground truth)
• Generates ONE Excel file per threshold
• Safe to call from FastAPI router (side-effect only)
Code to call the class
if response.data.status == AnalysisStatusEnum.COMPLETED:
    semantic_similarity_and_excel_export(
        analysis_response=response.model_dump(mode="json"),
        ground_truth_json_path=settings.GROUND_TRUTH_JSON_PATH,
        llm_model_name=settings.LLM_MODEL_NAME
    )
    
"""

from typing import Dict, Iterable
from datetime import datetime
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Core Function (CALL THIS FROM ROUTER)
# ============================================================
def semantic_similarity_and_excel_export(
    analysis_response: Dict,
    ground_truth_json_path: str,
    llm_model_name: str,
    thresholds: Iterable[int] = (60, 65, 70, 75),
) -> None:
    """
    Performs semantic similarity evaluation and exports Excel files.

    Excel Headers:
    Model
    Control ID
    Objective
    Adherence
    Status
    LLM Reason
    Expected Output
    Semantic Similarity Score
    Below Semantic Threshold
    Similarity Threshold
    """

    # --------------------------------------------------------
    # Load Ground Truth
    # --------------------------------------------------------
    with open(ground_truth_json_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    # --------------------------------------------------------
    # Load Embedding Model (Offline / Air-gapped safe)
    # --------------------------------------------------------
    embedder = SentenceTransformer("all-mpnet-base-v2")

    def compute_similarity(text_a: str, text_b: str) -> float:
        if not text_a or not text_b:
            return 0.0
        embeddings = embedder.encode(
            [text_a, text_b],
            normalize_embeddings=True
        )
        return round(
            float(cosine_similarity(
                [embeddings[0]],
                [embeddings[1]]
            )[0][0]) * 100,
            2
        )

    # --------------------------------------------------------
    # Extract Completed Controls from API Response
    # --------------------------------------------------------
    completed_controls = (
        analysis_response
        .get("data", {})
        .get("analysis", {})
        .get("completed", [])
    )

    if not completed_controls:
        # Nothing to evaluate
        return

    rows = []

    for control in completed_controls:
        control_id = control.get("label_id")
        llm_reason = control.get("reason", "")
        expected_output = ground_truth.get(control_id, "")

        similarity_score = compute_similarity(llm_reason, expected_output)

        rows.append({
            "Model": llm_model_name,
            "Control ID": control_id,
            "Objective": control.get("objective"),
            "Adherence": control.get("adherence"),
            "Status": control.get("compliance"),
            "LLM Reason": llm_reason,
            "Expected Output": expected_output,
            "Semantic Similarity Score": similarity_score
        })

    base_df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Export One Excel Workbook Per Threshold
    # --------------------------------------------------------
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    for threshold in thresholds:
        df = base_df.copy()
        df["Below Semantic Threshold"] = (
            df["Semantic Similarity Score"] < threshold
        )
        df["Similarity Threshold"] = threshold

        file_name = (
            f"Semantic_Evaluation_"
            f"{llm_model_name}_"
            f"Threshold_{threshold}_"
            f"{timestamp}.xlsx"
        )

        with pd.ExcelWriter(file_name, engine="openpyxl") as writer:
            df.to_excel(
                writer,
                index=False,
                sheet_name="Semantic Evaluation"
            )

            ws = writer.sheets["Semantic Evaluation"]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
