import json
import pandas as pd
from pathlib import Path


def export_criteria_metrics_to_excel(json_path: str, excel_path: str) -> None:
    """
    Read your criteria_summary_*.json file and export criteria_metrics
    (TOD + TOE) into a single Excel sheet.
    """

    json_path = Path(json_path)
    excel_path = Path(excel_path)

    # 1. Load JSON
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    execution_id = data.get("execution_id")
    criterias = data.get("criterias", [])

    rows = []

    # 2. Loop over each test-case result
    for tc in criterias:
        model = tc.get("model")
        metrics = tc.get("criteria_metrics", {})

        # base info that is same for all criteria of this test case
        base = {
            "execution_id": execution_id,
            "model": model,
            "framework": metrics.get("framework"),
            "control_id": metrics.get("control_id"),
            "control": metrics.get("control"),
            "subcontrol": metrics.get("subcontrol"),
            "semantic_similarity_threshold": metrics.get(
                "semantic_similarity_threshold"
            ),
        }

        # ----- TOD criteria -----
        for crit in metrics.get("tod", {}).get("criteria", []):
            row = {
                **base,
                "criteria_type": "TOD",
                "criteria_id": crit.get("criteria_id"),
                "human_criteria": crit.get("human_tod_criteria"),
                "llm_criteria": crit.get("llm_tod_criteria"),
                "similarity": crit.get("similarity"),
                "is_low_similarity": crit.get("is_low_similarity"),
            }
            rows.append(row)

        # ----- TOE criteria -----
        for crit in metrics.get("toe", {}).get("criteria", []):
            row = {
                **base,
                "criteria_type": "TOE",
                "criteria_id": crit.get("criteria_id"),
                "human_criteria": crit.get("human_toe_criteria"),
                "llm_criteria": crit.get("llm_toe_criteria"),
                "similarity": crit.get("similarity"),
                "is_low_similarity": crit.get("is_low_similarity"),
            }
            rows.append(row)

    # 3. Convert to DataFrame and write to Excel
    df = pd.DataFrame(rows)
    df.to_excel(excel_path, index=False)

    print(f"Exported {len(df)} rows to {excel_path}")


# Example usage:
# export_criteria_metrics_to_excel(
#     "results/criteria_summary_712f2cbc-57d7-482e-b037-5a4d0a055595.json",
#     "results/criteria_metrics_export.xlsx",
# )
