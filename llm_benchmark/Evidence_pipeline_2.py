"""
LLM Evidence Analysis + Evaluation Pipeline (Simple, JSON-driven)

This version is aligned to the sample ground_truth.json structure:

[
  {
    "control_id": "CTRL-001",
    "framework": "UAE IAR",
    "evidence": "Access control logs showing ...",
    "evidence_images": [
      {"image_id": "IMG-001-1", "file_path": "/evidence/ctrl-001/access_control_logs_screenshot.png"},
      {"image_id": "IMG-001-2", "file_path": "/evidence/ctrl-001/authentication_dashboard.jpg"},
      ...
    ],
    "control_question": "Are access control mechanisms ...?",
    "expected_outcome": "All authentication attempts are logged ... "
  },
  ...
]

You ONLY need to implement:
  - extract_text_from_image()
  - call_text_llm()

Everything else is orchestration, prompt building, and evaluation.
"""

import json
import os
from typing import Any, Dict, List

# ======================================================
# 1. Load ground truth JSON
# ======================================================

def load_ground_truth(json_path: str) -> List[Dict[str, Any]]:
    """
    Load ground truth controls from JSON.

    Expected JSON: list of objects, each with keys like:
      - control_id (string)
      - framework (string)
      - evidence (string)
      - evidence_images (list of { image_id, file_path, ... })
      - control_question (string)
      - expected_outcome (string)

    Args:
        json_path: Path to the ground_truth.json file.

    Returns:
        List of dictionaries (one dict per control).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Set safe defaults for missing optional fields
    for item in data:
        item.setdefault("framework", "")
        item.setdefault("evidence", "")
        item.setdefault("evidence_images", [])
        item.setdefault("control_question", "")
        item.setdefault("expected_outcome", "")
        # Optional SME score; if you add it later, handle here:
        item.setdefault("sme_score", None)

    return data


# ======================================================
# 2. Vision / OCR – extract text from each image
# ======================================================

def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from a single evidence image.

    IMPORTANT:
        This is a placeholder function.
        Replace the body with your actual Vision/OCR implementation:
          - Vision LLM (OpenAI / Azure OpenAI, etc.)
          - Azure Computer Vision
          - Tesseract OCR
          - Any other OCR library

    Args:
        image_path: Full path to the image file (string).

    Returns:
        Extracted text from the image. Return empty string if nothing found.
    """
    # TODO: Implement your Vision/OCR logic here.
    #
    # Example pseudo-code:
    #
    # from some_vision_client import VisionClient
    # client = VisionClient(api_key="...")
    # text = client.extract_text(image_path)
    # return text
    #
    # For now, just return an empty string so pipeline can still run.
    return ""


def build_evidence_text_index(ground_truth: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    For each control_id, concatenate extracted text from all its evidence images.

    This function uses the "evidence_images" list inside each control record.
    Each evidence image entry is expected to have "file_path".

    Args:
        ground_truth: List of ground truth records loaded from JSON.

    Returns:
        A dictionary mapping:
            control_id -> combined evidence text (from all its images)
    """
    evidence_text_by_control: Dict[str, str] = {}

    for record in ground_truth:
        control_id = record["control_id"]
        image_entries = record.get("evidence_images", [])

        text_chunks: List[str] = []

        for img in image_entries:
            # Some entries may also have "file_name"; we rely on "file_path"
            file_path = img.get("file_path")
            if not file_path:
                continue

            # If file_path is relative, adjust it according to your project structure.
            extracted_text = extract_text_from_image(file_path)

            if extracted_text:
                label = img.get("image_id", os.path.basename(file_path))
                text_chunks.append(f"[IMAGE: {label}]\n{extracted_text}")

        combined_text = "\n\n".join(text_chunks)
        evidence_text_by_control[control_id] = combined_text

    return evidence_text_by_control


# ======================================================
# 3. Prompt building for Text LLM
# ======================================================

def build_analysis_prompt(
    control_id: str,
    framework: str,
    control_question: str,
    evidence_description: str,
    evidence_text: str
) -> str:
    """
    Build the user prompt for the Text LLM.

    NOTE:
      - We do NOT include expected_outcome here.
        It is used later only for evaluation, not for guiding the model.

    Args:
        control_id: Unique ID of the control (e.g. "CTRL-001").
        framework: Framework name (e.g. "UAE IAR").
        control_question: "What is the control asking for?"
        evidence_description: Short description from "evidence" field.
        evidence_text: Extracted text from all evidence images of this control.

    Returns:
        String prompt to be passed as "user" content to the LLM.
    """
    if not evidence_text:
        evidence_text = "[NO TEXT COULD BE EXTRACTED FROM THE EVIDENCE IMAGES]"

    prompt = f"""
You are a cyber security compliance assessor.

You are evaluating whether the provided evidence satisfies the control
from the "{framework}" framework.

Control ID: {control_id}
Control question: {control_question}

High-level evidence description (from auditor):
{evidence_description}

Extracted evidence text (from uploaded images):
------------------------------------------------
{evidence_text}
------------------------------------------------

Task:
1. Decide how well the evidence satisfies the control.
2. Provide:
   - adherence: an integer between 0 and 100 (higher = more compliant)
   - status: one of "Compliant", "Partially Compliant", or "Non-Compliant"
   - reason: a short paragraph explaining your decision and explicitly
             referencing key parts of the evidence (if available).

Return ONLY a valid JSON object with the following structure:

{{
  "adherence": 0-100 integer,
  "status": "Compliant" | "Partially Compliant" | "Non-Compliant",
  "reason": "short explanation"
}}
"""
    return prompt.strip()


# ======================================================
# 4. Text LLM call + JSON parsing
# ======================================================

def call_text_llm(
    model_name: str,
    system_prompt: str,
    user_prompt: str
) -> str:
    """
    Call your Text LLM and return the raw string response.

    IMPORTANT:
      This is a placeholder. Replace with your actual LLM client code.
      Example options:
        - Ollama (local)
        - Azure OpenAI / OpenAI
        - vLLM / custom inference server
        - Any OpenAI-compatible endpoint

    Args:
        model_name: Model identifier (string).
        system_prompt: System-level instruction for the LLM.
        user_prompt: User-level content (built by build_analysis_prompt).

    Returns:
        Raw response from the model as a string.
    """
    # TODO: Implement this for your environment.
    #
    # Example (pseudo-code for an OpenAI-compatible endpoint):
    #
    # from openai import OpenAI
    # client = OpenAI(base_url="http://<your-host>:<port>/v1", api_key="YOUR_KEY")
    # response = client.chat.completions.create(
    #     model=model_name,
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt},
    #     ],
    #     temperature=0.2,
    # )
    # return response.choices[0].message.content
    #
    raise NotImplementedError("call_text_llm() is not implemented. Plug in your model here.")


def parse_llm_json(raw_text: str) -> Dict[str, Any]:
    """
    Parse a JSON object out of the LLM response in a robust way.

    The LLM is instructed to return ONLY JSON, but this function:
      1. Tries json.loads() on the full string.
      2. If that fails, tries to extract the first {...} block and parse that.

    Args:
        raw_text: Raw string returned by the LLM.

    Returns:
        Python dict representing the JSON object.

    Raises:
        json.JSONDecodeError if parsing fails.
    """
    import re

    # First, try to parse the entire content as JSON
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # Second, try to find a {...} block in the text
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        json_str = match.group(0)
        return json.loads(json_str)

    # If we reach here, parsing failed completely
    raise json.JSONDecodeError("Could not parse LLM JSON output.", raw_text, 0)


def analyze_control_with_llm(
    model_name: str,
    control_record: Dict[str, Any],
    evidence_text: str
) -> Dict[str, Any]:
    """
    Run the Text LLM for a single control and return a simple result dict.

    Args:
        model_name: LLM model identifier (string).
        control_record: One entry from ground truth JSON (dict).
        evidence_text: Combined evidence text for this control (from images).

    Returns:
        Dict with keys:
          - control_id
          - model_name
          - adherence (float)
          - status (string)
          - reason (string)
    """
    control_id = control_record["control_id"]
    framework = control_record.get("framework", "")
    control_question = control_record.get("control_question", "")
    evidence_description = control_record.get("evidence", "")

    system_prompt = (
        "You are a cyber security compliance assessor. "
        "Always return a concise JSON object with adherence, status, and reason."
    )

    user_prompt = build_analysis_prompt(
        control_id=control_id,
        framework=framework,
        control_question=control_question,
        evidence_description=evidence_description,
        evidence_text=evidence_text,
    )

    # Call the model
    raw_response = call_text_llm(
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    # Parse JSON result
    parsed = parse_llm_json(raw_response)

    adherence = float(parsed.get("adherence", 0))
    status = str(parsed.get("status", "Partially Compliant"))
    reason = str(parsed.get("reason", "")).strip()

    return {
        "control_id": control_id,
        "model_name": model_name,
        "adherence": adherence,
        "status": status,
        "reason": reason,
    }


# ======================================================
# 5. Semantic similarity evaluation
# ======================================================

def compute_semantic_similarity(
    expected_text: str,
    generated_text: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> float:
    """
    Compute semantic similarity between SME expected_outcome and LLM reason.

    Uses SentenceTransformers cosine similarity in [0, 1].

    Args:
        expected_text: SME-written expected_outcome string.
        generated_text: LLM-generated reason string.
        embedding_model: Name of the SentenceTransformers model to use.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    from sentence_transformers import SentenceTransformer, util

    if not expected_text or not generated_text:
        return 0.0

    model = SentenceTransformer(embedding_model)
    embeddings = model.encode([expected_text, generated_text], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

    return float(similarity)


def evaluate_model_against_ground_truth(
    ground_truth: List[Dict[str, Any]],
    llm_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Compare LLM outputs with ground truth and compute evaluation metrics.

    For each control:
      - Compute semantic similarity between expected_outcome and LLM reason.
      - If sme_score is present, compute (llm_adherence - sme_score).

    Args:
        ground_truth: List of ground truth records (dicts).
        llm_results: List of LLM result dicts from analyze_control_with_llm().

    Returns:
        List of evaluation rows (dicts), one per control.
    """
    # Quick lookup: control_id -> ground truth record
    gt_by_control: Dict[str, Dict[str, Any]] = {
        item["control_id"]: item for item in ground_truth
    }

    evaluation_rows: List[Dict[str, Any]] = []

    for res in llm_results:
        control_id = res["control_id"]
        gt = gt_by_control.get(control_id)
        if not gt:
            # Skip if no matching ground truth (should not normally happen)
            continue

        expected_outcome = gt.get("expected_outcome", "")
        llm_reason = res.get("reason", "")

        similarity = compute_semantic_similarity(
            expected_text=expected_outcome,
            generated_text=llm_reason,
        )

        row: Dict[str, Any] = {
            "control_id": control_id,
            "framework": gt.get("framework", ""),
            "model_name": res["model_name"],
            "llm_adherence": res["adherence"],
            "llm_status": res["status"],
            "semantic_similarity": similarity,
            "expected_outcome": expected_outcome,
            "llm_reason": llm_reason,
        }

        sme_score = gt.get("sme_score")
        if sme_score is not None:
            row["sme_score"] = sme_score
            row["score_error"] = res["adherence"] - sme_score

        evaluation_rows.append(row)

    return evaluation_rows


# ======================================================
# 6. Orchestration helper for a single model
# ======================================================

def run_evidence_analysis_for_model(
    ground_truth_json: str,
    model_name: str,
    output_dir: str,
) -> None:
    """
    Orchestrate the full pipeline for a single model:

    Steps:
      1. Load ground truth JSON.
      2. Build evidence_text_by_control using images in evidence_images.
      3. Run the Text LLM per control.
      4. Evaluate against ground truth using semantic similarity.
      5. Save:
           - evidence_text_by_control.json
           - llm_results_<model_name>.json
           - evaluation_<model_name>.json

    Args:
        ground_truth_json: Path to ground_truth.json file.
        model_name: Model identifier (string).
        output_dir: Directory where output JSON files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: load ground truth
    ground_truth = load_ground_truth(ground_truth_json)

    # Step 2: extract evidence text from all images for each control_id
    evidence_text_by_control = build_evidence_text_index(ground_truth)

    # Save for debugging / traceability
    with open(
        os.path.join(output_dir, "evidence_text_by_control.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(evidence_text_by_control, f, indent=2, ensure_ascii=False)

    # Step 3: run the LLM per control
    llm_results: List[Dict[str, Any]] = []
    for record in ground_truth:
        control_id = record["control_id"]
        evidence_text = evidence_text_by_control.get(control_id, "")
        result = analyze_control_with_llm(
            model_name=model_name,
            control_record=record,
            evidence_text=evidence_text,
        )
        llm_results.append(result)

    # Save raw LLM results
    with open(
        os.path.join(output_dir, f"llm_results_{model_name}.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(llm_results, f, indent=2, ensure_ascii=False)

    # Step 4: evaluation
    evaluation_rows = evaluate_model_against_ground_truth(ground_truth, llm_results)

    with open(
        os.path.join(output_dir, f"evaluation_{model_name}.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(evaluation_rows, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Finished evidence analysis for model: {model_name}")
    print(f"  - Evidence text: {os.path.join(output_dir, 'evidence_text_by_control.json')}")
    print(f"  - LLM results:   {os.path.join(output_dir, f'llm_results_{model_name}.json')}")
    print(f"  - Evaluation:    {os.path.join(output_dir, f'evaluation_{model_name}.json')}")


# ======================================================
# 7. Example usage (edit and run)
# ======================================================

if __name__ == "__main__":
    """
    Example main usage.

    1. Make sure your ground_truth.json follows the sample structure.
    2. Implement:
         - extract_text_from_image()
         - call_text_llm()
    3. Adjust the paths and model names below.
    4. Run:
         python evidence_pipeline.py
    """

    GROUND_TRUTH_JSON = "data/ground_truth.json"  # update path as needed
    OUTPUT_DIR = "outputs"

    MODELS_TO_EVALUATE = [
        "phi-4-mini",      # Example placeholder
        # "gpt-oss:20b",
        # "falcon3-10b",
    ]

    for model in MODELS_TO_EVALUATE:
        run_evidence_analysis_for_model(
            ground_truth_json=GROUND_TRUTH_JSON,
            model_name=model,
            output_dir=OUTPUT_DIR,
        )
