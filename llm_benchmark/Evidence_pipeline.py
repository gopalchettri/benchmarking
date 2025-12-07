"""
End-to-End Evidence Analysis Pipeline (Vision + Text) with Azure OpenAI

Flow:

1) Vision pre-processing (run occasionally)
   Input : ground_truth.json
   Output: evidence_text_batch.json
   - For each control and each evidence image:
       * Call Azure OpenAI Vision (Llama-3.2-11B-Vision-Instruct)
       * Save 'extracted_text' per image
       * Create 'combined_evidence_text' per control

2) Batch Text Analysis (run many times per model)
   Inputs: ground_truth.json, evidence_text_batch.json
   Outputs:
       - llm_results_<model>.json
       - evaluation_<model>.json

Expected ground_truth.json (simplified):

[
  {
    "control_id": "CTRL-001",
    "framework": "UAE IAR",
    "evidence": "Access control logs showing user authentication attempts ...",
    "evidence_images": [
      {"image_id": "IMG-001-1", "file_path": "/evidence/ctrl-001/access_control_logs_screenshot.png"},
      {"image_id": "IMG-001-2", "file_path": "/evidence/ctrl-001/authentication_dashboard.jpg"}
    ],
    "control_question": "Are access control mechanisms properly implemented ...?",
    "expected_outcome": "All authentication attempts are logged ...",
    "sme_score": 90
  }
]
"""

import base64
import json
import os
from typing import Any, Dict, List

from openai import OpenAI  # pip install openai


# ======================================================
# CONFIG – EDIT THESE FOR YOUR ENVIRONMENT
# ======================================================

AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "https://<your-endpoint>.services.ai.azure.com/openai/v1/"
)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "<your-api-key>")

# Vision deployment (image → text)
VISION_DEPLOYMENT_NAME = os.getenv(
    "VISION_DEPLOYMENT_NAME",
    "Llama-3.2-11B-Vision-Instruct"
)

# Text deployment (compliance reasoning)
TEXT_DEPLOYMENT_NAME = os.getenv(
    "TEXT_DEPLOYMENT_NAME",
    "gpt-4.1-mini"   # change to your text model deployment name
)


# ======================================================
# 0. OpenAI client helper
# ======================================================

def get_openai_client() -> OpenAI:
    """
    Reusable OpenAI client instance for Azure OpenAI.
    """
    return OpenAI(
        base_url=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
    )


# ======================================================
# 1. Ground truth loading
# ======================================================

def load_ground_truth(json_path: str) -> List[Dict[str, Any]]:
    """
    Load ground truth controls from JSON and normalise keys.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        item.setdefault("framework", "")
        item.setdefault("evidence", "")
        item.setdefault("evidence_images", [])
        item.setdefault("control_question", "")
        item.setdefault("expected_outcome", "")
        item.setdefault("sme_score", None)

    return data


# ======================================================
# 2. Vision – extract text from images and build batch JSON
# ======================================================

def _guess_image_mime_type(path: str) -> str:
    """
    Guess MIME type from file extension. Defaults to image/png.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".gif":
        return "image/gif"
    if ext == ".bmp":
        return "image/bmp"
    if ext == ".webp":
        return "image/webp"
    return "image/png"


def extract_text_from_image(image_path: str) -> str:
    """
    Call Azure OpenAI Vision deployment (Llama-3.2-11B-Vision-Instruct)
    to extract readable text from a single evidence image.

    Returns plain text; empty string on failure.
    """
    if not os.path.isabs(image_path):
        image_path = os.path.abspath(image_path)

    if not os.path.exists(image_path):
        print(f"[WARN] Image not found: {image_path}")
        return ""

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = _guess_image_mime_type(image_path)
    data_url = f"data:{mime_type};base64,{b64_image}"

    client = get_openai_client()

    try:
        response = client.chat.completions.create(
            model=VISION_DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract all readable text from this evidence image. "
                                "Return only the plain text, no JSON, no explanation."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                }
            ],
            max_tokens=512,
        )
        text = response.choices[0].message.content or ""
        return text.strip()
    except Exception as e:
        print(f"[ERROR] Vision extraction failed for {image_path}: {e}")
        return ""


def create_evidence_text_json(
    ground_truth_json_path: str,
    output_evidence_json_path: str,
) -> None:
    """
    PRE-PROCESSING STEP (Vision):

    For each control in ground_truth.json:
      - Call Vision model for every evidence image.
      - Store per-image 'extracted_text'.
      - Build 'combined_evidence_text' per control.

    Output JSON structure:

    [
      {
        "control_id": "CTRL-001",
        "evidence_images": [
          {
            "image_id": "...",
            "file_path": "...",
            "extracted_text": "..."
          }
        ],
        "combined_evidence_text": "[IMAGE: IMG-001-1] ...\\n\\n[IMAGE: IMG-001-2] ..."
      }
    ]
    """
    ground_truth = load_ground_truth(ground_truth_json_path)
    output_records: List[Dict[str, Any]] = []

    for record in ground_truth:
        control_id = record["control_id"]
        images = record.get("evidence_images", [])

        processed_images: List[Dict[str, Any]] = []
        combined_chunks: List[str] = []

        for img in images:
            file_path = img.get("file_path")
            if not file_path:
                continue

            if not os.path.isabs(file_path):
                file_path_to_use = os.path.abspath(file_path)
            else:
                file_path_to_use = file_path

            extracted = extract_text_from_image(file_path_to_use)

            new_img_entry = dict(img)
            new_img_entry["extracted_text"] = extracted or ""
            processed_images.append(new_img_entry)

            if extracted:
                label = img.get("image_id", os.path.basename(file_path))
                combined_chunks.append(f"[IMAGE: {label}]\n{extracted}")

        combined_text = "\n\n".join(combined_chunks)

        output_records.append(
            {
                "control_id": control_id,
                "evidence_images": processed_images,
                "combined_evidence_text": combined_text,
            }
        )

    with open(output_evidence_json_path, "w", encoding="utf-8") as f:
        json.dump(output_records, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Created evidence text JSON at: {output_evidence_json_path}")


def load_evidence_text_index(preprocessed_json_path: str) -> Dict[str, str]:
    """
    Load evidence_text_batch.json and build a simple mapping:

        control_id -> combined_evidence_text
    """
    with open(preprocessed_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    index: Dict[str, str] = {}
    for item in data:
        cid = item["control_id"]
        index[cid] = item.get("combined_evidence_text", "")
    return index


# ======================================================
# 3. Text LLM: prompt building, call, and parsing
# ======================================================

def build_analysis_prompt(
    control_id: str,
    framework: str,
    control_question: str,
    evidence_description: str,
    evidence_text: str,
) -> str:
    """
    Build the user prompt for the text LLM using the precomputed evidence text.
    """
    if not evidence_text:
        evidence_text = "[NO TEXT AVAILABLE FOR THIS CONTROL]"

    prompt = f"""
You are a cyber security compliance assessor.

You are evaluating whether the provided evidence satisfies the control
from the "{framework}" framework.

Control ID: {control_id}
Control question: {control_question}

High-level evidence description (from auditor):
{evidence_description}

Extracted evidence text (from preprocessed batch JSON):
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


def call_text_llm(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """
    Call the text LLM using Azure OpenAI / OpenAI SDK.
    """
    client = get_openai_client()

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=512,
    )

    return response.choices[0].message.content or ""


def parse_llm_json(raw_text: str) -> Dict[str, Any]:
    """
    Parse JSON from the text LLM response. Tries:
      1. json.loads on full string
      2. Extract first {...} block and parse
    """
    import json as _json
    import re

    try:
        return _json.loads(raw_text)
    except _json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        json_str = match.group(0)
        return _json.loads(json_str)

    raise _json.JSONDecodeError("Could not parse LLM JSON output.", raw_text, 0)


def analyze_control_with_llm(
    model_name: str,
    control_record: Dict[str, Any],
    evidence_text: str,
) -> Dict[str, Any]:
    """
    Run the text LLM for a single control using precomputed evidence_text.
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

    raw_response = call_text_llm(
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

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
# 4. Evaluation – semantic similarity vs SME ground truth
# ======================================================

def compute_semantic_similarity(
    expected_text: str,
    generated_text: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> float:
    """
    Compute semantic similarity between SME expected_outcome and LLM reason.
    """
    from sentence_transformers import SentenceTransformer, util

    if not expected_text or not generated_text:
        return 0.0

    model = SentenceTransformer(embedding_model)
    emb = model.encode([expected_text, generated_text], convert_to_tensor=True)
    sim = util.cos_sim(emb[0], emb[1]).item()
    return float(sim)


def evaluate_model_against_ground_truth(
    ground_truth: List[Dict[str, Any]],
    llm_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Compare LLM outputs with ground truth and compute evaluation metrics.
    """
    gt_by_control: Dict[str, Dict[str, Any]] = {
        item["control_id"]: item for item in ground_truth
    }

    rows: List[Dict[str, Any]] = []

    for res in llm_results:
        cid = res["control_id"]
        gt = gt_by_control.get(cid)
        if not gt:
            continue

        expected = gt.get("expected_outcome", "")
        reason = res.get("reason", "")

        sim = compute_semantic_similarity(expected_text=expected, generated_text=reason)

        row: Dict[str, Any] = {
            "control_id": cid,
            "framework": gt.get("framework", ""),
            "model_name": res["model_name"],
            "llm_adherence": res["adherence"],
            "llm_status": res["status"],
            "semantic_similarity": sim,
            "expected_outcome": expected,
            "llm_reason": reason,
        }

        sme_score = gt.get("sme_score")
        if sme_score is not None:
            row["sme_score"] = sme_score
            row["score_error"] = res["adherence"] - sme_score

        rows.append(row)

    return rows


# ======================================================
# 5. Orchestration helpers
# ======================================================

def preprocess_evidence_with_vision(
    ground_truth_json: str,
    output_evidence_json: str,
) -> None:
    """
    Wrapper to run the Vision pre-processing step.
    """
    create_evidence_text_json(
        ground_truth_json_path=ground_truth_json,
        output_evidence_json_path=output_evidence_json,
    )


def run_evidence_analysis_for_model(
    ground_truth_json: str,
    evidence_text_json: str,
    text_model_deployment: str,
    output_dir: str,
) -> None:
    """
    Full batch evidence analysis for a single text model.
    """
    os.makedirs(output_dir, exist_ok=True)

    ground_truth = load_ground_truth(ground_truth_json)
    evidence_text_by_control = load_evidence_text_index(evidence_text_json)

    # Optionally save the evidence mapping used
    with open(
        os.path.join(output_dir, "evidence_text_by_control_used.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(evidence_text_by_control, f, indent=2, ensure_ascii=False)

    llm_results: List[Dict[str, Any]] = []
    for record in ground_truth:
        cid = record["control_id"]
        ev_text = evidence_text_by_control.get(cid, "")
        result = analyze_control_with_llm(
            model_name=text_model_deployment,
            control_record=record,
            evidence_text=ev_text,
        )
        llm_results.append(result)

    with open(
        os.path.join(output_dir, f"llm_results_{text_model_deployment}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(llm_results, f, indent=2, ensure_ascii=False)

    eval_rows = evaluate_model_against_ground_truth(ground_truth, llm_results)
    with open(
        os.path.join(output_dir, f"evaluation_{text_model_deployment}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(eval_rows, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Finished batch evidence analysis for model: {text_model_deployment}")


# ======================================================
# 6. Example main
# ======================================================

if __name__ == "__main__":
    """
    Example usage.

    1) First run the Vision pre-processing to create evidence_text_batch.json
       (uncomment the call below the first time you run it).

    2) Then run the batch text analysis for each model you want to evaluate.
    """

    GROUND_TRUTH_JSON = "data/ground_truth.json"
    EVIDENCE_TEXT_JSON = "data/evidence_text_batch.json"
    OUTPUT_DIR = "outputs"

    # Step 1 – Vision preprocessing (run this when you want to regenerate)
    # preprocess_evidence_with_vision(
    #     ground_truth_json=GROUND_TRUTH_JSON,
    #     output_evidence_json=EVIDENCE_TEXT_JSON,
    # )

    # Step 2 – Batch text analysis
    MODELS = [TEXT_DEPLOYMENT_NAME]

    for model_name in MODELS:
        run_evidence_analysis_for_model(
            ground_truth_json=GROUND_TRUTH_JSON,
            evidence_text_json=EVIDENCE_TEXT_JSON,
            text_model_deployment=model_name,
            output_dir=OUTPUT_DIR,
        )
