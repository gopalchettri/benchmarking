"""
Vision Preprocessing Script: create evidence_text_batch.json + Excel

This script:
  1. Reads ground_truth.json
  2. For each control and each evidence image, calls Azure OpenAI Vision
     (Llama-3.2-11B-Vision-Instruct) to extract text.
  3. Builds a list of records like:

     [
       {
         "control_id": "CTRL-001",
         "evidence_images": [
           {
             "image_id": "IMG-001-1",
             "file_path": "/evidence/ctrl-001/access_control_logs_screenshot.png",
             "extracted_text": "..."
           },
           ...
         ],
         "combined_evidence_text": "[IMAGE: IMG-001-1] ...\n\n[IMAGE: IMG-001-2] ..."
       },
       ...
     ]

  4. Writes:
       - evidence_text_batch.json
       - evidence_text_batch.xlsx
         * Sheet 'per_control' : one row per control
         * Sheet 'per_image'   : one row per evidence image
"""

import base64
import json
import os
from typing import Any, Dict, List

import pandas as pd
from openai import OpenAI  # pip install openai


# ======================================================
# CONFIG – EDIT THESE FOR YOUR ENVIRONMENT
# ======================================================

AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "https://<your-endpoint>.services.ai.azure.com/openai/v1/"
)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "<your-api-key>")

# Vision deployment name as in Azure AI Foundry / Azure OpenAI
VISION_DEPLOYMENT_NAME = os.getenv(
    "VISION_DEPLOYMENT_NAME",
    "Llama-3.2-11B-Vision-Instruct"
)


# ======================================================
# 0. OpenAI client helper
# ======================================================

def get_openai_client() -> OpenAI:
    """
    Create a reusable OpenAI client for Azure OpenAI.
    """
    return OpenAI(
        base_url=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
    )


# ======================================================
# 1. Load ground truth
# ======================================================

def load_ground_truth(json_path: str) -> List[Dict[str, Any]]:
    """
    Load ground truth controls from JSON.

    Expected per record (simplified for this script):

      {
        "control_id": "CTRL-001",
        "evidence_images": [
          {"image_id": "IMG-001-1", "file_path": "/evidence/ctrl-001/....png"},
          ...
        ],
        ...
      }

    Other fields (framework, evidence, etc.) are ignored in this script.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        item.setdefault("evidence_images", [])
    return data


# ======================================================
# 2. Vision – extract text from images
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
    Extract readable text from a single evidence image using
    Azure OpenAI Vision (Llama-3.2-11B-Vision-Instruct).

    Returns plain text. Empty string on failure.
    """
    # Normalise path
    if not os.path.isabs(image_path):
        image_path = os.path.abspath(image_path)

    if not os.path.exists(image_path):
        print(f"[WARN] Image not found: {image_path}")
        return ""

    # Read file bytes
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


# ======================================================
# 3. Build evidence_text_batch data structure
# ======================================================

def build_evidence_text_records(
    ground_truth_json_path: str
) -> List[Dict[str, Any]]:
    """
    Build the list of records that will go into evidence_text_batch.json.

    For each control:
      - Extract text for each evidence image.
      - Create:
          {
            "control_id": "...",
            "evidence_images": [
              { "image_id": "...", "file_path": "...", "extracted_text": "..." },
              ...
            ],
            "combined_evidence_text": "[IMAGE: IMG-xxx] ...\n\n[IMAGE: IMG-yyy] ..."
          }
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

            # Path normalisation
            if not os.path.isabs(file_path):
                file_path_to_use = os.path.abspath(file_path)
            else:
                file_path_to_use = file_path

            extracted = extract_text_from_image(file_path_to_use)

            # New image entry with extracted_text
            new_img = dict(img)
            new_img["file_path"] = file_path_to_use
            new_img["extracted_text"] = extracted or ""
            processed_images.append(new_img)

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

    return output_records


# ======================================================
# 4. Save to JSON and Excel
# ======================================================

def save_evidence_text_json(records: List[Dict[str, Any]], json_path: str) -> None:
    """
    Save evidence text records to JSON.
    """
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[INFO] JSON written to: {json_path}")


def save_evidence_text_excel(records: List[Dict[str, Any]], excel_path: str) -> None:
    """
    Export evidence text data to Excel with two sheets:

      - Sheet 'per_control':
          * control_id
          * combined_evidence_text

      - Sheet 'per_image':
          * control_id
          * image_id
          * file_path
          * extracted_text
    """
    # Build per-control rows
    per_control_rows: List[Dict[str, Any]] = []
    # Build per-image rows
    per_image_rows: List[Dict[str, Any]] = []

    for rec in records:
        control_id = rec["control_id"]
        combined_text = rec.get("combined_evidence_text", "")

        per_control_rows.append(
            {
                "control_id": control_id,
                "combined_evidence_text": combined_text,
            }
        )

        for img in rec.get("evidence_images", []):
            per_image_rows.append(
                {
                    "control_id": control_id,
                    "image_id": img.get("image_id", ""),
                    "file_path": img.get("file_path", ""),
                    "extracted_text": img.get("extracted_text", ""),
                }
            )

    df_control = pd.DataFrame(per_control_rows)
    df_image = pd.DataFrame(per_image_rows)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_control.to_excel(writer, sheet_name="per_control", index=False)
        df_image.to_excel(writer, sheet_name="per_image", index=False)

    print(f"[INFO] Excel written to: {excel_path}")


# ======================================================
# 5. Orchestration
# ======================================================

def run_vision_preprocessing(
    ground_truth_json: str,
    output_json: str,
    output_excel: str,
) -> None:
    """
    Orchestrate:
      1. Build evidence_text_batch records (calling Vision).
      2. Save to JSON.
      3. Save to Excel.
    """
    print("[INFO] Starting Vision preprocessing...")
    records = build_evidence_text_records(ground_truth_json_path=ground_truth_json)

    save_evidence_text_json(records, output_json)
    save_evidence_text_excel(records, output_excel)

    print("[INFO] Vision preprocessing completed.")


# ======================================================
# 6. Example main
# ======================================================

if __name__ == "__main__":
    """
    Example usage:

      1. Adjust paths and CONFIG at the top of this file.
      2. Run:
           python vision_preprocess.py
    """

    # Adjust these paths as per your project structure
    GROUND_TRUTH_JSON = "data/ground_truth.json"
    OUTPUT_JSON = "data/evidence_text_batch.json"
    OUTPUT_EXCEL = "data/evidence_text_batch.xlsx"

    run_vision_preprocessing(
        ground_truth_json=GROUND_TRUTH_JSON,
        output_json=OUTPUT_JSON,
        output_excel=OUTPUT_EXCEL,
    )
