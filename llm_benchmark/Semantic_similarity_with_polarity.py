"""
Semantic Similarity + Compliance Polarity Evaluation (Production Hardened)
==========================================================================
OPTIMIZATIONS:
- GPU/CPU Auto-detection
- Singleton Model Loading (saves RAM)
- Smart Regex for Polarity (Handles "no violations" vs "violation")
- Corrected NLI Direction (Ground Truth = Premise)
"""

import os
import re
import json
import yaml
import torch
import logging
import pandas as pd
from typing import Dict, Iterable, Tuple, Optional
from datetime import datetime
from pathlib import Path

# Set Offline Environment Variables FIRST
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =============================================================================
# Configuration & Logging
# =============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG = {
    "ground_truth_path": "ground_truth.json",
    "polarity_yaml_path": "compliance_polarity_phrases.yaml",
    "embedding_model": "embedding_model/all-mpnet-base-v2",
    "nli_model": "facebook/bart-large-mnli",
    "llm_model_name": "Phi4-14B",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# =============================================================================
# Model Manager (Singleton Pattern)
# =============================================================================
class ModelManager:
    """Lazy loader to prevent RAM spikes and handle GPU movement."""
    _embedder = None
    _nli_tokenizer = None
    _nli_model = None

    @classmethod
    def get_embedder(cls):
        if cls._embedder is None:
            logger.info(f"Loading Embedding Model on {CONFIG['device']}...")
            cls._embedder = SentenceTransformer(CONFIG["embedding_model"], device=CONFIG['device'])
        return cls._embedder

    @classmethod
    def get_nli_components(cls):
        if cls._nli_model is None:
            logger.info(f"Loading NLI Model on {CONFIG['device']}...")
            cls._nli_tokenizer = AutoTokenizer.from_pretrained(CONFIG["nli_model"])
            cls._nli_model = AutoModelForSequenceClassification.from_pretrained(CONFIG["nli_model"])
            cls._nli_model.to(CONFIG['device'])
            cls._nli_model.eval()
        return cls._nli_tokenizer, cls._nli_model

# =============================================================================
# Logic: Smart Polarity (Regex-Based)
# =============================================================================
def load_polarity_config(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Polarity YAML not found at {path}")
        return {}

def rule_based_polarity(text: str, config: Dict) -> Tuple[str, Dict]:
    """
    Determines polarity using Regex with Negative Lookbehind.
    Prevents false positives like "no violations" triggering "violation".
    """
    if not text or not config:
        return "neutral", {}
    
    text_l = text.lower()
    defs = config.get("polarity_definitions", {})

    # Critical: Check in order of Severity (Negative -> Partial -> Positive)
    for polarity in ["negative", "partial", "positive"]:
        phrases = defs[polarity].get("phrases", [])
        
        for phrase in phrases:
            # 1. Clean phrase for Regex
            clean_phrase = re.escape(phrase.lower())
            
            # 2. Build Smart Regex
            # \b          = Word boundary
            # (?<!no\s)   = Lookbehind: Ensure "no " is NOT before
            # (?<!not\s)  = Lookbehind: Ensure "not " is NOT before
            # (?<!zero\s) = Lookbehind: Ensure "zero " is NOT before
            pattern = fr'(?<!no\s)(?<!not\s)(?<!zero\s)\b{clean_phrase}\b'
            
            if re.search(pattern, text_l):
                return polarity, defs[polarity].get("adherence_band", {})
    
    return "neutral", {}

# =============================================================================
# Logic: AI Inference
# =============================================================================
def nli_inference(premise: str, hypothesis: str) -> Dict[str, float]:
    """
    Logic Check.
    PREMISE = Ground Truth (The Fact)
    HYPOTHESIS = LLM Reason (The Claim)
    """
    if not premise or not hypothesis:
        return {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}

    tokenizer, model = ModelManager.get_nli_components()

    inputs = tokenizer(
        premise, 
        hypothesis, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512
    ).to(CONFIG['device'])

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]

    # Map output using model's own config to avoid index errors
    id2label = model.config.id2label
    result = {}
    for idx, score in enumerate(probs):
        label_name = id2label[idx].lower()
        result[label_name] = round(score.item(), 4)
        
    return result

def compute_similarity(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0

    embedder = ModelManager.get_embedder()
    embeddings = embedder.encode(
        [text_a, text_b], 
        convert_to_tensor=True, 
        normalize_embeddings=True
    )
    
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return round(score * 100, 2)

# =============================================================================
# Core Pipeline
# =============================================================================
def semantic_similarity_and_excel_export(
    analysis_id: str, 
    analysis_response: Dict, 
    thresholds: Iterable[int] = (60, 65, 70, 75)
) -> None:

    result_dir = Path("Results") / CONFIG["llm_model_name"]
    result_dir.mkdir(parents=True, exist_ok=True)

    # Load External Data
    try:
        with open(CONFIG["ground_truth_path"], "r", encoding="utf-8") as f:
            ground_truth = json.load(f)
    except FileNotFoundError:
        logger.error("Ground Truth file missing.")
        return

    polarity_config = load_polarity_config(CONFIG["polarity_yaml_path"])

    # Parse Response
    completed_controls = (
        analysis_response.get("data", {})
        .get("analysis", {})
        .get("completed", [])
    )
    
    if not completed_controls:
        logger.warning("No completed controls found in response.")
        return

    rows = []
    assessment_id = analysis_response.get("data", {}).get("assessment_id")
    assessment_label_id = analysis_response.get("data", {}).get("assessment_label_id")

    for control in completed_controls:
        control_id = control.get("label_id")
        llm_reason = control.get("reason", "")
        
        # Ground Truth Lookup
        gt_entry = ground_truth.get(control_id, {})
        expected_output = gt_entry.get("expected_output", "")
        evidence = gt_entry.get("evidence", [])

        # -------------------------
        # 1. NLI Check (Strict Logic)
        # -------------------------
        nli_result = nli_inference(premise=expected_output, hypothesis=llm_reason)

        # -------------------------
        # 2. Polarity Check (Regex)
        # -------------------------
        llm_polarity, adherence_band = rule_based_polarity(llm_reason, polarity_config)
        expected_polarity, _ = rule_based_polarity(expected_output, polarity_config)
        
        polarity_match = (llm_polarity == expected_polarity)

        # -------------------------
        # 3. Gated Similarity Logic
        # -------------------------
        # Gate: If Polarity Mismatch OR Strong Contradiction -> Score is 0
        if nli_result.get("contradiction", 0) >= 0.50 or not polarity_match:
            similarity_score = 0.0
            gating_status = "Fail: Polarity or Logic Mismatch"
        else:
            similarity_score = compute_similarity(llm_reason, expected_output)
            gating_status = "Pass"

        rows.append({
            "Analysis ID": analysis_id,
            "Assessment ID": assessment_id,
            "Control ID": control_id,
            "Objective": control.get("objective"),
            "Adherence": control.get("adherence"),
            "Status": control.get("compliance"),
            "LLM Reason": llm_reason,
            "Expected Output": expected_output,
            "Evidence": " ; ".join(evidence),
            "Semantic Similarity Score": similarity_score,
            "Gating Status": gating_status,
            "LLM Polarity": llm_polarity,
            "Expected Polarity": expected_polarity,
            "Polarity Match": polarity_match,
            "NLI Entailment": nli_result.get("entailment", 0),
            "NLI Contradiction": nli_result.get("contradiction", 0),
            "Adherence Band": f"{adherence_band.get('min')}–{adherence_band.get('max')}"
        })

    # Export Logic
    base_df = pd.DataFrame(rows)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    for threshold in thresholds:
        df = base_df.copy()
        df["Below Semantic Threshold"] = df["Semantic Similarity Score"] < threshold
        df["Similarity Threshold"] = threshold

        file_name = f"{threshold}_Compliance_assessment_{CONFIG['llm_model_name']}_{timestamp}.xlsx"
        file_path = result_dir / file_name

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="compliance_assessment")
            ws = writer.sheets["compliance_assessment"]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

        logger.info(f"Report generated: {file_path}")
