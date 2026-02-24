#!/usr/bin/env python3
"""
Semantic Similarity: criteria vs generated_criteria
Model : Alibaba-NLP/gte-multilingual-base
Input : all_controls_update.json
Output: <input>_semantic_score.json
"""

import json
import warnings
import numpy as np

from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_FILE   = "all_controls_updated.json"
OUTPUT_SCORE = INPUT_FILE.rsplit(".", 1)[0] + "_semantic_score.json"

# ─── MODE: "local" or "api" ──────────────────────────────────────────────────
MODE = "api" # "local" = SentenceTransformer, "api" = remote GTE endpoint

# API config — matches the deployed GTE inference service
API_URL   = "https://inference.ethreats.local/models/da099356-c9b5-408c-b75f-69f34e8924c7/proxy/v1/embeddings"
API_KEY   = "sk-nYtek6h8j9crD-orECG04a2FZDRN67HbbujPTSg5DqU"
API_MODEL = "Alibaba-NLP/gte-multilingual-base"

# Local config (used when MODE = "local")
MODEL_NAME = "gte-multilingual-base_local"
DEVICE     = "cpu"    # change to "cuda" if you have a GPU
BATCH_SIZE = 5

# API concurrency — how many batch requests to run in parallel
API_CONCURRENCY = 4
# ──────────────────────────────────────────────────────────────────────────────


def load_json(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_to_list(value) -> list:
    """Ensure value is always a clean list of strings."""
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def split_criteria_sentences(criteria_list: list) -> list:
    """Split criteria on newlines and strip leading bullet markers (e.g. '- ')."""
    sentences = []
    for text in criteria_list:
        for line in text.split("\n"):
            cleaned = line.strip().lstrip("-").strip()
            if cleaned:
                sentences.append(cleaned)
    return sentences


# ─── ENCODING: LOCAL ─────────────────────────────────────────────────────────

def encode_texts_local(model, texts: list) -> np.ndarray:
    """Encode via local SentenceTransformer. Adds 'query: ' prefix for GTE models."""
    if "gte" in MODEL_NAME.lower():
        texts = [f"query: {t}" for t in texts]
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


# ─── ENCODING: API (async httpx, matching deployed service pattern) ──────────

async def _post_batch(client, headers: dict, batch: list) -> list:
    """Post a single batch to the GTE embedding API and return vectors."""
    payload = {
        "model": API_MODEL,
        "input": batch,
        "encoding_format": "float",
    }
    response = await client.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return [item["embedding"] for item in data["data"]]


async def encode_texts_api_async(texts: list) -> np.ndarray:
    """
    Encode texts via the remote GTE API using async httpx with concurrency.
    Returns L2-normalized numpy array of shape (N, dim).
    """
    import asyncio
    import httpx

    headers = {"Authorization": f"Bearer {API_KEY}"}
    semaphore = asyncio.Semaphore(API_CONCURRENCY)
    all_embeddings: list = [None] * len(texts)   # pre-allocate for ordering

    async with httpx.AsyncClient(verify=False, timeout=60.0) as client:

        async def _send(start: int, batch: list):
            async with semaphore:
                vectors = await _post_batch(client, headers, batch)
                for j, vec in enumerate(vectors):
                    all_embeddings[start + j] = vec

        tasks = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            tasks.append(_send(i, batch))

        await asyncio.gather(*tasks)

    embs = np.array(all_embeddings, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0          # guard against zero-norm
    embs /= norms
    return embs


def encode_texts_api(texts: list) -> np.ndarray:
    """Sync wrapper so main() stays synchronous."""
    import asyncio
    return asyncio.run(encode_texts_api_async(texts))


# ─── SIMILARITY COMPUTATION ──────────────────────────────────────────────────

def compute_similarity(c_embs: np.ndarray, g_embs: np.ndarray) -> dict:
    """
    Compute similarity between criteria and generated_criteria embeddings.
    Returns best_match_avg (for summary) and per_gen_max (for per-item scores).
    """
    sim_matrix = c_embs @ g_embs.T                       # (M, N)

    best_match_avg = round(float(sim_matrix.max(axis=1).mean()), 4)
    per_gen_max    = sim_matrix.max(axis=0)               # (N,) — max score per gen

    return {
        "best_match_avg": best_match_avg,
        "per_gen_max":    per_gen_max,
    }


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load JSON ─────────────────────────────────────────────────────────
    print(f"\n Loading: {INPUT_FILE}")
    data = load_json(INPUT_FILE)
    if isinstance(data, dict):
        data = list(data.values())
    print(f"   → {len(data)} records loaded\n")

    # ── 2. Load Model (local only) ───────────────────────────────────────────
    model = None
    if MODE == "local":
        from sentence_transformers import SentenceTransformer
        print(f" Loading model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME, device=DEVICE, trust_remote_code=True)
        print(f"   → Model ready on [{DEVICE.upper()}]\n")
    else:
        print(f" Using API: {API_URL}\n")

    # ── 3. Pre-parse ALL records once (avoids double-parsing) ────────────────
    print(" Collecting all texts for batch encoding ...")
    parsed_records = []        # (criterias, gen_crits) per record
    all_texts      = []
    offsets        = []        # (start_c, end_c, start_g, end_g) per record
    concat_offsets = []        # (idx_concat_c, idx_concat_g) or (None, None)

    for row in data:
        criterias = split_criteria_sentences(normalize_to_list(row.get("criteria", [])))
        gen_crits = normalize_to_list(row.get("generated_criteria", []))
        parsed_records.append((criterias, gen_crits))

        start_c = len(all_texts)
        all_texts.extend(criterias)
        end_c = len(all_texts)

        start_g = len(all_texts)
        all_texts.extend(gen_crits)
        end_g = len(all_texts)

        offsets.append((start_c, end_c, start_g, end_g))

        # Concatenated texts: join all criteria / generated_criteria into one string each
        if criterias and gen_crits:
            idx_cc = len(all_texts)
            all_texts.append(" ".join(criterias))
            idx_cg = len(all_texts)
            all_texts.append(" ".join(gen_crits))
            concat_offsets.append((idx_cc, idx_cg))
        else:
            concat_offsets.append((None, None))

    print(f"   Total texts to embed: {len(all_texts)}")

    # ── 4. Encode ALL texts in one pass ──────────────────────────────────────
    active_model = MODEL_NAME if MODE == "local" else API_MODEL
    print(f"\n Encoding with [{active_model}] ...")
    if MODE == "local":
        all_embs = encode_texts_local(model, all_texts)
    else:
        all_embs = encode_texts_api(all_texts)
    print(" Encoding complete!\n")

    # ── 5. Compute similarities + build output in ONE pass ───────────────────
    score_results = []

    print(" Computing pairwise similarities ...")
    for r_idx, row in enumerate(tqdm(data, desc="Records")):
        criterias, gen_crits = parsed_records[r_idx]
        start_c, end_c, start_g, end_g = offsets[r_idx]

        if not criterias or not gen_crits:
            sim_info = {"best_match_avg": None, "per_gen_max": np.array([])}
        else:
            c_embs = all_embs[start_c:end_c]
            g_embs = all_embs[start_g:end_g]
            sim_info = compute_similarity(c_embs, g_embs)

        # Per-generated-criteria scores (from numpy, no dict scanning)
        scored_gen = []
        for g_idx, text in enumerate(gen_crits):
            score = round(float(sim_info["per_gen_max"][g_idx]), 4) if g_idx < len(sim_info["per_gen_max"]) else None
            scored_gen.append({"text": text, "similarity_score": score})

        # Pairwise average: mean of all per-generated-criteria semantic scores
        valid_scores = [s["similarity_score"] for s in scored_gen if s["similarity_score"] is not None]
        pairwise_avg = round(float(np.mean(valid_scores)), 4) if valid_scores else None

        # Concatenated similarity: single score comparing joined criteria vs joined generated_criteria
        idx_cc, idx_cg = concat_offsets[r_idx]
        concat_sim = None
        concat_scored = None
        if idx_cc is not None and idx_cg is not None:
            concat_sim = round(float(np.dot(all_embs[idx_cc], all_embs[idx_cg])), 4)
            concat_scored = {
                "text": " ".join(gen_crits),
                "similarity_score": concat_sim,
            }

        score_results.append({
            "record_idx":              r_idx,
            "sheet_source":            row.get("sheet_source", ""),
            "control_type":            row.get("control_type", ""),
            "control_family":          row.get("control_family", ""),
            "control_sub_family":      row.get("control_sub_family", ""),
            "control":                 row.get("control", ""),
            "control_description":     row.get("control_description", ""),
            "sub_control":             row.get("sub_control", ""),
            "sub_control_description": row.get("sub_control_description", ""),
            "evidence_request":        row.get("evidence_request", ""),
            "criteria":                row.get("criteria", []),
            "generated_criteria":      row.get("generated_criteria", []),
            "generated_criteria_pairwise_similarity_score": scored_gen,
            "generated_criteria_pairwise_avg_similarity_score":      pairwise_avg,
            "generated_criteria_concatenated_similarity_score": concat_scored,
        })

    # ── 6. Compute summary averages ──────────────────────────────────────────
    def _concat_score(r):
        cs = r["generated_criteria_concatenated_similarity_score"]
        return cs["similarity_score"] if cs is not None else None

    cc_all  = [s for r in score_results if (s := _concat_score(r)) is not None]
    cc_mgmt = [s for r in score_results if (s := _concat_score(r)) is not None and r["control_type"] == "Management Controls"]
    cc_tech = [s for r in score_results if (s := _concat_score(r)) is not None and r["control_type"] == "Technical Controls"]

    pw_all  = [r["generated_criteria_pairwise_avg_similarity_score"] for r in score_results if r["generated_criteria_pairwise_avg_similarity_score"] is not None]
    pw_mgmt = [r["generated_criteria_pairwise_avg_similarity_score"] for r in score_results if r["generated_criteria_pairwise_avg_similarity_score"] is not None and r["control_type"] == "Management Controls"]
    pw_tech = [r["generated_criteria_pairwise_avg_similarity_score"] for r in score_results if r["generated_criteria_pairwise_avg_similarity_score"] is not None and r["control_type"] == "Technical Controls"]

    summary = {
        "pairwise_similarity_score_avg_overall":                        round(float(np.mean(pw_all)), 4) if pw_all else None,
        "pairwise_similarity_score_avg_management_sub_controls":    round(float(np.mean(pw_mgmt)), 4) if pw_mgmt else None,
        "pairwise_similarity_score_avg_technical_sub_controls":     round(float(np.mean(pw_tech)), 4) if pw_tech else None,
        "concatenated_similarity_score_avg_overall":                    round(float(np.mean(cc_all)), 4) if cc_all else None,
        "concatenated_similarity_score_avg_management_sub_controls": round(float(np.mean(cc_mgmt)), 4) if cc_mgmt else None,
        "concatenated_similarity_score_avg_technical_sub_controls":  round(float(np.mean(cc_tech)), 4) if cc_tech else None,
    }

    output = {
        "summary": summary,
        "records": score_results,
    }

    # ── 7. Write output ──────────────────────────────────────────────────────
    with open(OUTPUT_SCORE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n Score JSON saved → {OUTPUT_SCORE}")

    # ── 8. Print summary ─────────────────────────────────────────────────────
    if pw_all or cc_all:
        print(f"\n{'─'*65}")
        print(f"  Model                                : {active_model}")
        print(f"  Records processed                    : {len(score_results)}")
        print(f"{'─'*65}")
        if pw_all:
            print(f"  Pairwise Avg (all)                   : {summary['pairwise_similarity_score_avg_overall']}")
        if pw_mgmt:
            print(f"  Pairwise Avg (Management)            : {summary['pairwise_similarity_score_avg_management_sub_controls']}")
        if pw_tech:
            print(f"  Pairwise Avg (Technical)             : {summary['pairwise_similarity_score_avg_technical_sub_controls']}")
        print(f"{'─'*65}")
        if cc_all:
            print(f"  Concatenated Avg (all)               : {summary['concatenated_similarity_score_avg_overall']}")
        if cc_mgmt:
            print(f"  Concatenated Avg (Management)        : {summary['concatenated_similarity_score_avg_management_sub_controls']}")
        if cc_tech:
            print(f"  Concatenated Avg (Technical)         : {summary['concatenated_similarity_score_avg_technical_sub_controls']}")
        print(f"{'─'*65}\n")


if __name__ == "__main__":
    main()
