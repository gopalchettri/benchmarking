"""
Evaluator for LLM-generated content
Optimized for performance, reusability, and maintainability
Save as: model_evaluator.py
"""

import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Iterable
import logging
import json
import re

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

# -----------------------
# Configuration / Globals
# -----------------------

# Ensure HF / Transformers run fully offline (prevent any Hub calls)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# Threadpool for blocking loads / encode offloading
_LOAD_EXECUTOR = ThreadPoolExecutor(max_workers=2)

# Default local model folder (adjust if your path differs)
# Expect the folder to contain files like config.json, pytorch_model.bin, tokenizer.json, sentence_bert_config.json
MODEL_LOCAL_DIR = "src/config/embedding_model/all-MiniLM-L6-v2_local"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# Data classes
# -----------------------


@dataclass
class EvaluationConfig:
    """Configuration for evaluation thresholds and weights"""
    semantic_weight: float = 0.30
    exact_match_weight: float = 0.15
    completeness_weight: float = 0.30
    precision_weight: float = 0.15
    keyword_weight: float = 0.10
    semantic_threshold: float = 0.75
    exact_match_threshold: float = 0.90
    embedding_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 32


@dataclass
class EvaluationResult:
    """Structured evaluation result"""
    control_id: str
    model: str
    framework: str
    semantic_similarity: float
    exact_match_score: float
    completeness_recall: float
    precision: float
    keyword_coverage: float
    f1_score: float
    composite_score: float
    expected_count: int
    generated_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------
# Utility functions
# -----------------------
import torch
from sentence_transformers import util

def inner_product_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Inner product (dot product) between two SentenceTransformer embeddings,
    computed via sentence_transformers.util.dot_score.
    """
    v1 = np.asarray(vec1, dtype=np.float32)
    v2 = np.asarray(vec2, dtype=np.float32)

    # Convert to torch tensors with batch dimension
    t1 = torch.from_numpy(v1).unsqueeze(0)  # shape: (1, dim)
    t2 = torch.from_numpy(v2).unsqueeze(0)  # shape: (1, dim)

    # util.dot_score returns a 1x1 tensor here
    ip = util.dot_score(t1, t2)[0][0].item()
    return float(ip)


def _select_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _validate_model_path(path: str) -> str:
    """
    Validate and return model path. Accepts either the exact model folder path
    or its parent directory (attempts to find the local model subfolder).
    """
    p = Path(path)
    if p.exists():
        # if p is a directory that contains transformer files, return it
        return str(p)
    # try common suffix
    alt = p / "all-MiniLM-L6-v2_local"
    if alt.exists():
        return str(alt)
    # try parent + suffix
    parent_alt = Path("src/config/embedding_model") / "all-MiniLM-L6-v2_local"
    if parent_alt.exists():
        return str(parent_alt)
    raise FileNotFoundError(f"Embedding model path not found: {path}. Checked: {p}, {alt}, {parent_alt}")


# -----------------------
# Main Evaluator
# -----------------------


class ModelEvaluator:
    """
    Evaluator for LLM-generated content

    - Async-friendly: model load happens off the event loop
    - Batched embedding generation, caching
    - Normalized embeddings for cosine similarity
    """

    def __init__(self, config: Optional[EvaluationConfig] = None, model_directory: Optional[str] = None):
        self.config = config or EvaluationConfig()
        self.model_dir = model_directory or MODEL_LOCAL_DIR
        self._encoder: Optional[SentenceTransformer] = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
        # a small executor for offloading encode if needed
        self._executor = ThreadPoolExecutor(max_workers=4)
        logger.info(f"ModelEvaluator initialized. model_dir={self.model_dir}")

    # -----------------
    # Model Loading
    # -----------------
    @property
    async def embedding_model(self) -> SentenceTransformer:
        """
        Lazy-load SentenceTransformer from local folder without blocking the asyncio loop.
        Returns the same instance on repeated calls (cached in self._encoder).
        """
        if self._encoder is not None:
            return self._encoder

        model_path = _validate_model_path(self.model_dir)
        loop = asyncio.get_event_loop()

        def _load_model():
            device = _select_device()
            logger.info(f"Loading SentenceTransformer from local path: {model_path} on device={device}")
            model = SentenceTransformer(model_path, device=device)
            # Warm-up a tiny call (no heavy work) to ensure everything is initialized
            try:
                model.encode(["warmup"], convert_to_numpy=True, show_progress_bar=False)
            except Exception:
                # ignore warmup failures; model still loaded
                pass
            return model

        try:
            self._encoder = await loop.run_in_executor(_LOAD_EXECUTOR, _load_model)
            logger.info("Local SentenceTransformer loaded successfully.")
            return self._encoder
        except Exception as e:
            logger.exception("Failed to load local SentenceTransformer", exc_info=True)
            raise

    # -----------------
    # Public API
    # -----------------
    async def evaluate_criteria(
        self,
        llm_content: List[str],
        test_case: Dict[str, Any],
        model_name: str
    ) -> EvaluationResult:
        """
        Main evaluation method - calculates all metrics asynchronously

        Args:
            llm_content: List[str] generated by LLM
            test_case: dict containing compare_content with expected_content
            model_name: name of the model being evaluated (for records)
        """
        # Validate inputs
        if not isinstance(llm_content, list):
            raise ValueError("llm_content must be a list")
        if not isinstance(test_case, dict):
            raise ValueError("test_case must be a dictionary")

        compare_content = test_case.get("compare_content", {})
        expected_content = compare_content.get("expected_content", [])

        if not llm_content or not compare_content:
            return self._create_empty_result(test_case, model_name)

        # Pre-compute embeddings once (awaits model load if needed)
        llm_generated_content_embeddings, expected_content_embeddings = await self._compute_embeddings_batch(
            llm_content, expected_content
        )

        # Run concurrent metric computations
        scores = await asyncio.gather(
            self._calculate_semantic_similarity(llm_generated_content_embeddings, expected_content_embeddings),
            self._calculate_exact_match(llm_content, expected_content),
            self._calculate_completeness(llm_generated_content_embeddings, expected_content_embeddings, compare_content),
            self._calculate_precision(llm_generated_content_embeddings, expected_content_embeddings),
            self._calculate_keyword_coverage(llm_content, expected_content),
            return_exceptions=True
        )

        # Handle exceptions in concurrent results
        for i, score in enumerate(scores):
            if isinstance(score, Exception):
                logger.error(f"Error calculating metric {i}: {score}", exc_info=True)
                scores[i] = 0.0

        semantic_sim, exact_match, completeness, precision, keyword_cov = scores

        f1_score = self._calculate_f1(precision, completeness)
        composite_score = self._calculate_composite(semantic_sim, exact_match, completeness, precision, keyword_cov)

        return EvaluationResult(
            control_id=test_case.get("control_id", "unknown"),
            model=model_name,
            framework=test_case.get("framework", ""),
            semantic_similarity=round(semantic_sim, 4),
            exact_match_score=round(exact_match, 4),
            completeness_recall=round(completeness, 4),
            precision=round(precision, 4),
            keyword_coverage=round(keyword_cov, 4),
            f1_score=round(f1_score, 4),
            composite_score=round(composite_score, 4),
            expected_count=compare_content.get("expected_count", len(expected_content)),
            generated_count=len(llm_content)
        )

    # -----------------
    # Embedding helpers
    # -----------------
    async def _compute_embeddings_batch(
        self,
        llm_generated: List[str],
        expected: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute embeddings in batches for better performance.
        Uses cache when possible.
        Returns (llm_gen_embeddings, exp_embeddings) as numpy arrays (normalized).
        """
        try:
            # await model
            embedding_model = await self.embedding_model

            # check cache
            llm_gen_cached = self._get_cached_embeddings(llm_generated)
            exp_cached = self._get_cached_embeddings(expected)

            if llm_gen_cached is not None:
                llm_gen_embeddings = llm_gen_cached
            else:
                # offload encoding to threadpool to keep loop free
                loop = asyncio.get_event_loop()
                llm_gen_embeddings = await loop.run_in_executor(
                    self._executor,
                    lambda: embedding_model.encode(
                        llm_generated,
                        batch_size=self.config.batch_size,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
                )
                # normalize
                llm_gen_embeddings = self._normalize(llm_gen_embeddings)
                self._set_cached_embeddings(llm_generated, llm_gen_embeddings)

            if exp_cached is not None:
                exp_embeddings = exp_cached
            else:
                loop = asyncio.get_event_loop()
                exp_embeddings = await loop.run_in_executor(
                    self._executor,
                    lambda: embedding_model.encode(
                        expected,
                        batch_size=self.config.batch_size,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
                )
                exp_embeddings = self._normalize(exp_embeddings)
                self._set_cached_embeddings(expected, exp_embeddings)

            if llm_gen_embeddings is None or exp_embeddings is None:
                raise ValueError("Failed to compute embeddings (one of them is None).")

            return llm_gen_embeddings, exp_embeddings

        except Exception as e:
            logger.error(f"_compute_embeddings_batch error: {e}", exc_info=True)
            raise

    def _get_cached_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        if not texts:
            return None
        # Use hash of joined texts to respect order
        key = str(hash("||".join(texts)))
        return self._embedding_cache.get(key)

    def _set_cached_embeddings(self, texts: List[str], emb: np.ndarray) -> None:
        if not texts:
            return
        key = str(hash("||".join(texts)))
        self._embedding_cache[key] = emb

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings is None:
            return embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    # -----------------
    # Metric calculations
    # -----------------
    async def _calculate_semantic_similarity(self, llm_gen_embeddings: np.ndarray, exp_embeddings: np.ndarray) -> float:
        similarity_matrix = cosine_similarity(llm_gen_embeddings, exp_embeddings)
        # Best-match per expected item
        best_matches = similarity_matrix.max(axis=0)
        return float(np.mean(best_matches)) if best_matches.size > 0 else 0.0

    async def _calculate_exact_match(self, llm_generated: List[str], expected: List[str]) -> float:
        matches = 0
        threshold = self.config.exact_match_threshold
        for exp in expected:
            exp_lower = exp.lower()
            for gen in llm_generated:
                ratio = SequenceMatcher(None, exp_lower, gen.lower()).ratio()
                if ratio >= threshold:
                    matches += 1
                    break
        return matches / len(expected) if expected else 0.0

    async def _calculate_completeness(
        self, llm_gen_embeddings: np.ndarray, exp_embeddings: np.ndarray, compare_content: Dict[str, Any]
    ) -> float:
        expected_count = compare_content.get("expected_count", exp_embeddings.shape[0] if exp_embeddings is not None else 0)
        if expected_count == 0:
            return 0.0
        similarity_matrix = cosine_similarity(exp_embeddings, llm_gen_embeddings)
        matches = (similarity_matrix.max(axis=1) > self.config.semantic_threshold).sum()
        return float(matches / expected_count)

    async def _calculate_precision(self, llm_gen_embeddings: np.ndarray, exp_embeddings: np.ndarray) -> float:
        if llm_gen_embeddings is None or llm_gen_embeddings.shape[0] == 0:
            return 0.0
        similarity_matrix = cosine_similarity(llm_gen_embeddings, exp_embeddings)
        correct = (similarity_matrix.max(axis=1) > self.config.semantic_threshold).sum()
        return float(correct / llm_gen_embeddings.shape[0])

    async def _calculate_keyword_coverage(self, llm_generated: List[str], expected: List[str]) -> float:
        # Extract keywords from expected content (simple heuristic)
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might',
            'must', 'can', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
            'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 's', 't', 'just', 'don', 'now', 'based', 'using'
        }
        expected_keywords = set()
        for criterion in expected:
            words = re.findall(r'\b[a-z]{3,}\b', criterion.lower())
            meaningful = [w for w in words if w not in stop_words]
            expected_keywords.update(meaningful)

        if not expected_keywords:
            return 1.0

        llm_text = " ".join(llm_generated).lower()
        found = sum(1 for kw in expected_keywords if kw in llm_text)
        return found / len(expected_keywords)

    # -----------------
    # Helpers
    # -----------------
    def _calculate_f1(self, precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _calculate_composite(self, semantic_sim: float, exact_match: float, completeness: float, precision: float, keyword_cov: float) -> float:
        return (
            self.config.semantic_weight * semantic_sim +
            self.config.exact_match_weight * exact_match +
            self.config.completeness_weight * completeness +
            self.config.precision_weight * precision +
            self.config.keyword_weight * keyword_cov
        )

    def _create_empty_result(self, test_case: Dict[str, Any], model_name: str) -> EvaluationResult:
        return EvaluationResult(
            control_id=test_case.get("control_id", "unknown"),
            model=model_name,
            framework=test_case.get("framework", ""),
            semantic_similarity=0.0,
            exact_match_score=0.0,
            completeness_recall=0.0,
            precision=0.0,
            keyword_coverage=0.0,
            f1_score=0.0,
            composite_score=0.0,
            expected_count=0,
            generated_count=0
        )

    def save_results(self, result: EvaluationResult, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'a') as f:
            json.dump(result.to_dict(), f)
            f.write('\n')

    def clear_cache(self) -> None:
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")

    def close(self) -> None:
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass


# -----------------------
# Factory & Example usage
# -----------------------


def create_evaluator_for_use_case(use_case: str, model_directory: Optional[str] = None) -> ModelEvaluator:
    """
    Factory function to create evaluator with use-case-specific config
    (Synchronous factory; model is loaded lazily on first use)
    """
    configs = {
        'criteria': EvaluationConfig(
            semantic_weight=0.35,
            completeness_weight=0.35,
            precision_weight=0.20,
            exact_match_weight=0.05,
            keyword_weight=0.05
        ),
        'question': EvaluationConfig(
            semantic_weight=0.35,
            precision_weight=0.35,
            completeness_weight=0.20,
            exact_match_weight=0.05,
            keyword_weight=0.05
        ),
        'evidence': EvaluationConfig(
            semantic_weight=0.55,
            precision_weight=0.25,
            completeness_weight=0.10,
            exact_match_weight=0.05,
            keyword_weight=0.05
        ),
        'policy': EvaluationConfig(
            semantic_weight=0.50,
            exact_match_weight=0.25,
            precision_weight=0.15,
            completeness_weight=0.05,
            keyword_weight=0.05
        ),
        'general': EvaluationConfig()
    }
    config = configs.get(use_case.lower(), EvaluationConfig())
    logger.info(f"Created evaluator for use case: {use_case}")
    return ModelEvaluator(config=config, model_directory=model_directory or MODEL_LOCAL_DIR)


async def example_usage():
    """Demonstrate how to use the evaluator"""
    # Create evaluator (sync factory)
    evaluator = create_evaluator_for_use_case('criteria')

    # Sample test_case
    test_case = {
        "control_id": "IA-1.1.1",
        "control": "Access Control",
        "framework": "UAE IAR",
        "compare_content": {
            "expected_content": [
                "All users must be uniquely identified",
                "Access rights are granted based on least privilege",
                "Multi-factor authentication is enforced for privileged accounts"
            ],
            "expected_count": 3
        }
    }

    llm_generated_criteria = [
        "Every user shall have a unique identifier",
        "Access permissions follow least privilege principle",
        "Privileged users must use two-factor authentication"
    ]

    # Evaluate (this will load the local model lazily if not loaded)
    result = await evaluator.evaluate_criteria(
        llm_content=llm_generated_criteria,
        test_case=test_case,
        model_name="local-all-MiniLM-L6-v2"
    )

    # Print results
    print(f"\nEvaluation Results for {result.control_id}")
    print(f"Model: {result.model}")
    print(f"Semantic Similarity: {result.semantic_similarity}")
    print(f"Exact Match Score: {result.exact_match_score}")
    print(f"Completeness (Recall): {result.completeness_recall}")
    print(f"Precision: {result.precision}")
    print(f"Keyword Coverage: {result.keyword_coverage}")
    print(f"F1 Score: {result.f1_score}")
    print(f"Composite Score: {result.composite_score}")
    print(f"Expected Count: {result.expected_count}")
    print(f"Generated Count: {result.generated_count}")

    # Save to file
    output_path = Path("results/local_model_scores.jsonl")
    evaluator.save_results(result, output_path)
    print(f"\nResults saved to {output_path}")

    # cleanup
    evaluator.close()


# -----------------------
# Run example when executed as script
# -----------------------
if __name__ == "__main__":
    asyncio.run(example_usage())
