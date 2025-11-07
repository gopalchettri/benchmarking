import asyncio
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import logging

from src.core.utils import load_json_async, save_json_async
from src.core.database import mongo_insert, mysql_save_metrics
from src.usecases.criteria_generation.prompts import get_criteria_prompt  # PROMPT MODULE
from src.usecases.criteria_generation.evaluator import evaluate_criteria_quality  # EVALUATOR MODULE
from registry.model_registry import get_supported_models
from src.core.model_client_factory import initialize_model_clients_from_config

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    test_case_id: str
    model: str
    criteria: Optional[str]
    scores: Dict[str, float]
    execution_time_ms: int
    status: str
    error: Optional[str] = None

@dataclass
class BenchmarkSummary:
    total_runs: int
    successful: int
    avg_relevance: float
    avg_execution_time: float
    best_model: Optional[str]
    test_cases_run: List[str]

async def load_and_validate_test_cases_async(
    test_cases_file: str,
    test_case_ids: Optional[List[str]] = None
) -> List[Dict]:
    data = await load_json_async(test_cases_file)
    all_cases = data if isinstance(data, list) else data.get("test_cases", [])
    # Validate necessary fields
    cases = [tc for tc in all_cases if all(k in tc for k in ("control_id", "control", "subcontrol"))]
    if test_case_ids:
        ids = set(test_case_ids)
        filtered = [tc for tc in cases if tc["control_id"] in ids]
        missing = ids - set(tc["control_id"] for tc in filtered)
        if missing:
            logger.warning(f"Missing test cases for control_ids: {missing}")
        cases = filtered
    if not cases:
        raise ValueError("No valid test cases found")
    return cases

async def load_parameter_sets_async(parameter_set_file: str, parameter_indices: List[int]) -> List[Dict]:
    params_data = await load_json_async(parameter_set_file)
    combos = params_data.get("parameter_combos", [])
    selected = [combos[i] for i in parameter_indices if 0 <= i < len(combos)]
    if not selected:
        raise ValueError("No valid parameter sets found")
    return selected

async def initialize_models_async(model_names: List[str]):
    supported = set(get_supported_models())
    valid = [m for m in model_names if m in supported]
    if not valid:
        raise ValueError("No supported models")
    return await initialize_model_clients_from_config({m: {} for m in valid})

async def save_raw_output_async(
    execution_id: str,
    test_case: Dict,
    model_name: str,
    params: Dict,
    criteria: str,
    framework: str
):
    await mongo_insert("criteria_raw", {
        "execution_id": execution_id,
        "test_case_id": test_case["control_id"],
        "model": model_name,
        "params": params,
        "criteria": criteria,
        "control": test_case["control"],
        "subcontrol": test_case["subcontrol"],
        "framework": framework,
        "metadata": {"timestamp": datetime.now().isoformat()}
    })

async def save_metrics_async(
    execution_id: str,
    model_name: str,
    params: Dict,
    test_case_id: str,
    criteria: str,
    eval_scores: Dict
):
    await mysql_save_metrics(
        execution_id=execution_id,
        model_name=model_name,
        parameter_combo=str(params),
        use_case="criteria_generation",
        test_case=test_case_id,
        model_output=criteria,
        relevance_score=eval_scores.get("relevance", 0.0),
        specificity_score=eval_scores.get("specificity", 0.0),
        policy_level_focus=eval_scores.get("policy_level_focus", 0.0),
        completeness_score=eval_scores.get("completeness", 0.0),
        clarity_score=eval_scores.get("clarity", 0.0),
        measurability_score=eval_scores.get("measurability", 0.0),
        compliance_alignment=eval_scores.get("alignment", 0.0),
        criteria_count=len([l for l in criteria.split("\n") if l.lstrip().startswith(("1.", "•", "-"))]),
        token_efficiency=len(criteria.split()) / (params.get("max_tokens", 512) or 1)
    )

async def run_single_test_case_async(
    model_clients: Dict[str, Any],
    model_name: str,
    params: Dict,
    test_case: Dict,
    execution_id: str,
    semaphore: asyncio.Semaphore,
    framework: Optional[str] = "UAE IA"
) -> BenchmarkResult:
    async with semaphore:
        client = model_clients.get(model_name)
        if not client:
            return BenchmarkResult(
                test_case_id=test_case["control_id"],
                model=model_name,
                criteria=None,
                scores={},
                execution_time_ms=0,
                status="error",
                error="Model not initialized"
            )
        try:
            start_time = datetime.now()
            framework = test_case.get("framework", framework)
            # PROMPT USAGE
            system_prompt, user_prompt = get_criteria_prompt(test_case["control"], test_case["subcontrol"], framework)
            # MODEL INFERENCE (async)
            criteria = await client.query_model_async(system_prompt, user_prompt)
            execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            # EVALUATOR USAGE
            eval_scores = evaluate_criteria_quality(criteria, test_case)
            # Async DB saves
            await asyncio.gather(
                save_raw_output_async(execution_id, test_case, model_name, params, criteria, framework),
                save_metrics_async(execution_id, model_name, params, test_case["control_id"], criteria, eval_scores)
            )
            return BenchmarkResult(
                test_case_id=test_case["control_id"],
                model=model_name,
                criteria=criteria,
                scores=eval_scores,
                execution_time_ms=execution_time_ms,
                status="success"
            )
        except Exception as e:
            logger.exception(f"Failure: {model_name} on {test_case['control_id']}")
            return BenchmarkResult(
                test_case_id=test_case["control_id"],
                model=model_name,
                criteria=None,
                scores={},
                execution_time_ms=0,
                status="error",
                error=str(e)
            )

def compute_benchmark_summary(results: List[BenchmarkResult]) -> BenchmarkSummary:
    successful = [r for r in results if r.status == "success"]
    n = len(successful)
    avg_relevance = sum(r.scores.get("relevance", 0) for r in successful) / n if n else 0
    avg_time = sum(r.execution_time_ms for r in successful) / n if n else 0
    best_model = max(successful, key=lambda x: x.scores.get("relevance", 0), default=None)
    return BenchmarkSummary(
        total_runs=len(results),
        successful=n,
        avg_relevance=avg_relevance,
        avg_execution_time=avg_time,
        best_model=best_model.model if best_model else None,
        test_cases_run=list({r.test_case_id for r in results})
    )

async def run_criteria_benchmark_async(
    model_names: List[str],
    parameter_indices: List[int],
    test_case_ids: Optional[List[str]] = None,
    test_cases_file: str = "config/test_cases_uc1.json",
    parameter_set_file: str = "config/parameter_set.json",
    max_concurrent: int = 10
) -> Dict[str, Any]:
    execution_id = str(uuid.uuid4())
    test_cases = await load_and_validate_test_cases_async(test_cases_file, test_case_ids)
    parameter_sets = await load_parameter_sets_async(parameter_set_file, parameter_indices)
    model_clients = await initialize_models_async(model_names)
    semaphore = asyncio.Semaphore(max_concurrent)
    combos = [
        (model_name, params, test_case)
        for model_name in model_clients.keys()
        for params in parameter_sets
        for test_case in test_cases
    ]
    tasks = [
        run_single_test_case_async(model_clients, model_name, params, test_case, execution_id, semaphore)
        for model_name, params, test_case in combos
    ]
    results: List[BenchmarkResult] = await asyncio.gather(*tasks)
    summary = compute_benchmark_summary(results)
    await save_json_async(
        {"execution_id": execution_id, "summary": summary.__dict__},
        f"results/criteria_summary_{execution_id}.json"
    )
    return {
        "execution_id": execution_id,
        "results": [r.__dict__ for r in results],
        "summary": summary.__dict__
    }
