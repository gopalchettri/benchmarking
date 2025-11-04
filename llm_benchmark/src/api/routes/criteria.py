# POST /benchmark/criteria - Run criteria gen tests
# GET /results/criteria/{execution_id} - Fetch results

# src/api/routers/criteria.py

from fastapi import APIRouter, Body
from src.api.models import (
    CriteriaBenchmarkRequest,
    CriteriaBenchmarkResponse,
    CriteriaBenchmarkResult
)
from src.usecases.criteria_generation.benchmark import run_criteria_benchmark

router = APIRouter()

@router.post(
    "/",
    response_model=CriteriaBenchmarkResponse,
    summary="Run Criteria Generation Benchmark",
    description="Runs model(s) on test case(s) with selected parameter set(s) for Criteria Generation use-case."
)
def run_criteria_benchmark_api(request: CriteriaBenchmarkRequest = Body(...)):
    """
    Launches the Criteria Generation benchmark for selected models, params, and test cases.
    """
    # Delegate to usecase runner & build response
    result_dict = run_criteria_benchmark(
        model_names=request.model_names,
        param_indices=request.param_indices,
        test_case_ids=request.test_case_ids
    )
    # Adapt to CriteriaBenchmarkResponse structure
    response = CriteriaBenchmarkResponse(
        results=[
            CriteriaBenchmarkResult(
                execution_id=result["execution_id"],
                model=result["model"],
                params=result["params"],
                test_case=result["test_case"],
                metrics=result["metrics"]
            ) for result in result_dict["results"]
        ]
    )
    return response

