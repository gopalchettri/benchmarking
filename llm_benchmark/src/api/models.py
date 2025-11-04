# Pydantic models for request/response schemas
# - BenchmarkRequest, TestCaseInput, BenchmarkResult
# - Validates API payloads, ensures type safety


from pydantic import BaseModel
from typing import List, Dict, Any

class CriteriaBenchmarkRequest(BaseModel):
    model_names: List[str]
    param_indices: List[int]
    test_case_ids: List[str]

class CriteriaBenchmarkResult(BaseModel):
    execution_id: str
    model: str
    params: Dict[str, Any]
    test_case: str
    metrics: Dict[str, float]

class CriteriaBenchmarkResponse(BaseModel):
    results: List[CriteriaBenchmarkResult]


# class BenchmarkRequest(BaseModel):
#     model_names: List[str]
#     param_indices: List[int]
#     test_case_ids: List[str]
