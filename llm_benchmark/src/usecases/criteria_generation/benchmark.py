# LLM-based JSON schema generation and control extraction# Test executor for Criteria Generation
# - Loads 25 test cases from config/test_cases_uc1.json
# - Runs 6 models x parameter combinations (temp, top_p, etc.)
# - Stores outputs in MongoDB, metrics in MySQL
# - Main function: run_criteria_benchmark()


import json
from src.core.model_client import ModelClient
from src.core.metrics import calculate_relevance, calculate_specificity  # or your custom evals
from src.core.database import mongo_save_raw_output, mysql_save_metrics
import uuid

def run_criteria_benchmark(model_names, param_indices, test_case_ids):
    # Load configs for models, parameters, and test cases
    with open("src/config/models.json") as f:
        model_configs = json.load(f)
    
    # Load configs for parameters
    with open("src/config/parameter_sets.json") as f:
        param_sets = json.load(f)

    # Load configs for test cases   
    with open("src/config/test_cases_criteria_generation.json") as f:
        test_cases = json.load(f)

    # Filter models, param sets, and test cases
    selected_models = [m for m in model_configs if m["name"] in model_names]
    selected_params = [param_sets[i] for i in param_indices]

    # PENDING - need to work on the test case ids
    selected_tests = [tc for tc in test_cases if tc["control_id"] in test_case_ids]

    results = []
    # for m in selected_models:
    #     client = ModelClient(m["name"], m["api_url"], m.get("auth_token"))
    #     for params in selected_params:
    #         for tc in selected_tests:
    #             prompt = tc["description"]  # or use prompt template logic here
    #             output = client.generate(prompt, params)
    #             results.append({
    #                 "model": m["name"],
    #                 "params": params,
    #                 "test_case": tc["control_id"],
    #                 "output": output
    #             })

    for m in selected_models:
        client = ModelClient(m["name"], m["api_url"], m.get("auth_token"))
        for params in selected_params:
            for tc in selected_tests:
                prompt = tc["description"]
                output = client.generate(prompt, params)
                execution_id = str(uuid.uuid4())
                # Save raw output in Mongo
                mongo_save_raw_output(
                    model_name=m["name"],
                    use_case="criteria_generation",
                    test_case_id=tc["control_id"],
                    params=params,
                    prompt=prompt,
                    output=output
                )
                # Compute metrics (illustrative)
                metrics = {
                    "relevance": calculate_relevance(output, prompt),
                    "specificity": calculate_specificity(output)
                }
                # Save metrics in MySQL
                mysql_save_metrics(
                    execution_id=execution_id,
                    model_name=m["name"],
                    use_case="criteria_generation",
                    test_case_id=tc["control_id"],
                    metrics=metrics
                )
                # Also accumulate if you need a summary response
                result = {
                    "execution_id": execution_id,
                    "model": m["name"],
                    "params": params,
                    "test_case": tc["control_id"],
                    "llm_output": output,
                    "metrics": metrics
                }
                results.append(result)
    return {"results": results}


