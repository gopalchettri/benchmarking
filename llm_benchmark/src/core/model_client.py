# Configuration for all 6 LLMs (endpoints, parameters, deployment names)
#   LLM inference client: Handles API calls to 6 models
# - Manages authentication, 
# - Supports: phi-4, falcon-2, GPT-OSS, Llama 3.2, Jais, K2-think
# - Functions: generate_response(model, prompt, params)


# src/core/model_client.py
"""
Simple ModelClient for LLM REST API inference.
Designed for use with src/config/models.json and src/config/parameter_sets.json.
Usage:
    - Load model configs and param sets in your script.
    - For each model, create a ModelClient and call .generate(prompt, params).
"""

import requests

class ModelClient:
    def __init__(self, name, api_url, auth_token=None):
        self.name = name
        self.api_url = api_url
        self.auth_token = auth_token

    def generate(self, prompt, params):
        """
        Call the LLM REST API, return generated text output.
        - prompt: string
        - params: dictionary of generation parameters
        Returns: text result or "" if error.
        """
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        data = {"prompt": prompt}
        data.update(params)

        try:
            resp = requests.post(self.api_url, json=data, headers=headers, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            # Accept either "text" or "result" as output field
            return result.get("text", result.get("result", ""))
        except Exception as e:
            print(f"[ModelClient:{self.name}] Error: {e}")
            return ""

# ---------- Example Loader/Usage (not in this file!) ----------

# import json
# from src.core.model_client import ModelClient
#
# # Load models
# with open("src/config/models.json") as f:
#     model_configs = json.load(f)
# # Load param sets
# with open("src/config/parameter_sets.json") as f:
#     param_sets = json.load(f)
#
# # Initialize ModelClient objects
# clients = [
#     ModelClient(cfg["name"], cfg["api_url"], cfg.get("auth_token")) for cfg in model_configs
# ]
#
# # For each prompt/test, use:
# for client in clients:
#     for params in param_sets:
#         output = client.generate("Prompt info goes here", params)
#         print(client.name, params, output)
#
# --------------------------------------------------------------
