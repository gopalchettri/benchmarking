"""
llm_clients.py

Complete unified LLM client module supporting both Ollama and Azure Phi-4.
Drop this file into your project and import: from llm_clients import get_llm_client

Author: Generated for GenAI Platform Integration
Date: 2025-12-18
"""

import json
import time
import logging
import asyncio
from typing import Dict, Any, Optional, List, Protocol
from dataclasses import dataclass

import aiohttp
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# Configuration Classes
# ============================================================

@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""
    ollama_uri: str
    model_name: str
    temperature: float = 0.5
    top_p: float = 0.9
    request_timeout: int = 120


@dataclass
class Phi4Config:
    """Configuration for Phi-4 Azure client."""
    api_url: str
    auth_token: str
    api_version: str
    model_name: str
    deployment_name: str
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 2048
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    seed: Optional[int] = None
    do_sample: Optional[bool] = None
    system_prompt: str = "You are a helpful assistant."


# ============================================================
# Protocol (Interface)
# ============================================================

class LLMClient(Protocol):
    """
    Provider-agnostic interface that all LLM clients must implement.
    
    This ensures consistency across different LLM providers.
    """

    async def generate_response(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a response for a given prompt.

        Args:
            prompt: User prompt/query string

        Returns:
            Dict containing:
            {
                "message": {"content": "<response_text>"},
                "usage": {
                    "input_tokens": int,
                    "output_tokens": int,
                },
                "provider": str,
                "model_name": str,
                "latency_ms": float (optional),
                "raw": dict (optional, provider-specific raw response)
            }
        """
        ...


# ============================================================
# Ollama Client Implementation
# ============================================================

class OllamaClient:
    """
    Async HTTP client for Ollama API.
    
    Communicates with Ollama via REST API and returns normalized responses.
    """

    def __init__(
        self,
        ollama_uri: str,
        model_name: str,
        temperature: float = 0.5,
        top_p: float = 0.9,
        request_timeout: int = 120,
    ) -> None:
        """
        Initialize Ollama client.

        Args:
            ollama_uri: Full URI to Ollama API endpoint (e.g., http://localhost:11434/api/chat)
            model_name: Name of the Ollama model to use
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Nucleus sampling parameter
            request_timeout: HTTP request timeout in seconds
        """
        if not ollama_uri or not isinstance(ollama_uri, str):
            raise ValueError("ollama_uri must be a non-empty string")
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")

        self._ollama_uri = ollama_uri
        self.model_name = model_name
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self._request_timeout = request_timeout

        logger.info(
            f"OllamaClient initialized: model='{self.model_name}', uri='{self._ollama_uri}'"
        )

    async def generate_response(self, prompt: str) -> Dict[str, Any]:
        """
        Generate response from Ollama model.

        Args:
            prompt: User prompt string

        Returns:
            Normalized response dictionary

        Raises:
            ValueError: If prompt is empty or response is malformed
            RuntimeError: If Ollama API request fails
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt must be a non-empty string")

        # Build Ollama API payload
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }

        logger.info(
            f"Sending request to Ollama: model='{self.model_name}', prompt_length={len(prompt)}"
        )

        start_time = time.perf_counter()

        try:
            # Make async HTTP request to Ollama
            timeout = aiohttp.ClientTimeout(total=self._request_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self._ollama_uri,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    response_text = await response.text()
                    status_code = response.status

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000.0

            # Check HTTP status
            if status_code != 200:
                logger.error(
                    f"Ollama API request failed: status={status_code}, "
                    f"response={response_text[:500]}"
                )
                raise RuntimeError(
                    f"Ollama API returned status {status_code}: {response_text[:200]}"
                )

            # Parse JSON response
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Ollama JSON response: {e}")
                raise ValueError(
                    f"Invalid JSON from Ollama: {response_text[:500]}"
                ) from e

            # Extract message content
            message = data.get("message")
            if not isinstance(message, dict):
                raise ValueError(
                    f"Ollama response missing 'message' dict. Got: {data}"
                )

            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError(
                    f"Ollama response missing 'message.content' string. Got: {message}"
                )

            # Extract token usage (if available)
            usage = data.get("usage") or {}
            input_tokens = usage.get("prompt_eval_count", 0)
            output_tokens = usage.get("eval_count", 0)

            logger.info(
                f"Ollama response received: latency={latency_ms:.2f}ms, "
                f"in_tokens={input_tokens}, out_tokens={output_tokens}"
            )

            # Return normalized response
            return {
                "message": {"content": content},
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
                "provider": "ollama",
                "model_name": self.model_name,
                "latency_ms": latency_ms,
                "raw": data,
            }

        except aiohttp.ClientError as e:
            logger.error(f"Network error communicating with Ollama: {e}")
            raise RuntimeError(f"Ollama network error: {e}") from e
        except asyncio.TimeoutError as e:
            logger.error(f"Ollama request timeout after {self._request_timeout}s")
            raise RuntimeError(
                f"Ollama request timeout after {self._request_timeout}s"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error in Ollama client: {e}", exc_info=True)
            raise


# ============================================================
# Phi-4 Base Client (Azure via LangChain)
# ============================================================

class Phi4ModelClient:
    """
    Client for Microsoft Phi-4 model via Azure AI using LangChain.
    
    Handles model initialization, message building, and async invocation.
    """

    def __init__(
        self,
        api_url: str,
        auth_token: str,
        api_version: str,
        model_name: str,
        deployment_name: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        max_tokens: int = 2048,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        do_sample: Optional[bool] = None,
    ) -> None:
        """
        Initialize Phi-4 model client.

        Args:
            api_url: Azure endpoint URL
            auth_token: Azure authentication token
            api_version: Azure API version
            model_name: Model identifier
            deployment_name: Azure deployment name
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            max_tokens: Maximum tokens to generate
            top_k: Top-k sampling parameter
            min_p: Minimum probability threshold
            repetition_penalty: Repetition penalty
            seed: Random seed for reproducibility
            do_sample: Whether to use sampling
        """
        self.api_url = api_url
        self.auth_token = auth_token
        self.api_version = api_version
        self.model_name = model_name
        self.deployment_name = deployment_name
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.seed = seed
        self.do_sample = do_sample

        if not self.api_url or not self.auth_token:
            raise ValueError("Azure api_url and auth_token must be provided")

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the Azure AI Chat Completions model."""
        try:
            # Build model_kwargs for additional parameters
            model_kwargs = {}
            if self.top_k is not None:
                model_kwargs["top_k"] = self.top_k
            if self.min_p is not None:
                model_kwargs["min_p"] = self.min_p
            if self.repetition_penalty is not None:
                model_kwargs["repetition_penalty"] = self.repetition_penalty
            if self.do_sample is not None:
                model_kwargs["do_sample"] = self.do_sample

            self.model = AzureAIChatCompletionsModel(
                endpoint=self.api_url,
                credential=self.auth_token,
                model=self.model_name,
                api_version=self.api_version,
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                max_tokens=self.max_tokens,
                seed=self.seed,
                model_kwargs=model_kwargs if model_kwargs else None,
            )

            logger.info(f"Phi-4 model '{self.model_name}' initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Phi-4 model '{self.model_name}': {e}")
            raise

    @staticmethod
    def _build_message(
        system_prompt: str,
        user_prompt: str,
    ) -> List[BaseMessage]:
        """
        Build LangChain message list for model invocation.

        Args:
            system_prompt: System instruction message
            user_prompt: User query message

        Returns:
            List of BaseMessage objects
        """
        if not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string")
        if not isinstance(user_prompt, str):
            raise TypeError("user_prompt must be a string")

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

    async def query_model_async(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int, int, float]:
        """
        Query Phi-4 model asynchronously.

        Args:
            system_prompt: System instruction
            user_prompt: User query

        Returns:
            Tuple of (response_text, input_tokens, output_tokens, latency_ms)
        """
        try:
            messages = self._build_message(system_prompt, user_prompt)

            logger.info(
                f"Querying Phi-4 model '{self.model_name}', "
                f"user_prompt_length={len(user_prompt)}"
            )

            start_time = time.perf_counter()

            # Run synchronous invoke in thread pool to avoid blocking
            response = await asyncio.to_thread(self.model.invoke, messages)

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000.0

            # Extract usage metadata
            usage_metadata = getattr(response, "usage_metadata", {}) or {}
            input_tokens = usage_metadata.get("input_tokens", 0)
            output_tokens = usage_metadata.get("output_tokens", 0)

            # Extract response content
            try:
                response_text = response.content
            except AttributeError as e:
                raise ValueError(
                    f"Unexpected Phi-4 response format: {response}"
                ) from e

            logger.info(
                f"Phi-4 response received: latency={latency_ms:.2f}ms, "
                f"in_tokens={input_tokens}, out_tokens={output_tokens}"
            )

            return response_text, input_tokens, output_tokens, latency_ms

        except Exception as e:
            logger.error(f"Error during Phi-4 query: {e}", exc_info=True)
            raise


# ============================================================
# Phi-4 Adapter (matches LLMClient interface)
# ============================================================

class Phi4LLMClient:
    """
    Adapter wrapping Phi4ModelClient to match LLMClient interface.
    
    Provides same API as OllamaClient for drop-in replacement.
    """

    def __init__(
        self,
        api_url: str,
        auth_token: str,
        api_version: str,
        model_name: str,
        deployment_name: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        max_tokens: int = 2048,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        do_sample: Optional[bool] = None,
        system_prompt: str = "You are a helpful assistant.",
    ) -> None:
        """
        Initialize Phi-4 LLM client adapter.

        Args:
            api_url: Azure endpoint URL
            auth_token: Azure API key
            api_version: Azure API version
            model_name: Model name/ID
            deployment_name: Azure deployment name
            temperature: Sampling temperature
            top_p: Nucleus sampling
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            max_tokens: Max tokens to generate
            top_k: Top-k sampling
            min_p: Min probability threshold
            repetition_penalty: Repetition penalty
            seed: Random seed
            do_sample: Enable sampling
            system_prompt: Default system instruction
        """
        self.system_prompt = system_prompt

        self._phi_client = Phi4ModelClient(
            api_url=api_url,
            auth_token=auth_token,
            api_version=api_version,
            model_name=model_name,
            deployment_name=deployment_name,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            do_sample=do_sample,
        )

        logger.info(f"Phi4LLMClient initialized with model '{model_name}'")

    async def generate_response(self, prompt: str) -> Dict[str, Any]:
        """
        Generate response from Phi-4 model (matches LLMClient interface).

        Args:
            prompt: User prompt string

        Returns:
            Normalized response dictionary

        Raises:
            ValueError: If prompt is empty or response is malformed
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt must be a non-empty string")

        text, input_tokens, output_tokens, latency_ms = await self._phi_client.query_model_async(
            system_prompt=self.system_prompt,
            user_prompt=prompt,
        )

        return {
            "message": {"content": text},
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            "provider": "phi4",
            "model_name": self._phi_client.model_name,
            "latency_ms": latency_ms,
        }


# ============================================================
# Factory Function
# ============================================================

def get_llm_client(
    provider: str,
    **config: Any
) -> LLMClient:
    """
    Factory function to instantiate the correct LLM client based on provider.

    Args:
        provider: Either "ollama" or "phi4"
        **config: Provider-specific configuration parameters

    Returns:
        LLMClient implementation (OllamaClient or Phi4LLMClient)

    Raises:
        ValueError: If provider is unknown or config is invalid

    Examples:
        # Ollama client
        client = get_llm_client(
            provider="ollama",
            ollama_uri="http://localhost:11434/api/chat",
            model_name="llama3",
            temperature=0.7,
            top_p=0.9,
        )

        # Phi-4 client
        client = get_llm_client(
            provider="phi4",
            api_url="https://your-endpoint.openai.azure.com",
            auth_token="your-key",
            api_version="2024-08-01-preview",
            model_name="phi-4",
            deployment_name="phi4-deploy",
            temperature=0.7,
        )

        # Use client (same interface for both)
        response = await client.generate_response("Your prompt here")
        answer = response["message"]["content"]
    """
    provider = provider.lower().strip()

    if provider == "ollama":
        required = ["ollama_uri", "model_name"]
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(
                f"Ollama provider requires: {', '.join(required)}. Missing: {missing}"
            )

        return OllamaClient(
            ollama_uri=config["ollama_uri"],
            model_name=config["model_name"],
            temperature=config.get("temperature", 0.5),
            top_p=config.get("top_p", 0.9),
            request_timeout=config.get("request_timeout", 120),
        )

    elif provider == "phi4":
        required = ["api_url", "auth_token", "api_version", "model_name", "deployment_name"]
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(
                f"Phi4 provider requires: {', '.join(required)}. Missing: {missing}"
            )

        return Phi4LLMClient(
            api_url=config["api_url"],
            auth_token=config["auth_token"],
            api_version=config["api_version"],
            model_name=config["model_name"],
            deployment_name=config["deployment_name"],
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.9),
            frequency_penalty=config.get("frequency_penalty", 0.0),
            presence_penalty=config.get("presence_penalty", 0.0),
            max_tokens=config.get("max_tokens", 2048),
            top_k=config.get("top_k"),
            min_p=config.get("min_p"),
            repetition_penalty=config.get("repetition_penalty"),
            seed=config.get("seed"),
            do_sample=config.get("do_sample"),
            system_prompt=config.get("system_prompt", "You are a helpful assistant."),
        )

    else:
        raise ValueError(
            f"Unknown provider: '{provider}'. Supported: 'ollama', 'phi4'"
        )


# ============================================================
# Example Usage (for testing)
# ============================================================

if __name__ == "__main__":
    async def test_ollama():
        """Test Ollama client."""
        print("\n=== Testing Ollama Client ===")
        client = get_llm_client(
            provider="ollama",
            ollama_uri="http://localhost:11434/api/chat",
            model_name="llama3",
            temperature=0.5,
        )
        
        response = await client.generate_response(
            "Explain what is RAG in AI in one sentence."
        )
        print(f"Response: {response['message']['content']}")
        print(f"Tokens: {response['usage']}")
        print(f"Latency: {response.get('latency_ms', 0):.2f}ms")

    async def test_phi4():
        """Test Phi-4 client."""
        print("\n=== Testing Phi-4 Client ===")
        client = get_llm_client(
            provider="phi4",
            api_url="https://your-azure-endpoint.openai.azure.com",
            auth_token="YOUR_AZURE_KEY",
            api_version="2024-08-01-preview",
            model_name="phi-4",
            deployment_name="phi4-deploy",
            temperature=0.7,
        )
        
        response = await client.generate_response(
            "Explain what is RAG in AI in one sentence."
        )
        print(f"Response: {response['message']['content']}")
        print(f"Tokens: {response['usage']}")
        print(f"Latency: {response.get('latency_ms', 0):.2f}ms")

    # Run tests
    asyncio.run(test_ollama())
    # asyncio.run(test_phi4())  # Uncomment when you have Azure credentials
