import json
import logging
from typing import Any, Dict
import aiohttp

from app.models.error_models import LLMInteractionError
from app.utils.general_utils import (
    sanitize_llm_response_util,
    clean_and_load_json_string
)

# ---------------------------------------------------
# Logging Configuration
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LLMService_DeepSeek:
    """
    Azure AI Foundry adapter for DeepSeek v3.2
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        deployment_name: str,
        request_timeout: int = 120
    ):
        if not api_url or not api_key or not deployment_name:
            raise ValueError("api_url, api_key, and deployment_name are mandatory")

        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.request_timeout = request_timeout

        # -----------------------------
        # Model Parameters (TUNABLE)
        # -----------------------------
        self.temperature = 0.2
        self.top_p = 0.9
        self.presence_penalty = 0.0
        self.frequency_penalty = 0.0
        self.seed = 11
        self.max_tokens = 1024

        self.full_url = (
            f"{self.api_url}/chat/completions"
            f"?api-version=2024-05-01-preview"
        )

        logger.info(
            "DeepSeek v3.2 LLMService initialized. Deployment=%s",
            self.deployment_name
        )

    # ---------------------------------------------------
    # Main Generation Method
    # ---------------------------------------------------
    async def generate_response(self, prompt: str) -> Dict[str, Any]:
        """
        Sends a prompt to DeepSeek v3.2 and returns parsed JSON output
        """

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        payload = {
            "model": self.deployment_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict JSON-only response engine. "
                        "Return ONLY a valid JSON object. "
                        "Do not include markdown, explanations, or extra text."
                    )
                },
                {
                    "role": "user",
                    "content": prompt  # PASS PROMPT VERBATIM
                }
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "stream": False
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        raw_api_text = ""

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.request_timeout)
            ) as session:
                async with session.post(
                    self.full_url,
                    json=payload,
                    headers=headers
                ) as response:

                    raw_api_text = await response.text()
                    logger.debug(
                        "Raw DeepSeek response (status=%s): %s",
                        response.status,
                        raw_api_text[:1000]
                    )

                    if not response.ok:
                        raise LLMInteractionError(
                            message=f"DeepSeek API error: {response.status}",
                            status_code=response.status,
                            response_body=raw_api_text
                        )

            # ---------------------------------------------
            # Parse Azure Chat Completion Envelope
            # ---------------------------------------------
            try:
                response_data = json.loads(raw_api_text)
                llm_content = (
                    response_data["choices"][0]
                    ["message"]["content"]
                    .strip()
                )
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
                raise LLMInteractionError(
                    message="Invalid DeepSeek response structure",
                    status_code=500,
                    response_body=raw_api_text
                ) from e

            logger.debug("LLM raw content: %s", llm_content)

            # ---------------------------------------------
            # Sanitize + Parse JSON
            # ---------------------------------------------
            sanitized = sanitize_llm_response_util(llm_content)
            return clean_and_load_json_string(sanitized)

        except aiohttp.ClientError as e:
            logger.error(
                "Network error calling DeepSeek API: %s",
                str(e),
                exc_info=True
            )
            raise LLMInteractionError(
                message=f"HTTP client error: {e}",
                status_code=500,
                response_body=raw_api_text
            ) from e

        except Exception as e:
            logger.error(
                "Unexpected error during DeepSeek interaction: %s",
                str(e),
                exc_info=True
            )
            raise LLMInteractionError(
                message=f"Unexpected error: {e}",
                status_code=500,
                response_body=raw_api_text
            ) from e
