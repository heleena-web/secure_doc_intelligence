import re
from typing import Any, Optional
import httpx

from app.core.config import settings
from app.core.errors import (
    OllamaGenerationError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)


class OllamaClient:
    """Async client for interacting with the local Ollama LLM service.
    Guarantees that document data is never sent to external cloud APIs.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        temperature: Optional[float] = None,
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.default_model = model or settings.OLLAMA_MODEL
        self.timeout = timeout if timeout is not None else settings.OLLAMA_TIMEOUT_SECONDS
        self.temperature = temperature if temperature is not None else settings.OLLAMA_TEMPERATURE

    async def check_health(self) -> bool:
        """Checks if the local Ollama daemon is active and responding."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """Queries local Ollama tags endpoint to list all locally available model names."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                if res.status_code != 200:
                    raise OllamaUnavailableError(self.base_url)
                data = res.json()
                models = [
                    m.get("name", "")
                    for m in data.get("models", [])
                    if m.get("name")
                ]
                return models
        except httpx.ConnectError:
            raise OllamaUnavailableError(self.base_url)
        except httpx.TimeoutException:
            raise OllamaUnavailableError(self.base_url)
        except Exception as e:
            if isinstance(e, (OllamaUnavailableError, OllamaModelNotFoundError)):
                raise
            raise OllamaUnavailableError(self.base_url)

    async def validate_availability(self, model: Optional[str] = None) -> None:
        """Validates that local Ollama is running and the target model is installed.
        Raises OllamaUnavailableError or OllamaModelNotFoundError if not.
        """
        target_model = model or self.default_model
        available = await self.list_models()

        def model_matches(avail: str, target: str) -> bool:
            if avail.lower() == target.lower():
                return True
            if f"{avail.lower()}:latest" == target.lower() or avail.lower() == f"{target.lower()}:latest":
                return True
            # Match base name if target specified without tag
            if ":" not in target and avail.lower().startswith(f"{target.lower()}:"):
                return True
            return False

        if not any(model_matches(avail, target_model) for avail in available):
            raise OllamaModelNotFoundError(target_model, available)

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Sends inference request to local Ollama /api/generate endpoint.
        Handles reasoning model thinking separation, timeouts, and errors.
        """
        target_model = model or self.default_model
        temp = temperature if temperature is not None else self.temperature

        payload: dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temp,
            },
        }
        if system:
            payload["system"] = system

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(f"{self.base_url}/api/generate", json=payload)
                if res.status_code == 404:
                    available = await self.list_models()
                    raise OllamaModelNotFoundError(target_model, available)
                if res.status_code != 200:
                    raise OllamaGenerationError(f"HTTP {res.status_code}: {res.text}")

                data = res.json()
                raw_response = data.get("response", "")

                # Strip internal reasoning tags if present in output
                cleaned = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()
                return cleaned

        except httpx.ConnectError:
            raise OllamaUnavailableError(self.base_url)
        except httpx.TimeoutException:
            raise OllamaTimeoutError(target_model, self.timeout)
        except (
            OllamaUnavailableError,
            OllamaModelNotFoundError,
            OllamaTimeoutError,
            OllamaGenerationError,
        ):
            raise
        except Exception as e:
            raise OllamaGenerationError(str(e))
