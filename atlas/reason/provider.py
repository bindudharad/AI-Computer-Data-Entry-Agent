"""LLM providers and the decision advisor.

The advisor answers the agent's live questions (see MISSION): what screen is
visible, which field is next, which source value belongs here, should we
type/click/select/scroll/wait, did the last action succeed, and what recovery
should run. It returns structured JSON that the planner / recovery engine can
act on. When no LLM is configured the advisor returns ``None`` and the
deterministic planner carries on.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from atlas.config import ReasoningConfig
from atlas.core.logging import logger


class LLMProvider(ABC):
    """Text LLM interface for decisioning."""

    name = "abstract"

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Return the model's completion for (system, user)."""

    def close(self) -> None:
        pass


class OpenAILLMProvider(LLMProvider):
    """OpenAI-compatible chat completions (also DeepSeek, Ollama, vLLM...)."""

    name = "openai"

    def __init__(self, config: ReasoningConfig) -> None:
        self._config = config
        self._base = (config.api_base or "https://api.openai.com/v1").rstrip("/")
        self._model = config.model or "gpt-4o-mini"

    def complete(self, system: str, user: str) -> str:
        import requests

        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
        }
        resp = requests.post(
            f"{self._base}/chat/completions", headers=headers, json=payload, timeout=self._config.timeout
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class GeminiLLMProvider(LLMProvider):
    """Google Gemini text generation via REST."""

    name = "gemini"

    def __init__(self, config: ReasoningConfig) -> None:
        self._config = config
        self._model = config.model or "gemini-2.0-flash"

    def complete(self, system: str, user: str) -> str:
        if not self._config.api_key:
            raise RuntimeError("GeminiLLMProvider requires REASONING_API_KEY")
        import requests

        payload = {
            "contents": [{"role": "user", "parts": [{"text": system + "\n\n" + user}]}],
            "generationConfig": {"temperature": 0.0},
        }
        base = (self._config.api_base or "https://generativelanguage.googleapis.com").rstrip("/")
        resp = requests.post(
            f"{base}/v1beta/models/{self._model}:generateContent",
            headers={"x-goog-api-key": self._config.api_key},
            json=payload,
            timeout=self._config.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict))


def _parse_json_response(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


class LLMAdvisor:
    """Consults an LLM for structured decisions.

    ``consult(system, user)`` returns a parsed JSON dict or ``None`` (no LLM
    configured, or the response could not be parsed). All call sites must treat
    ``None`` as "fall back to deterministic logic".
    """

    SYSTEM_PROMPT = """You are the decision engine of ATLAS, an autonomous desktop data-entry agent.
The agent works in an Observe->Understand->Reason->Plan->Execute->Verify loop.
You receive a JSON context describing the current application, screen, source record,
field mapping, and any recent failures. You reply with a SINGLE JSON object.

Reply shape (always the same):
{
  "assessment": "one short sentence describing what is happening on screen",
  "next_action": {
    "type": "click|type|select|scroll|wait|verify|submit|analyze|stop|retry",
    "field_id": "<target field id or null>",
    "value": "<value to type/select or null>",
    "reason": "why"
  },
  "submit": true|false,
  "confidence": 0.0-1.0
}
Rules:
- If a previous action failed, prefer retry with a corrective action first.
- Never invent field ids; only use ids present in the context.
- If the mapping is complete, set submit=true.
- If the screen is a loading screen or popup, advise waiting or clicking the popup button.
Return valid JSON only."""

    def __init__(self, provider: LLMProvider | None, confidence_threshold: float = 0.5) -> None:
        self._provider = provider
        self._threshold = confidence_threshold

    @property
    def available(self) -> bool:
        return self._provider is not None

    def consult(self, context: dict[str, Any]) -> dict | None:
        if self._provider is None:
            return None
        user = json.dumps(context, ensure_ascii=False, default=str)
        try:
            text = self._provider.complete(self.SYSTEM_PROMPT, user)
        except Exception as exc:
            logger.warning("LLM consult failed: {}", exc)
            return None
        data = _parse_json_response(text)
        if data is None:
            logger.warning("LLM returned unparseable decision")
        return data

    def close(self) -> None:
        if self._provider is not None:
            try:
                self._provider.close()
            except Exception:
                pass


def create_llm_provider(config: ReasoningConfig) -> LLMProvider | None:
    """Instantiate the configured LLM provider, or None (auto rule-only)."""
    provider = config.provider.lower()
    if provider in {"openai", "auto"} and (config.api_key or config.api_base):
        return OpenAILLMProvider(config)
    if provider == "gemini" and config.api_key:
        return GeminiLLMProvider(config)
    if provider in {"openai", "gemini"}:
        logger.info("LLM provider '{}' configured without a key - running rule-only", provider)
    return None


__all__ = [
    "LLMProvider",
    "OpenAILLMProvider",
    "GeminiLLMProvider",
    "LLMAdvisor",
    "create_llm_provider",
]
