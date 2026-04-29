"""OpenRouter chat-completions adapter."""

import time

import requests

from shared.config import (
    MAX_TOKENS,
    OPENROUTER_URL,
    REQUEST_TIMEOUT,
    SYSTEM_INSTRUCTION,
    TEMPERATURE,
)


class OpenRouterError(Exception):
    """Base OpenRouter transport error."""


class OpenRouterAuthError(OpenRouterError):
    """Raised on 401/403 so long grids fail fast on bad credentials."""


def call_openrouter(
    model_id: str,
    user_prompt: str,
    api_key: str,
    session: requests.Session,
    max_tokens: int = MAX_TOKENS,
    system: str = SYSTEM_INSTRUCTION,
) -> str:
    """Call OpenRouter and return the assistant message content."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
    }

    last_err: Exception | None = None
    for _attempt in range(2):
        try:
            resp = session.post(
                OPENROUTER_URL, headers=headers, json=body, timeout=REQUEST_TIMEOUT
            )
        except requests.RequestException as exc:
            last_err = exc
            time.sleep(2)
            continue

        if resp.status_code in (401, 403):
            raise OpenRouterAuthError(
                f"HTTP {resp.status_code} from OpenRouter "
                f"(check OPENROUTER_API_KEY): {resp.text[:300]}"
            )
        if resp.status_code in (429, 500, 502, 503, 504):
            last_err = OpenRouterError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            time.sleep(2)
            continue
        if resp.status_code != 200:
            raise OpenRouterError(
                f"HTTP {resp.status_code} from OpenRouter: {resp.text[:500]}"
            )
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenRouterError(f"Malformed OpenRouter response: {data}") from exc

    raise OpenRouterError(f"OpenRouter call failed after retries: {last_err}")
