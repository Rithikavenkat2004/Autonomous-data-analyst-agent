"""
core/llm_client.py
------------------
Thin wrapper around OpenRouter's free tier so the rest of the app never
talks to the API directly. OpenRouter is genuinely free -- no credit
card required to sign up or to use models ending in ":free". Free tier
gives 50 requests/day (plenty for demoing this project), 20 req/minute.

Get a free API key at: https://openrouter.ai (sign up with email/Google,
no card needed) -> click your profile -> Keys -> Create Key.

OpenRouter exposes an OpenAI-compatible API, so we use the `openai`
Python package pointed at OpenRouter's base URL instead of OpenAI's.

Default model is "openrouter/free" -- OpenRouter's own auto-router,
which picks whichever free model/provider is actually available right
now instead of pinning to one provider that might be temporarily
overloaded (a known issue with specific free models like Venice-hosted
Llama/Qwen during peak hours). We also retry on 429s with the
provider-supplied backoff time, since free-tier rate limits are
expected, not exceptional.
"""

import os
import time
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openrouter/free")


class LLMClient:
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Copy .env.example to .env and add your key "
                "from https://openrouter.ai (Keys page, no card required)."
            )
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024,
                 _retries_left: int = 3) -> str:
        """Single-turn completion. Returns plain text. Retries on 429 (rate limit)."""
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except RateLimitError as e:
            if _retries_left <= 0:
                raise
            # Free-tier providers occasionally throttle; back off and retry
            # rather than surfacing a scary traceback for an expected condition.
            wait_seconds = 15
            try:
                wait_seconds = int(e.response.json()["error"]["metadata"]["retry_after_seconds"]) + 2
            except Exception:
                pass
            time.sleep(min(wait_seconds, 60))
            return self.generate(system_prompt, user_prompt, max_tokens, _retries_left - 1)

    def generate_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        """Same as generate() but instructs the model to return raw JSON only."""
        strict_system = (
            system_prompt
            + "\n\nCRITICAL: Respond with ONLY valid JSON. No markdown fences, "
            "no preamble, no explanation text before or after the JSON."
        )
        raw = self.generate(strict_system, user_prompt, max_tokens)
        return raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()