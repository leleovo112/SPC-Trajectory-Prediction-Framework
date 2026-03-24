# src/llm/llm_client.py
"""
LLMClient: unified interface to multiple LLM backends.

Supported providers:
  openai    – OpenAI GPT-4-Turbo (and snapshots)
  deepseek  – DeepSeek-V3 via DeepSeek API (OpenAI-compatible)
  qwen      – Qwen-2.5-* via DashScope OpenAI-compatible endpoint
  local     – Any locally served OpenAI-compatible model (e.g. vLLM)

Rate limiting is applied to openai calls (20 calls / 60 s) to avoid
429 errors during large-scale inference.
"""
import os
import json
import logging
import re
import time

from config import cfg

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    logger.warning("openai package not found. Install with: pip install openai")

try:
    from ratelimit import limits, sleep_and_retry

    @sleep_and_retry
    @limits(calls=20, period=60)
    def _rate_limited_call(client, **kwargs):
        return client.chat.completions.create(**kwargs)

except ImportError:
    logger.warning("ratelimit package not found; rate limiting disabled.")

    def _rate_limited_call(client, **kwargs):
        return client.chat.completions.create(**kwargs)


# -----------------------------------------------------------------------
# Provider registry
# -----------------------------------------------------------------------
_PROVIDER_CONFIGS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4-turbo",
        "api_key_env": "OPENAI_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen2.5-72b-instruct",
        "api_key_env": "QWEN_API_KEY",
    },
    "local": {
        "base_url": "http://localhost:8000/v1",
        "default_model": "local-model",
        "api_key_env": None,
    },
}


class LLMClient:
    """
    Unified LLM client supporting openai / deepseek / qwen / local backends.
    All non-OpenAI providers use the OpenAI-compatible chat-completions API.
    """

    def __init__(self, provider: str = "openai", model: str = None):
        self.provider = provider.lower()
        self.client   = None

        pconf = _PROVIDER_CONFIGS.get(self.provider, _PROVIDER_CONFIGS["openai"])
        self.base_url = pconf["base_url"]
        self.model    = model or pconf["default_model"]
        api_key_env   = pconf.get("api_key_env")

        # Resolve API key
        if api_key_env:
            self.api_key = os.environ.get(api_key_env, "")
            # Fall back to cfg attribute if env variable is empty
            cfg_attr = api_key_env  # e.g. "OPENAI_API_KEY"
            if not self.api_key and hasattr(cfg, cfg_attr):
                self.api_key = getattr(cfg, cfg_attr, "")
        else:
            self.api_key = "EMPTY"  # local model

        self._init_client()

    def _init_client(self):
        if OpenAI is None:
            logger.error("openai package is required. Run: pip install openai")
            return
        if not self.api_key and self.provider != "local":
            logger.warning(
                f"No API key found for provider '{self.provider}'. "
                "Set the corresponding environment variable."
            )
            return
        try:
            self.client = OpenAI(
                api_key=self.api_key or "EMPTY",
                base_url=self.base_url,
            )
            logger.info(f"LLMClient initialised: provider={self.provider}, model={self.model}")
        except Exception as e:
            logger.warning(f"LLM client initialisation failed: {e}")

    # ------------------------------------------------------------------
    def _call_api(self, prompt: str) -> str | None:
        if self.client is None:
            return None
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise traffic safety reasoning assistant "
                    "for autonomous driving decision-making. "
                    "Always respond with valid JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            resp = _rate_limited_call(
                self.client,
                model=self.model,
                messages=messages,
                temperature=cfg.LLM_TEMPERATURE,
                max_tokens=cfg.LLM_MAX_TOKENS,
                timeout=45,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning(f"LLM API call failed ({self.provider}/{self.model}): {e}")
            return None

    # ------------------------------------------------------------------
    def _parse_output(self, content: str):
        if not content:
            return None, "Empty response"

        # Strip markdown code fences
        clean = re.sub(r'```(?:json)?\s*|```', '', content, flags=re.IGNORECASE).strip()

        try:
            result    = json.loads(clean)
            action_id = int(result.get("action_id", -1))
            reasoning = json.dumps(result, ensure_ascii=False, indent=2)
            return action_id, reasoning
        except Exception:
            pass

        # Regex fallback: extract action_id integer
        m = re.search(r'"action_id"\s*:\s*(\d+)', clean)
        if m:
            return int(m.group(1)), f"Regex-extracted action_id from: {clean[:200]}"

        logger.warning(f"LLM output parse failed. Raw content (first 300 chars): {clean[:300]}")
        return None, "Parse failed"

    # ------------------------------------------------------------------
    def query(self, prompt: str):
        """
        Query the LLM and return (action_id, reasoning_str).
        Returns (None, None) if the call fails or produces an invalid action_id.
        """
        content = self._call_api(prompt)
        if content:
            action_id, reasoning = self._parse_output(content)
            if action_id is not None and 0 <= action_id < cfg.NUM_ACTIONS:
                return action_id, reasoning
        return None, None
