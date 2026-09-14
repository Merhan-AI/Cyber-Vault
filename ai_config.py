"""
ai_config.py
------------
Backend AI provider configuration for SIH26105.

Reads API keys and provider selection from the .env file.
To switch providers or update your key, ONLY edit .env — no code changes needed.

Supported providers:
  - gemini     → Google Gemini (default: gemini-2.0-flash)
  - openai     → OpenAI (default: gpt-4o-mini)
  - groq       → Groq (default: llama3-8b-8192)
  - anthropic  → Anthropic Claude (default: claude-3-haiku-20240307)
"""

import os
from pathlib import Path

# Load .env file from the project root (works locally and on servers)
try:
    from dotenv import load_dotenv
    # Look for .env in the same directory as this file
    _env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass  # python-dotenv not installed; rely on environment variables already set


# ---------------------------------------------------------------------------
# Read configuration from environment
# ---------------------------------------------------------------------------
def get_provider() -> str:
    """Return the configured AI provider name (lowercase)."""
    return os.getenv("AI_PROVIDER", "gemini").strip().lower()


def get_model() -> str | None:
    """Return the configured model name, or None to use the provider default."""
    val = os.getenv("AI_MODEL", "").strip()
    return val if val else None


def get_api_key(provider: str = None) -> str | None:
    """Return the API key for the given provider (or current provider if not specified)."""
    if provider is None:
        provider = get_provider()

    key_map = {
        "gemini":    "GEMINI_API_KEY",
        "openai":    "OPENAI_API_KEY",
        "groq":      "GROQ_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    env_var = key_map.get(provider)
    if not env_var:
        return None
    val = os.getenv(env_var, "").strip()
    return val if val else None


def is_ai_configured() -> bool:
    """Return True if a valid provider + non-empty API key is configured."""
    return bool(get_api_key())


# Default models per provider
_DEFAULT_MODELS = {
    "gemini":    "gemini-2.0-flash",
    "openai":    "gpt-4o-mini",
    "groq":      "llama3-8b-8192",
    "anthropic": "claude-3-haiku-20240307",
}


def get_effective_model() -> str:
    """Return the model to use: env override if set, else provider default."""
    return get_model() or _DEFAULT_MODELS.get(get_provider(), "gemini-2.0-flash")


# ---------------------------------------------------------------------------
# Unified ask_ai() — one function to call regardless of provider
# ---------------------------------------------------------------------------
def ask_ai(question: str, context: str) -> str:
    """
    Send a question + data context to the configured AI provider.

    Returns the AI response as a string.
    Returns an error string (starting with ⚠️) if the call fails or is unconfigured.
    """
    provider = get_provider()
    api_key = get_api_key(provider)
    model = get_effective_model()

    if not api_key:
        return (
            "⚠️ No API key configured. "
            f"Set {provider.upper()}_API_KEY in your .env file to enable AI answers."
        )

    prompt = f"{context}\n\n=== USER QUESTION ===\n{question}"

    try:
        if provider == "gemini":
            return _call_gemini(prompt, api_key, model)
        elif provider == "openai":
            return _call_openai(prompt, api_key, model)
        elif provider == "groq":
            return _call_groq(prompt, api_key, model)
        elif provider == "anthropic":
            return _call_anthropic(prompt, api_key, model)
        else:
            return f"⚠️ Unknown AI provider '{provider}'. Check AI_PROVIDER in your .env file."
    except Exception as e:
        return f"⚠️ AI error ({provider}/{model}): {e}"


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------
def _call_gemini(prompt: str, api_key: str, model: str) -> str:
    from google import genai
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text


def _call_openai(prompt: str, api_key: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def _call_groq(prompt: str, api_key: str, model: str) -> str:
    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def _call_anthropic(prompt: str, api_key: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ---------------------------------------------------------------------------
# Status summary (used by app.py to show config state in UI)
# ---------------------------------------------------------------------------
def get_status() -> dict:
    """Return a dict describing the current AI configuration state."""
    provider = get_provider()
    api_key = get_api_key(provider)
    model = get_effective_model()
    return {
        "provider": provider,
        "model": model,
        "configured": bool(api_key),
        "key_preview": f"{api_key[:6]}...{api_key[-4:]}" if api_key and len(api_key) > 10 else None,
    }
