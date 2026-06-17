"""
LLM Factory — create a LangChain chat model for the configured provider.

Supported providers
-------------------
  "groq"   → ChatGroq   (free tier, no billing needed)
             Get key at: https://console.groq.com/keys
  "google" → ChatGoogleGenerativeAI (requires billing for current models)
             Get key at: https://aistudio.google.com/app/apikey

Usage
-----
    from app.utils.llm_factory import build_llm
    llm = build_llm(provider="groq", api_key="gsk_...", model_name="llama-3.3-70b-versatile")
    response = llm.invoke("Hello")
"""

from __future__ import annotations

from typing import Any


# Default models per provider ──────────────────────────────────────────────────

GROQ_DEFAULTS = {
    "extract": "llama-3.3-70b-versatile",   # fast + capable; free tier
    "verify":  "llama-3.3-70b-versatile",   # same model; consistent reasoning
}

GOOGLE_DEFAULTS = {
    "extract": "gemini-2.0-flash-lite",     # cheapest Google model
    "verify":  "gemini-2.0-flash",          # stronger reasoning
}


def build_llm(
    provider:    str,
    api_key:     str,
    model_name:  str,
    temperature: float = 0.0,
) -> Any:
    """
    Return a LangChain-compatible chat model for the given provider.

    Both returned objects support:  llm.invoke(prompt_string) → AIMessage

    Parameters
    ----------
    provider    : "groq" or "google"
    api_key     : provider API key
    model_name  : model identifier for that provider
    temperature : 0.0 = deterministic (recommended for fact-checking)
    """
    if not api_key:
        raise EnvironmentError(
            f"No API key provided for provider '{provider}'. "
            "Add it to your .env file or the sidebar. "
            "(Expected env var examples: 'GROQ_API_KEY' or 'GOOGLE_API_KEY')"
        )

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model       = model_name,
            groq_api_key= api_key,
            temperature = temperature,
        )

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model          = model_name,
            google_api_key = api_key,
            temperature    = temperature,
            convert_system_message_to_human = True,
        )

    else:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            "Choose 'groq' or 'google'."
        )


def detect_provider(google_api_key: str | None, groq_api_key: str | None) -> str:
    """
    Auto-detect which provider to use based on available keys.
    Groq is preferred (free, no billing).

    Returns "groq", "google", or raises EnvironmentError.
    """
    if groq_api_key and groq_api_key.strip():
        return "groq"
    if google_api_key and google_api_key.strip():
        return "google"
    raise EnvironmentError(
        "No LLM API key found. Add a Groq key (free) or Google Gemini key to the sidebar."
    )
