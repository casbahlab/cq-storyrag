import os
import json
import re
from typing import Callable, Dict, Any, Optional

# -------- Ollama (local) --------
try:
    from ollama import chat as ollama_chat_api
except Exception:
    ollama_chat_api = None


def _env_or_default(key: str, default: Any) -> Any:
    val = os.environ.get(key)
    if val is None:
        return default
    if isinstance(default, int):
        try:
            return int(val)
        except ValueError:
            return default
    if isinstance(default, float):
        try:
            return float(val)
        except ValueError:
            return default
    return val


# Defaults for Ollama
OLLAMA_MODEL = _env_or_default("OLLAMA_MODEL", "llama3.1")
OLLAMA_TEMP = _env_or_default("OLLAMA_TEMPERATURE", 0.2)
OLLAMA_MAX_TOKENS = _env_or_default("OLLAMA_MAX_TOKENS", 800)

_CONTENT_PATTERNS = [
    re.compile(r"content='(.*?)'", re.DOTALL),            # Message(..., content='...')
    re.compile(r'content="(.*?)"', re.DOTALL),            # Message(..., content="...")
    re.compile(r'"content"\s*:\s*"(.*?)"', re.DOTALL),    # JSON-like "content": "..."
    re.compile(r"'content'\s*:\s*'(.*?)'", re.DOTALL),    # JSON-like 'content': '...'
]

import ast
import re


def extract_narrative_text(resp: Any) -> str:
    """
    Robustly extract the assistant's text content from various return types:
    - Ollama ChatResponse (Pydantic): resp.message.content
    - Objects with .dict() / .model_dump()
    - dict / list streaming
    - stringified reprs (fallback)
    """
    # 1) Native Ollama ChatResponse object (Pydantic)
    try:
        # Most reliable: attribute access
        if hasattr(resp, "message") and getattr(resp, "message") is not None:
            msg = getattr(resp, "message")
            if hasattr(msg, "content") and msg.content:
                return str(msg.content).strip()
    except Exception:
        pass

    # 2) Pydantic dump to dict
    try:
        if hasattr(resp, "model_dump"):
            d = resp.model_dump()
            return (d.get("message", {}) or {}).get("content", "") or d.get("content", "") or ""
    except Exception:
        pass
    try:
        if hasattr(resp, "dict"):
            d = resp.dict()
            return (d.get("message", {}) or {}).get("content", "") or d.get("content", "") or ""
    except Exception:
        pass

    # 3) Plain dict / list (non-pydantic)
    if isinstance(resp, dict):
        return (resp.get("message", {}) or {}).get("content", "") or resp.get("content", "") or ""
    if isinstance(resp, list):
        return "".join(
            (chunk.get("message", {}) or {}).get("content", "") or chunk.get("content", "") or ""
            for chunk in resp
        ).strip()

    # 4) String reprs — last resort: try to parse then regex
    if isinstance(resp, str):
        try:
            parsed = ast.literal_eval(resp)
            return extract_narrative_text(parsed)
        except Exception:
            pass
        # Try to catch Message(..., content='...') forms
        m = re.search(r"content=['\"](.*?)['\"]\s*,\s*(?:thinking|images|tool_calls)=", resp, re.DOTALL)
        if m:
            return m.group(1).strip()
        return resp.strip()

    # 5) Fallback
    return str(resp).strip()



def ollama_chat(system_prompt, user_prompt, *, model=None, temperature=None, max_tokens=None) -> str:
    if ollama_chat_api is None:
        raise RuntimeError("Ollama client not installed. Run: pip install ollama")

    mname = model or OLLAMA_MODEL
    temp = temperature if temperature is not None else OLLAMA_TEMP
    mtok = max_tokens if max_tokens is not None else OLLAMA_MAX_TOKENS

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]
    resp = ollama_chat_api(
        model=mname,
        messages=messages,
        options={"temperature": temp, "num_predict": mtok},
    )

    return extract_narrative_text(resp)



# -------- Gemini (Google) --------
# pip install google-generativeai
_gemini_loaded = False
try:
    import google.generativeai as genai
    _gemini_loaded = True
except Exception:
    pass


def _ensure_gemini_configured():
    if not _gemini_loaded:
        raise RuntimeError("google-generativeai not installed. Run: pip install google-generativeai")
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY environment variable for Gemini.")
    genai.configure(api_key=api_key)


def gemini_chat(
    system_prompt: str,
    user_prompt: str,
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> str:
    """
    Chat wrapper for Gemini with same signature as ollama_chat.
    """
    _ensure_gemini_configured()

    model_name = model or os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
    temp = temperature if temperature is not None else float(os.environ.get("GEMINI_TEMPERATURE", "0.2"))
    mtok = max_tokens if max_tokens is not None else int(os.environ.get("GEMINI_MAX_TOKENS", "800"))

    # Gemini supports 'system_instruction' on model creation
    gmodel = genai.GenerativeModel(
        model_name,
        system_instruction=system_prompt.strip()
    )

    gen_cfg = genai.types.GenerationConfig(
        temperature=temp,
        max_output_tokens=mtok,
    )

    # Single-turn call; you can upgrade to chat sessions if needed
    resp = gmodel.generate_content(user_prompt.strip(), generation_config=gen_cfg)
    # Newer SDK: resp.text; fallback to candidates
    text = getattr(resp, "text", None)
    if text:
        return text.strip()

    # Fallback extraction
    try:
        return resp.candidates[0].content.parts[0].text.strip()
    except Exception:
        return ""


# -------- Registry --------

def get_llm_client(name: str) -> Callable[..., str]:
    """
    Return a function(system_prompt, user_prompt, *, model, temperature, max_tokens) -> str
    """
    n = (name or "").lower()
    if n in ("ollama", "local", "llama"):
        return ollama_chat
    if n in ("gemini", "google"):
        return gemini_chat
    raise ValueError(f"Unknown LLM provider: {name!r}")


def call_llm_by_config(
    provider: str,
    system_prompt: str,
    user_prompt: str,
    *,
    cfg: Optional[Dict[str, Any]] = None
) -> str:
    """
    Convenience helper: selects provider and pulls defaults from config dict.
    cfg example:
      {
        "llm": {"model": "llama3.1", "temperature": 0.2, "max_tokens": 600},
        "gemini": {"model": "gemini-1.5-pro", "temperature": 0.2, "max_tokens": 600}
      }
    """
    fn = get_llm_client(provider)
    # pick subset based on provider
    if provider.lower() == "gemini":
        section = (cfg or {}).get("gemini", {})
    else:
        section = (cfg or {}).get("llm", {})
    return fn(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=section.get("model"),
        temperature=section.get("temperature"),
        max_tokens=section.get("max_tokens"),
    )
