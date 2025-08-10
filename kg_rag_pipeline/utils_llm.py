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

def _extract_gemini_text(resp) -> str:
    """
    Safely extract concatenated text from a Gemini response.
    Works even if response.text is unavailable.
    """
    if resp is None:
        return ""
    text_chunks = []

    # Newer client: resp.candidates -> candidate.content.parts[*].text
    candidates = getattr(resp, "candidates", None) or []
    for cand in candidates:
        # finish_reason for debug
        fr = getattr(cand, "finish_reason", None)
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if parts:
            for p in parts:
                t = getattr(p, "text", None)
                if t:
                    text_chunks.append(t)

    if text_chunks:
        return "\n".join(text_chunks).strip()

    # Fallback: resp.prompt_feedback (block reason)
    pf = getattr(resp, "prompt_feedback", None)
    block_reason = getattr(pf, "block_reason", None) if pf else None

    # As a last resort, try repr
    return f""  # keep empty so caller can decide to retry/log with finish_reason/block_reason


from typing import Optional
import os

def gemini_chat(
    system_prompt: str,
    user_prompt: str,
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> str:
    try:
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
    except Exception as e:
        raise RuntimeError("google-generativeai not installed. Run: pip install google-generativeai") from e

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in environment")

    genai.configure(api_key=api_key)

    mname = model or os.environ.get("GEMINI_MODEL", "models/gemini-1.5-pro")
    temp = 0.7 if temperature is None else float(temperature)
    mtok = 4000 if max_tokens is None else int(max_tokens)

    # Prefer system_instruction to stuffing system text into user content
    model_obj = genai.GenerativeModel(
        mname,
        system_instruction=system_prompt.strip()
    )

    generation_config = {
        "temperature": temp,
        "max_output_tokens": mtok,
    }

    # Relax safety just enough to avoid over-blocking factual museum content
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
    }

    resp = model_obj.generate_content(
        user_prompt.strip(),
        generation_config=generation_config,
        safety_settings=safety_settings,
    )

    # Robust extraction (don’t use resp.text)
    text = _extract_gemini_text(resp)

    if text:
        return text

    # If empty, log some useful debug info and try one gentle retry
    candidates = getattr(resp, "candidates", None) or []
    finish_reasons = [getattr(c, "finish_reason", None) for c in candidates]
    block_reason = getattr(getattr(resp, "prompt_feedback", None), "block_reason", None)

    print(f"[WARN] Gemini returned no text. finish_reasons={finish_reasons}, block_reason={block_reason}")

    # Retry once with lower temp & more tokens (covers MAX_TOKENS early stops)
    if mtok < 2048:
        try:
            resp2 = model_obj.generate_content(
                user_prompt.strip(),
                generation_config={"temperature": 0.4, "max_output_tokens": max(2048, mtok*2)},
                safety_settings=safety_settings,
            )
            text2 = _extract_gemini_text(resp2)
            if text2:
                return text2
            fr2 = [getattr(c, "finish_reason", None) for c in (getattr(resp2, "candidates", None) or [])]
            br2 = getattr(getattr(resp2, "prompt_feedback", None), "block_reason", None)
            print(f"[WARN] Retry also empty. finish_reasons={fr2}, block_reason={br2}")
        except Exception as e:
            print(f"[WARN] Retry failed: {e}")

    # Final fallback so pipeline doesn’t crash; return empty string
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
