import json
import requests
from typing import Optional

from .config import (
    GEMINI_API_KEY, GEMINI_MODEL,
    GROQ_API_KEY, GROQ_MODEL,
    OLLAMA_HOST, OLLAMA_MODEL
)

# ------------------------
# Gemini (Google AI Studio)
# ------------------------
def call_gemini(prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
    if not GEMINI_API_KEY:
        return "[ERROR] GEMINI_API_KEY not set."

    # Use v1beta with 1.5 models and contents/parts/text payload
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
    }
    try:
        r = requests.post(url, headers=headers, params=params, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        # Extract text
        cand = (data.get("candidates") or [{}])[0]
        parts = ((cand.get("content") or {}).get("parts") or [])
        texts = []
        for p in parts:
            if isinstance(p, dict) and "text" in p:
                texts.append(p["text"])
            elif isinstance(p, str):
                texts.append(p)
        return "".join(texts).strip() or json.dumps(data)[:2000]
    except requests.HTTPError as he:
        return f"[ERROR] Gemini HTTP {getattr(he.response,'status_code', '?')}: {getattr(he.response, 'text','')}"
    except Exception as e:
        return f"[ERROR] Gemini failure: {str(e)}"


# -------------
# Groq (OpenAI)
# -------------
def call_groq(prompt: str, temperature: float = 0.3) -> str:
    if not GROQ_API_KEY:
        return "[ERROR] GROQ_API_KEY not set."

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.HTTPError as he:
        return f"[ERROR] Groq HTTP {getattr(he.response,'status_code', '?')}: {getattr(he.response, 'text','')}"
    except Exception as e:
        return f"[ERROR] Groq failure: {str(e)}"


# --------------------
# Ollama (local model)
# --------------------
def call_ollama(prompt: str) -> str:
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip() or json.dumps(data)[:2000]
    except requests.HTTPError as he:
        return f"[ERROR] Ollama HTTP {getattr(he.response,'status_code', '?')}: {getattr(he.response, 'text','')}"
    except Exception as e:
        return f"[ERROR] Ollama failure: {str(e)}"


# --------------------
# Router
# --------------------
def call_llm(llm: str, prompt: str, **kwargs) -> str:
    llm = (llm or "").lower().strip()
    if llm == "groq":
        return call_groq(prompt, temperature=kwargs.get("temperature", 0.3))
    if llm == "ollama":
        return call_ollama(prompt)
    # default to gemini
    return call_gemini(prompt, max_tokens=kwargs.get("max_tokens", 1024), temperature=kwargs.get("temperature", 0.3))
