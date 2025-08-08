# utils_llm.py
import requests
OLLAMA_HOST = "http://127.0.0.1:11434"
MODEL = "llama3.1-128k"  # change to your local model tag

def ollama_chat(prompt: str) -> str:
    r = requests.post(f"{OLLAMA_HOST}/api/generate", json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }, timeout=600)
    r.raise_for_status()
    j = r.json()
    return j.get("response","").strip()
