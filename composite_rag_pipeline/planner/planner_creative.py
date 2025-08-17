#!/usr/bin/env python3
"""
planner_creative.py — LLM-driven narrative planner with CQ lookup (KG/Hybrid)

Flow
----
1) Generate a creative plan with an LLM (beats: title, intent, must_cover, items).
   - Or use --dry_run for a deterministic, dependency-free plan.
2) Embed each beat’s intent (deterministic pseudo-vectors; no external deps).
3) Search your CQ index (FAISS if present, else CPU cosine over embeddings.npy).
4) Select a diverse set across beats (light novelty penalty).
5) Emit the plan + selected CQ items as JSON to stdout.

Features
--------
- **On-demand Ollama**: `--auto_ollama` spins up a local server on a free port.
  Use `--ollama_pull` to pull the requested model before use.
- **Providers**: `--llm_provider {ollama|openai|none}`; set `OPENAI_API_KEY` for OpenAI.
- **Backends**: FAISS (`faiss.index`) or CPU (`embeddings.npy`).
- **Schemas**: Supports new {"order","metadata"} or legacy shapes.

Examples
--------
# Dry run (no LLM calls), KG
python3 planner_creative.py \
  --index_dir ./KG \
  --meta ./KG/cq_metadata.json \
  --persona WembleyRewind --length Medium --mode KG \
  --items_per_beat 2 --num_beats 5 --dry_run > plan_KG.json

# Creative plan via Ollama (auto-start + pull model)
python3 planner_creative.py \
  --index_dir ./Hybrid \
  --meta ./Hybrid/cq_metadata.json \
  --persona WembleyRewind --length Medium --mode Hybrid \
  --items_per_beat 2 --num_beats 5 \
  --llm_provider ollama --llm_model llama3.1 \
  --auto_ollama --ollama_pull > plan_Hybrid.json
"""
from __future__ import annotations

import argparse
import json
import os
import atexit
import subprocess
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional FAISS
try:
    import faiss  # pip install faiss-cpu
    _HAVE_FAISS = True
except Exception:
    _HAVE_FAISS = False


# ------------------------ Ollama helpers (on demand) ------------------------

def _ollama_up(host: str, timeout: float = 1.0) -> bool:
    try:
        import requests
    except Exception:
        return False
    try:
        r = requests.get(host.rstrip("/") + "/", timeout=timeout)
        if r.ok and "Ollama is running" in (r.text or ""):
            return True
    except Exception:
        pass
    try:
        r = requests.get(host.rstrip("/") + "/api/tags", timeout=timeout)
        return r.ok
    except Exception:
        return False


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ensure_ollama(host: str, auto: bool, pull_model: bool, model: str) -> str:
    """
    Ensure an Ollama server is reachable. If not and auto=True, start a private
    server on a free port and optionally `ollama pull <model>`.
    Returns a base URL to use.
    """
    if _ollama_up(host):
        if pull_model:
            subprocess.run(["ollama", "pull", model], check=False)
        return host

    if not auto:
        return host  # user handles availability

    port = _find_free_port()
    auto_host = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    proc = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    def _kill():
        try:
            proc.terminate()
        except Exception:
            pass

    atexit.register(_kill)

    start = time.time()
    while time.time() - start < 10.0:
        if _ollama_up(auto_host, timeout=0.5):
            if pull_model:
                subprocess.run(["ollama", "pull", model], check=False, env=env)
            return auto_host
        time.sleep(0.25)

    # Fallback to original host (might be a system service)
    return host


# ------------------------ Metadata / Index loading ------------------------

def _load_meta(meta_path: Path):
    """
    Return aligned arrays for ids, beats, texts, questions, answers, sparqls
    from either the new {"order","metadata"} schema or legacy shapes.
    """
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    def _from_rows(order_key: str, rows_key: str):
        order = meta[order_key]
        rows = meta[rows_key]
        ids = order
        beats = [rows[c].get("beats") or rows[c].get("beat") or "" for c in order]
        texts = [rows[c].get("question", "") for c in order]
        questions = [rows[c].get("question", "") for c in order]
        answers = [rows[c].get("answer", "") for c in order]
        sparqls = [rows[c].get("sparql", "") for c in order]
        return ids, beats, texts, questions, answers, sparqls

    if "metadata" in meta and "order" in meta:
        return _from_rows("order", "metadata")
    if "rows" in meta and "order" in meta:
        return _from_rows("order", "rows")

    # Legacy arrays
    ids = meta["ids"]
    beats = meta.get("beats", [""] * len(ids))
    texts = meta.get("texts", [""] * len(ids))
    questions = meta.get("questions", [""] * len(ids))
    answers = meta.get("answers", [""] * len(ids))
    sparqls = meta.get("sparqls", [""] * len(ids))
    return ids, beats, texts, questions, answers, sparqls


def _load_backend(index_dir: Path, embeddings_npy: Optional[Path] = None):
    """
    Returns ("faiss", faiss_index) or ("cpu", embeddings_matrix).
    """
    index_path = index_dir / "faiss.index"
    if _HAVE_FAISS and index_path.exists():
        index = faiss.read_index(str(index_path))
        return ("faiss", index)
    else:
        if embeddings_npy is None:
            embeddings_npy = index_dir / "embeddings.npy"
        E = np.load(str(embeddings_npy)).astype("float32", copy=False)
        return ("cpu", E)


def _backend_dim(backend_tuple) -> int:
    kind, obj = backend_tuple
    if kind == "faiss":
        # Most FAISS indexes expose d; otherwise use an aux vector
        try:
            return obj.d
        except Exception:
            # Fallback: try reading from stored vectors is not available here
            # Default to 384 (SBERT mini) as a safe guess
            return 384
    else:
        E = obj  # numpy array
        return int(E.shape[1])


# ------------------------ Search backends ------------------------

def _faiss_search(index, Q: np.ndarray, topk: int, nprobe: Optional[int] = None):
    if nprobe is not None and hasattr(index, "nprobe"):
        index.nprobe = int(nprobe)
    return index.search(Q, topk)


def _cpu_search(E: np.ndarray, Q: np.ndarray, topk: int):
    # Cosine similarity (embeddings are expected normalized from index build)
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    S = Qn @ E.T
    k = min(topk, E.shape[0])
    idxs = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
    row_sorted = np.take_along_axis(S, idxs, axis=1).argsort(axis=1)[:, ::-1]
    idxs = np.take_along_axis(idxs, row_sorted, axis=1)
    scores = np.take_along_axis(S, idxs, axis=1)
    return scores, idxs


# ------------------------ Beat embeddings (deterministic) ------------------------

def _embed_beats(beats: List[Dict[str, Any]], dim: int) -> np.ndarray:
    """
    Produce deterministic pseudo-embeddings per beat intent/must_cover.
    This avoids needing a second embedder and keeps selection reproducible.
    """
    Q = np.zeros((len(beats), dim), dtype="float32")
    for i, b in enumerate(beats):
        intent = b.get("intent", "")
        must = b.get("must_cover", [])
        seed = abs(hash(intent + "|" + ",".join(must))) % (10**9)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(dim).astype("float32")
        v /= (np.linalg.norm(v) + 1e-12)
        Q[i] = v
    return Q


# ------------------------ LLM plumbing ------------------------

PLAN_PROMPT = """You are a creative story planner for a factual football music show called "Wembley Rewind".
Given a persona and a target length, produce a compact JSON plan with beats.
Each beat should have: title, intent (one sentence), must_cover (list of terms/entities),
and items (how many CQs to pick for that beat).
Return ONLY valid JSON.

Persona: {persona}
Length: {length}
Mode: {mode} (KG or Hybrid) — same retrieval infra; mode only changes which CQ set is eligible.
Constraints:
- Keep {num_beats} beats total.
- Aim for variety; avoid repeating the same entity across beats.
- Use concise titles.
JSON schema:
{{
  "persona": "...",
  "length": "...",
  "mode": "...",
  "beats": [
    {{
      "title": "...",
      "intent": "...",
      "must_cover": ["...", "..."],
      "items": {items_per_beat}
    }}
  ]
}}
"""

def _call_llm(provider: str, model: str, prompt: str, temperature: float = 0.8, base_url: Optional[str] = None) -> str:
    if provider == "ollama":
        import requests
        url = (base_url or "http://localhost:11434").rstrip("/") + "/api/generate"

        # ask Ollama to return strict JSON
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False,
            "format": "json",   # <— this line does it
        }

        # first try with JSON format
        r = requests.post(url, json=payload, timeout=180)
        if not r.ok:
            # some models don’t support 'format': 'json' — retry once without it
            payload.pop("format", None)
            r = requests.post(url, json=payload, timeout=180)
            r.raise_for_status()

        data = r.json()
        return data.get("response", "")

    elif provider == "openai":
        import requests, os
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        url = (base_url or "https://api.openai.com/v1/chat/completions").rstrip("/")
        headers = {"Authorization": f"Bearer {key}"}
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful planner."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"}
        }
        r = requests.post(url, headers=headers, json=body, timeout=180)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    else:
        raise RuntimeError(f"Unknown provider '{provider}'.")


def _dry_run_plan(persona: str, length: str, mode: str, num_beats: int, items_per_beat: int) -> Dict[str, Any]:
    seeds = ["Opening Hook", "History Hit", "Rivalry Pulse", "Iconic Moment", "Finale"]
    beats = []
    for i in range(num_beats):
        title = seeds[i % len(seeds)]
        beats.append({
            "title": title,
            "intent": f"Explore a {title.lower()} relevant to {persona}.",
            "must_cover": (["Queen", "1985"] if i % 2 == 0 else ["Wembley", "Live Aid"]),
            "items": items_per_beat
        })
    return {"persona": persona, "length": length, "mode": mode, "beats": beats}


# ------------------------ Selection (diversity-aware) ------------------------

def _select_per_plan(scores: np.ndarray, idxs: np.ndarray, questions: List[str], k: int, novelty_lambda: float = 0.12) -> List[int]:
    """
    Round-robin across beats, with a light penalty on overlapping tokens to promote variety.
    """
    chosen: List[int] = []
    seen_terms = set()

    def tokens(s: str):
        return {t.lower() for t in s.split() if len(t) > 2}

    B, K = idxs.shape
    ptr = np.zeros(B, dtype=np.int32)

    while len(chosen) < k:
        progressed = False
        for b in range(B):
            if len(chosen) >= k:
                break
            p = int(ptr[b])
            if p >= K:
                continue
            pos = int(idxs[b, p])
            ptr[b] += 1
            base = float(scores[b, p])
            ov = len(tokens(questions[pos]) & seen_terms)
            score = base - novelty_lambda * ov  # currently not used for ordering within the step
            if pos not in chosen:
                chosen.append(pos)
                seen_terms |= tokens(questions[pos])
                progressed = True
        if not progressed:
            break

    return chosen[:k]

import re

def _extract_json_object(raw: str) -> dict:
    """Extract a JSON object from LLM text that may include prose or ``` fences."""
    s = (raw or "").strip()
    if not s:
        raise ValueError("Empty LLM response")

    # Try fenced code block first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.S | re.I)
    if m:
        txt = m.group(1)
    else:
        # Fall back to first {...} span
        i, j = s.find("{"), s.rfind("}")
        if i == -1 or j == -1 or j <= i:
            raise ValueError("No JSON object found in LLM output")
        txt = s[i:j+1]

    # Normalize smart quotes just in case
    txt = (txt
           .replace("\u201c", '"')
           .replace("\u201d", '"')
           .replace("\u2018", "'")
           .replace("\u2019", "'"))
    return json.loads(txt)


# ------------------------ Main ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--persona", required=True)
    ap.add_argument("--length", required=True, choices=["Short", "Medium", "Long"])
    ap.add_argument("--mode", required=True, choices=["KG", "Hybrid"])

    ap.add_argument("--items_per_beat", type=int, default=2)
    ap.add_argument("--num_beats", type=int, default=5)

    ap.add_argument("--llm_provider", default="ollama", choices=["ollama", "openai", "none"])
    ap.add_argument("--llm_model", default="llama3.1")
    ap.add_argument("--llm_base_url", default=None)
    ap.add_argument("--auto_ollama", action="store_true", help="Start Ollama on demand if not already running")
    ap.add_argument("--ollama_pull", action="store_true", help="Pull the specified Ollama model before use")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--dry_run", action="store_true")

    ap.add_argument("--nprobe", type=int, default=None)
    ap.add_argument("--embeddings_npy", default=None)

    ap.add_argument("--per_beat_topk", type=int, default=None, help="Candidate pool per beat (default: max(8, items_per_beat*6))")
    ap.add_argument("--novelty_lambda", type=float, default=0.12, help="Penalty for token overlap across selected questions")

    args = ap.parse_args()

    # Load metadata and backend
    ids, beats_arr, texts, questions, answers, sparqls = _load_meta(Path(args.meta))
    backend = _load_backend(Path(args.index_dir), Path(args.embeddings_npy) if args.embeddings_npy else None)
    dim = _backend_dim(backend)

    # Plan creation
    if args.dry_run or args.llm_provider == "none":
        plan = _dry_run_plan(args.persona, args.length, args.mode, args.num_beats, args.items_per_beat)
    else:
        # Ensure Ollama if selected
        if args.llm_provider == "ollama":
            host = args.llm_base_url or "http://localhost:11434"
            host = _ensure_ollama(host, args.auto_ollama, args.ollama_pull, args.llm_model)
            args.llm_base_url = host

        prompt = PLAN_PROMPT.format(
            persona=args.persona,
            length=args.length,
            mode=args.mode,
            num_beats=args.num_beats,
            items_per_beat=args.items_per_beat
        )

        # payload = {
        #     "model": model,
        #     "prompt": prompt,
        #     "temperature": temperature,
        #     "stream": False,
        #     "format": "json",  # ask for JSON directly
        # }

        raw = _call_llm(args.llm_provider, args.llm_model, prompt, args.temperature, base_url=args.llm_base_url)
        try:
            plan = json.loads(raw)
        except Exception:
            plan = _extract_json_object(raw)

    # Embed beats (deterministic vectors sized to index dimension)
    Q = _embed_beats(plan["beats"], dim=dim)

    # Determine candidate pool per beat
    total_ids = len(ids)
    default_topk = max(8, args.items_per_beat * 6)
    topk = min(args.per_beat_topk or default_topk, total_ids)

    # Search
    if backend[0] == "faiss":
        scores, idxs = _faiss_search(backend[1], Q, topk, nprobe=args.nprobe)
    else:
        scores, idxs = _cpu_search(backend[1], Q, topk)

    # Select final set
    per_plan_k = min(args.items_per_beat * len(plan["beats"]), total_ids)
    chosen_positions = _select_per_plan(scores, idxs, questions, per_plan_k, novelty_lambda=args.novelty_lambda)

    items = [{
        "id": ids[pos],
        "question": questions[pos],
        "answer": answers[pos],
        "sparql": sparqls[pos]
    } for pos in chosen_positions]

    out = {
        "persona": plan["persona"],
        "length": plan["length"],
        "mode": plan["mode"],
        "beats": plan["beats"],
        "items": items
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
