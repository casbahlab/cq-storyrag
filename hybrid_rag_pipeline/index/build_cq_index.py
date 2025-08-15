#!/usr/bin/env python3
"""
Build a FAISS index from your CQ catalog using Ollama embeddings,
and write metadata keyed by CQ_ID (per-CQ mapping).

Inputs
------
- CQ JSON (array or keyed dict). Fields (best-effort):
  * CQ_ID | CQ-ID | id | ID
  * Question | question | CQ
  * Answer | answer | ExpectedAnswer | AnswerTemplate | TemplateAnswer
  * Beats | beat
  * SPARQL (string or file): see _DIRECT_SP_KEYS / _FILE_SP_KEYS below
- .rq template with sections:
  #CQ-ID:<ID> <Question text...>
  ... SPARQL ...
  (until next #CQ-ID: or EOF)

Outputs
-------
- cq_index.faiss
- cq_metadata.json
  {
    "model": "...",
    "dim": D,
    "count": N,
    "order": ["CQ-E1", "CQ-L22", ...],       # FAISS row -> CQ_ID
    "rows": {
      "CQ-E1": {
        "i": 0,
        "beat": "Performance Detail",
        "text": "What kind of event was ...\\n[Beat] Performance Detail",
        "sparql": "SELECT ...",
        "question": "What kind of event was [Event]?",
        "answer": "Benefit concert"         # if present in CSV/JSON
      },
      ...
    },
    # legacy arrays (still written for compatibility):
    "ids": [...], "beats": [...], "texts": [...],
    "sparqls": [...], "questions": [...], "answers": [...]
  }
"""

import argparse, json, re
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

try:
    import faiss  # pip install faiss-cpu
except Exception as e:
    raise RuntimeError("faiss is required. pip install faiss-cpu") from e

try:
    from ollama import embed as ollama_embed  # pip install ollama
except Exception as e:
    raise RuntimeError("ollama client is required. pip install ollama") from e

# ---------- helpers ----------

def load_cq_rows(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):  return list(data.values())
    if isinstance(data, list):  return data
    raise ValueError("CQ JSON must be an array or a keyed dict.")

def first_nonempty(row: Dict[str, Any], *names: str) -> str:
    for n in names:
        v = row.get(n)
        if isinstance(v, str) and v.strip(): return v.strip()
    return ""

_DIRECT_SP_KEYS = [
    "SPARQL","sparql","SPARQLTemplate","sparql_template",
    "SPARQL Query","SPARQLQuery","sparqlQuery","sparqlquery",
]
_FILE_SP_KEYS = [
    "SPARQL_File","sparql_file","SPARQLFile","sparqlFile",
    "SPARQL_Path","sparql_path","SPARQLTemplateFile","sparql_template_file",
]
_ANSWER_KEYS = ["Answer","answer","ExpectedAnswer","AnswerTemplate","TemplateAnswer"]

def read_sparql_from_row(row: Dict[str, Any], default_root: Path, cli_root: Path) -> str:
    s = first_nonempty(row, *_DIRECT_SP_KEYS)
    if s: return s
    fp = first_nonempty(row, *_FILE_SP_KEYS)
    if not fp: return ""
    p = Path(fp)
    if not p.is_absolute():
        root = cli_root if str(cli_root) != "." else default_root
        p = (root / fp).resolve()
    try:
        return p.read_text(encoding="utf-8").strip()
    except Exception:
        return ""

def parse_rq_sections(rq_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Parse .rq where sections start '#CQ-ID:<ID> <Question...>'
    Returns { id: {question, sparql} }
    """
    txt = rq_path.read_text(encoding="utf-8")
    lines = txt.splitlines()
    results: Dict[str, Dict[str, str]] = {}
    header_re = re.compile(r'^\s*#CQ-ID\s*:\s*([^\s]+)\s*(.*)$', re.IGNORECASE)

    cur_id, cur_q, buf = "", "", []
    def flush():
        nonlocal cur_id, cur_q, buf
        if cur_id:
            results[cur_id] = {"question": cur_q.strip(), "sparql": "\n".join(buf).strip()}
        cur_id, cur_q, buf = "", "", []

    for line in lines:
        m = header_re.match(line)
        if m:
            flush()
            cur_id = m.group(1).strip()
            cur_q  = (m.group(2) or "").strip()
        else:
            buf.append(line)
    flush()
    return results

def embed_texts_ollama(texts: List[str], model: str, batch: int = 64) -> np.ndarray:
    vecs: List[List[float]] = []
    for i in range(0, len(texts), batch):
        batch_texts = texts[i:i+batch]
        resp = ollama_embed(model=model, input=batch_texts)
        embs = resp.get("embeddings") or []
        if not embs: raise RuntimeError("Ollama returned no embeddings.")
        vecs.extend(embs)
    X = np.array(vecs, dtype="float32")
    norms = np.linalg.norm(X, axis=1, keepdims=True); norms[norms==0.0]=1.0
    return X / norms  # cosine via inner product

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cq_json", required=True)
    ap.add_argument("--rq_file", required=True, help="SPARQL templates with '#CQ-ID:' headers")
    ap.add_argument("--out_index", default="cq_index.faiss")
    ap.add_argument("--out_meta",  default="cq_metadata.json")
    ap.add_argument("--model",     default="nomic-embed-text")
    ap.add_argument("--batch",     type=int, default=64)
    ap.add_argument("--sparql_root", default=".", help="Root for SPARQL file paths in CSV/JSON")
    args = ap.parse_args()

    cq_path = Path(args.cq_json).resolve()
    rq_path = Path(args.rq_file).resolve()
    rows_in = load_cq_rows(cq_path)
    rq_map  = parse_rq_sections(rq_path)

    default_root = cq_path.parent
    cli_root     = Path(args.sparql_root).resolve()

    # Build per-CQ records
    order: List[str] = []
    rows_map: Dict[str, Dict[str, Any]] = {}

    direct_sp = file_sp = from_rq = q_from_rq = 0

    for r in rows_in:
        cid = first_nonempty(r, "CQ_ID","CQ-ID","id","ID")
        if not cid:
            continue
        q_text  = first_nonempty(r, "Question","question","CQ")
        answer  = first_nonempty(r, *_ANSWER_KEYS)
        beat    = first_nonempty(r, "Beats","beat")

        sparql  = read_sparql_from_row(r, default_root, cli_root)
        if sparql:
            if any(k in r and isinstance(r[k], str) and r[k].strip() for k in _FILE_SP_KEYS):
                file_sp += 1
            else:
                direct_sp += 1
        elif cid in rq_map and rq_map[cid].get("sparql"):
            sparql = rq_map[cid]["sparql"]; from_rq += 1

        if not q_text and cid in rq_map:
            rq_q = rq_map[cid].get("question","")
            if rq_q:
                q_text = rq_q; q_from_rq += 1

        embed_text = f"{q_text}\n[Beat] {beat}" if beat else q_text

        i = len(order)
        order.append(cid)
        rows_map[cid] = {
            "i": i,
            "beat": beat,
            "text": embed_text,
            "sparql": sparql,
            "question": q_text,
            "answer": answer
        }

    if not order:
        print("No rows with CQ_ID found. Nothing to index."); return

    # Embed in 'order'
    texts = [rows_map[cid]["text"] for cid in order]
    X = embed_texts_ollama(texts, args.model, batch=args.batch)
    dim = X.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(X)
    faiss.write_index(index, args.out_index)

    # Also produce legacy arrays for compatibility
    ids       = order
    beats     = [rows_map[cid]["beat"] for cid in order]
    sparqls   = [rows_map[cid]["sparql"] for cid in order]
    questions = [rows_map[cid]["question"] for cid in order]
    answers   = [rows_map[cid]["answer"] for cid in order]

    meta = {
        "model": args.model,
        "dim": dim,
        "count": len(order),
        "order": order,
        "rows": rows_map,
        # legacy arrays:
        "ids": ids,
        "beats": beats,
        "texts": texts,
        "sparqls": sparqls,
        "questions": questions,
        "answers": answers
    }
    Path(args.out_meta).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    missing_sp = len(order) - (direct_sp + file_sp + from_rq)
    print(f"OK: indexed {len(order)} CQs | SPARQL sources: direct={direct_sp}, file={file_sp}, rq={from_rq}, missing={missing_sp}")
    print(f"Questions sourced from .rq headers: {q_from_rq}")
    print(f"Index: {args.out_index}\nMeta:  {args.out_meta}")

if __name__ == "__main__":
    main()
