#!/usr/bin/env python3
# build_cq_index_v2.py
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import subprocess
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional deps for embeddings / faiss
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore


# =============================================================================
# Helpers
# =============================================================================

def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def _json_load(p: Path) -> Any:
    return json.loads(_read_text(p))

def _json_dump(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _as_list(x: Any) -> List[str]:
    """Coerce value to list[str] with trimming; understands JSON-ish strings."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        out = [str(v).strip() for v in x if str(v).strip()]
        return out
    s = str(x).strip()
    if not s:
        return []
    # Try JSON array first (allow parentheses too)
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            arr = json.loads(s.replace("(", "[").replace(")", "]"))
            return _as_list(arr)
        except Exception:
            pass
    # Otherwise split common separators
    for sep in (";", "|", ","):
        if sep in s:
            return [t.strip() for t in s.split(sep) if t.strip()]
    return [s]

def _norm_mode_list(x: Any) -> List[str]:
    return [t.lower() for t in _as_list(x)]

def _mode_matches(row_modes: Any, current_mode: str) -> bool:
    """Return True iff current_mode is present in row's RetrievalMode/mode list (case-insensitive)."""
    m = (current_mode or "").strip().lower()
    vals = _norm_mode_list(row_modes)
    return m in vals

def _slug(s: str) -> str:
    s2 = (s or "").strip().lower()
    s2 = re.sub(r"[^a-z0-9]+", "-", s2)
    return re.sub(r"-+", "-", s2).strip("-")

def _norm_beat_title(x: Any) -> str:
    # For downstream grouping we want a single string here
    if isinstance(x, str) and x.strip():
        return x.strip()
    arr = _as_list(x)
    if arr:
        return arr[0]
    return "Unspecified"

def _first_nonempty(*vals: Any) -> str:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

# =============================================================================
# Load CQ rows (JSON/CSV)
# =============================================================================

def _load_cqs(cq_path: Path) -> List[Dict[str, Any]]:
    if cq_path.suffix.lower() == ".json":
        data = _json_load(cq_path)
        base = data.get("cqs") if isinstance(data, dict) and "cqs" in data else data
        rows: List[Dict[str, Any]] = []
        if isinstance(base, list):
            for v in base:
                if isinstance(v, dict):
                    rows.append(v)
        elif isinstance(base, dict):
            for k, v in base.items():
                if isinstance(v, dict):
                    r = {"id": k}
                    r.update(v)
                    rows.append(r)
        else:
            raise ValueError(f"Unsupported JSON shape in {cq_path}")
        return rows

    # CSV
    rows: List[Dict[str, Any]] = []
    with cq_path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for rec in rd:
            row: Dict[str, Any] = dict(rec)
            # normalize common field names
            cq_id = row.get("CQ_ID") or row.get("id") or row.get("cq_id") or row.get("CQ-ID")
            question = (
                row.get("Refactored CQ") or row.get("Question") or row.get("question") or ""
            )
            beats = row.get("Beats") or row.get("beat_title") or row.get("beat") or row.get("beat_slug") or ""
            retrieval = row.get("RetrievalMode") or row.get("mode") or ""

            row_norm = {
                "id": str(cq_id).strip() if cq_id else "",
                "question": str(question).strip(),
                "Beats": _as_list(beats),
                "RetrievalMode": _norm_mode_list(retrieval),
            }
            rows.append(row_norm)
    return rows

# =============================================================================
# SPARQL template parsing
# =============================================================================

# More permissive: accepts "# CQ-ID:" or "# CQ_ID:" and allows trailing text (e.g., question)
_CQ_SPLIT_RX = re.compile(
    r"^\s*#\s*CQ[_-]?ID\s*:\s*([A-Za-z0-9_\-:.]+)\b.*$",
    re.M | re.I
)

def _parse_rq_text(text: str) -> Dict[str, str]:
    """
    Parse a templated .rq file with sections marked by:
      # CQ-ID: XYZ
    Returns {cq_id -> full_query_string}
    """
    chunks = _CQ_SPLIT_RX.split(text)
    # chunks looks like: [preamble, CQID1, body1, CQID2, body2, ...]
    out: Dict[str, str] = {}
    if len(chunks) < 3:
        return out
    # iterate pairs (cqid, body)
    for i in range(1, len(chunks), 2):
        cid = chunks[i].strip()
        body = chunks[i + 1].strip()
        if cid and body:
            out[cid] = body
    return out

def _collect_rq_maps(
    sparql_root: Optional[Path],
    sparql_kg: Optional[Path],
    sparql_hybrid: Optional[Path],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Priority:
      1) explicit --sparql_kg / --sparql_hybrid
      2) sparql_root/kg/*.rq and sparql_root/hybrid/*.rq (concatenated)
      3) common filenames in root: cqs_queries_template_kg.rq, cqs_queries_template_hybrid.rq
    """
    def parse_many(files: List[Path]) -> Dict[str, str]:
        acc: Dict[str, str] = {}
        for fp in files:
            if fp.exists():
                m = _parse_rq_text(_read_text(fp))
                acc.update(m)  # later files override earlier ones
        return acc

    rq_map_kg: Dict[str, str] = {}
    rq_map_hy: Dict[str, str] = {}

    # 1) explicit
    if sparql_kg and sparql_kg.exists():
        rq_map_kg = parse_many([sparql_kg])
    if sparql_hybrid and sparql_hybrid.exists():
        rq_map_hy = parse_many([sparql_hybrid])

    # 2) root with subfolders
    if sparql_root and sparql_root.exists():
        kg_dir = sparql_root / "kg"
        hy_dir = sparql_root / "hybrid"
        if not rq_map_kg and kg_dir.exists():
            rq_map_kg = parse_many(sorted(kg_dir.glob("*.rq")))
        if not rq_map_hy and hy_dir.exists():
            rq_map_hy = parse_many(sorted(hy_dir.glob("*.rq")))

        # 3) common filenames at root
        if not rq_map_kg:
            cand = sparql_root / "cqs_queries_template_kg.rq"
            if cand.exists():
                rq_map_kg = parse_many([cand])
        if not rq_map_hy:
            cand = sparql_root / "cqs_queries_template_hybrid.rq"
            if cand.exists():
                rq_map_hy = parse_many([cand])

    return rq_map_kg, rq_map_hy

# =============================================================================
# Build per-mode metadata with beat fan-out
# =============================================================================

def _build_meta_for_mode(
    *, mode: str,
    rows_all: List[Dict[str, Any]],
    rq_map_kg: Dict[str, str],
    rq_map_hy: Dict[str, str],
) -> List[Dict[str, Any]]:
    mode_lc = (mode or "").strip().lower()
    keep: List[Dict[str, Any]] = []

    for r in rows_all:
        cq_id  = _first_nonempty(r.get("id"), r.get("CQ_ID"), r.get("cq_id"))
        q_text = _first_nonempty(r.get("question"), r.get("Refactored CQ"), r.get("Question"))
        beats  = _as_list(r.get("Beats") or r.get("beat_title") or r.get("beat") or r.get("beat_slug"))
        if not beats:
            beats = ["Unspecified"]

        row_modes = r.get("RetrievalMode") or r.get("mode")
        if not _mode_matches(row_modes, mode_lc):
            # Strict: only include if current mode is explicitly present
            continue

        # Choose SPARQL per mode
        sparql = ""
        if mode_lc == "kg":
            sparql = rq_map_kg.get(cq_id, "")
        else:
            sparql = rq_map_hy.get(cq_id, "")

        # Fan out one record per beat
        for bt in beats:
            rec = {
                "id": cq_id,
                "question": q_text,
                "beat_title": _norm_beat_title(bt),
                "beat_titles": beats,  # preserve full list
                "RetrievalMode": _as_list(row_modes),
                "mode": _as_list(row_modes),
                "sparql": sparql,
                "source": "csv/json",
            }
            keep.append(rec)

    return keep

# =============================================================================
# Validation / summary
# =============================================================================

def _validate_and_summarize(meta_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    errs: List[str] = []
    by_beat: Counter = Counter()
    with_sparql = 0
    with_question = 0

    for i, r in enumerate(meta_rows):
        cid = r.get("id", "")
        q   = r.get("question", "")
        bt  = r.get("beat_title", "")
        sp  = r.get("sparql", "")

        if not cid:
            errs.append(f"row#{i}: missing id")
        if q:
            with_question += 1
        if bt:
            by_beat[_slug(bt)] += 1
        if sp.strip():
            with_sparql += 1

    summary = {
        "total": len(meta_rows),
        "with_sparql": with_sparql,
        "with_question": with_question,
        "by_beat": by_beat.most_common(),
        "errors": errs,
    }
    return summary

def _print_summary(mode: str, meta_rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    total = summary["total"]
    ws = summary["with_sparql"]
    wq = summary["with_question"]
    by = summary["by_beat"]
    print(f"Mode={mode} | rows={total} | with_sparql={ws} | with_question={wq}")
    if by:
        top = ", ".join([f"('{k}', {v})" for k, v in by[:8]])
        print(f"Top beats: [{top}]")
    if summary["errors"]:
        print(f"Validation: {len(summary['errors'])} issues:")
        for e in summary["errors"][:20]:
            print("  -", e)
        if len(summary["errors"]) > 20:
            print(f"  ... and {len(summary['errors']) - 20} more")

# =============================================================================
# Ollama helpers
# =============================================================================

def _ollama_has_model(name: str) -> bool:
    try:
        cp = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        return (cp.returncode == 0) and (name in cp.stdout)
    except Exception:
        return False

def _ollama_pull(name: str) -> None:
    try:
        subprocess.run(["ollama", "pull", name], check=False)
    except Exception:
        pass

def _ollama_embed(texts: List[str], model: str, endpoint: str = "http://localhost:11434/api/embeddings"):
    if np is None:
        raise RuntimeError("numpy is required for ollama embeddings")
    vecs = []
    for t in texts:
        body = json.dumps({"model": model, "prompt": t}).encode("utf-8")
        req = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        emb = data.get("embedding")
        if not emb:
            raise RuntimeError(f"Ollama embeddings response missing 'embedding' for text: {t[:120]}...")
        vecs.append(emb)
    arr = np.array(vecs, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms  # cosine-ready

# =============================================================================
# Embeddings / FAISS (optional)
# =============================================================================

def _build_embeddings_and_faiss(
    meta_rows: List[Dict[str, Any]],
    out_dir: Path,
    embedder: str,
    sbert_model: Optional[str],
    ollama_model: Optional[str] = None,
    auto_ollama: bool = False,
    ollama_pull_flag: bool = False,
) -> None:
    if not meta_rows:
        print("No rows to embed; skipping embeddings.")
        return
    if np is None:
        print("numpy not available; skipping embeddings.")
        return

    texts = [f"{r.get('question','')} [Beat: {r.get('beat_title','')}]" for r in meta_rows]

    if embedder == "sbert":
        if SentenceTransformer is None:
            print("sentence-transformers not available; skipping embeddings.")
            return
        model_name = sbert_model or "all-MiniLM-L6-v2"
        print(f"Embedding {len(meta_rows)} rows with SBERT ({model_name})...")
        model = SentenceTransformer(model_name)
        embs = model.encode(
            texts, batch_size=64, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32", copy=False)

    elif embedder == "ollama":
        model_name = ollama_model or "nomic-embed-text"
        if auto_ollama and not _ollama_has_model(model_name):
            print(f"Ollama model '{model_name}' not found.")
            if ollama_pull_flag:
                print(f"Pulling '{model_name}'...")
                _ollama_pull(model_name)
        print(f"Embedding {len(meta_rows)} rows with Ollama ({model_name})...")
        embs = _ollama_embed(texts, model=model_name).astype("float32", copy=False)

    else:
        print(f"Embedder '{embedder}' not supported; skipping embeddings.")
        return

    # Save numpy
    npy_path = out_dir / "embeddings.npy"
    np.save(npy_path, embs)
    print(f"Saved embeddings → {npy_path}")

    # FAISS (cosine via IP with normalized vectors)
    if faiss is None:
        print("faiss not installed; skipping faiss.index")
        return
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss_path = out_dir / "faiss.index"
    faiss.write_index(index, str(faiss_path))
    print(f"Saved FAISS index → {faiss_path}")

# =============================================================================
# Main build
# =============================================================================

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def build_index(
    *,
    cq_path: Path,
    retrieval_mode: str,
    out_root: Path,
    sparql_root: Optional[Path],
    sparql_kg: Optional[Path],
    sparql_hybrid: Optional[Path],
    build_faiss: bool,
    embedder: str,
    sbert_model: Optional[str],
    validate: bool,
    # Ollama
    ollama_model: Optional[str] = None,
    auto_ollama: bool = False,
    ollama_pull: bool = False,
) -> None:
    retrieval_mode = retrieval_mode.strip()
    assert retrieval_mode in {"KG", "Hybrid", "Both"}, "--retrieval_mode must be one of KG|Hybrid|Both"

    # Load rows
    rows_all = _load_cqs(cq_path)
    print(f"Loaded {len(rows_all)} CQ rows from {cq_path}")

    # Collect SPARQL maps
    rq_map_kg, rq_map_hy = _collect_rq_maps(sparql_root, sparql_kg, sparql_hybrid)
    print(f"SPARQL map sizes → KG: {len(rq_map_kg)} | Hybrid: {len(rq_map_hy)}")

    modes_to_build = ["KG", "Hybrid"] if retrieval_mode == "Both" else [retrieval_mode]

    for mode in modes_to_build:
        print(f"\n== Building mode: {mode} ==")
        out_dir = _ensure_dir(out_root / mode)
        meta_rows = _build_meta_for_mode(mode=mode, rows_all=rows_all, rq_map_kg=rq_map_kg, rq_map_hy=rq_map_hy)

        # Write metadata
        meta_path = out_dir / "cq_metadata.json"
        _json_dump(meta_path, {"cqs": meta_rows})
        print(f"Wrote metadata → {meta_path}")

        # Validate / summarize
        if validate:
            summary = _validate_and_summarize(meta_rows)
            _print_summary(mode, meta_rows, summary)

        # Embeddings / FAISS
        if build_faiss:
            _build_embeddings_and_faiss(
                meta_rows, out_dir, embedder, sbert_model,
                ollama_model=ollama_model, auto_ollama=auto_ollama, ollama_pull_flag=ollama_pull
            )

# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Build per-mode CQ index with beat fan-out and optional FAISS (SBERT or Ollama).")
    ap.add_argument("--cq_path", required=True, help="Path to CQ JSON/CSV (supports fields CQ_ID/Refactored CQ/Beats/RetrievalMode).")
    ap.add_argument("--retrieval_mode", required=True, choices=["KG", "Hybrid", "Both"], help="Which mode to build.")
    ap.add_argument("--out_root", required=True, help="Output root directory; per-mode dirs will be created here.")

    ap.add_argument("--sparql_root", default=None, help="Root with 'kg/' and 'hybrid/' .rq files (or common filenames).")
    ap.add_argument("--sparql_kg", default=None, help="Explicit path to KG .rq template file.")
    ap.add_argument("--sparql_hybrid", default=None, help="Explicit path to Hybrid .rq template file.")

    ap.add_argument("--build_faiss", action="store_true", help="Compute embeddings and FAISS index.")
    ap.add_argument("--embedder", default="sbert", choices=["sbert","ollama"], help="Embedding backend.")
    ap.add_argument("--sbert_model", default="all-MiniLM-L6-v2", help="SBERT model name.")
    ap.add_argument("--ollama_model", default="nomic-embed-text", help="Ollama embedding model id.")
    ap.add_argument("--auto_ollama", action="store_true", help="Auto-detect Ollama model presence.")
    ap.add_argument("--ollama_pull", action="store_true", help="If auto_ollama & model missing, run 'ollama pull'.")

    ap.add_argument("--validate", action="store_true", help="Print summary and basic metadata validation.")
    args = ap.parse_args()

    cq_path = Path(args.cq_path)
    out_root = Path(args.out_root)
    sparql_root = Path(args.sparql_root) if args.sparql_root else None
    sparql_kg = Path(args.sparql_kg) if args.sparql_kg else None
    sparql_hybrid = Path(args.sparql_hybrid) if args.sparql_hybrid else None

    build_index(
        cq_path=cq_path,
        retrieval_mode=args.retrieval_mode,
        out_root=out_root,
        sparql_root=sparql_root,
        sparql_kg=sparql_kg,
        sparql_hybrid=sparql_hybrid,
        build_faiss=bool(args.build_faiss),
        embedder=args.embedder,
        sbert_model=args.sbert_model,
        validate=bool(args.validate),
        # Ollama
        ollama_model=args.ollama_model,
        auto_ollama=bool(args.auto_ollama),
        ollama_pull=bool(args.ollama_pull),
    )

if __name__ == "__main__":
    main()
