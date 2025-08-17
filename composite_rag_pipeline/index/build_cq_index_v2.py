#!/usr/bin/env python3
# build_cq_index_v2.py
"""
Builds per-mode CQ index folders with clean metadata (and optional embeddings/FAISS).

What it does
------------
- Loads CQs from JSON or CSV (id, question, beat, RetrievalMode).
- Loads SPARQL per CQ from filled-in .rq files (KG / Hybrid) and
  **extracts questions from the '#CQ-ID:<id> <question>' header** (or '#QUESTION:' on the next line).
- Normalizes beat_title to a plain string and adds beat_slug.
- (Optional) Remaps beats via --beat_map (slug -> target title).
- Writes per-mode folders: <out_root>/<MODE>/cq_metadata.json (+ id_map.json).
- (Optional) Embeds questions and saves embeddings.npy (+ FAISS index if available).
- (Optional) Validates coverage and prints a summary.

Assumptions
-----------
- CQ IDs look like:  CQ-...  (regex: r"\\bCQ[-A-Za-z0-9_]*\\b")
- Filled .rq files contain sections beginning with a header line like:
    #CQ-ID:CQ-L12 How many viewers watched [Event] globally?
  or
    # CQ-ID: CQ-L12
    # QUESTION: How many viewers watched [Event] globally?
  The SPARQL block continues until the next '#CQ-ID:'.

CLI examples
------------
# KG only, meta only (explicit .rq path)
python3 build_cq_index_v2.py \
  --cq_path ../data/WembleyRewindCQs_with_beats_trimmed.json \
  --retrieval_mode KG \
  --out_root ../index \
  --rq_kg ../data/sparql_templates/kg/cqs_queries_template_filled_in.rq \
  --validate

# Hybrid only with embeddings/FAISS (SBERT)
python3 build_cq_index_v2.py \
  --cq_path ../data/WembleyRewindCQs_with_beats_trimmed.json \
  --retrieval_mode Hybrid \
  --out_root ../index \
  --rq_hybrid ../data/sparql_templates/hybrid/cqs_queries_template_filled_in.rq \
  --build_faiss \
  --embedder sbert --sbert_model all-MiniLM-L6-v2 \
  --validate

# Both modes in one shot (meta only)
python3 build_cq_index_v2.py \
  --cq_path ../data/WembleyRewindCQs_with_beats_trimmed.json \
  --retrieval_mode Both \
  --out_root ../index \
  --rq_kg ../data/sparql_templates/kg/cqs_queries_template_filled_in.rq \
  --rq_hybrid ../data/sparql_templates/hybrid/cqs_queries_template_filled_in.rq \
  --validate
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional deps
try:
    import numpy as np
except Exception:
    np = None

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

# SBERT (optional)
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore

# ----------------------------- regexes ---------------------------------

CQ_RX = re.compile(r"\bCQ[-A-Za-z0-9_]*\b")
# Header like: "#CQ-ID:CQ-L12 How many viewers…"
CQ_HEADER_RX = re.compile(
    r"^\s*#\s*C?Q?-?\s*ID\s*:\s*(\S+)(?:\s+(.*))?\s*$",
    re.IGNORECASE,
)
# Optional 2nd-line question like: "#QUESTION: …" or "#Q: …"
QUESTION_RX = re.compile(
    r"^\s*#\s*(?:QUESTION|Q)\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)

# ----------------------------- helpers ---------------------------------

def _slug_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s-]+", "", s)
    s = re.sub(r"\s+", "-", s)
    return re.sub(r"-{2,}", "-", s).strip("-")

def _to_text(x: Any) -> str:
    """Coerce beat-like values (str/list/dict/other) to a readable string."""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        for y in x:
            if isinstance(y, str) and y.strip():
                return y
        return " / ".join(str(y) for y in x if y is not None)
    if isinstance(x, dict):
        for k in ("title", "label", "name", "value"):
            v = x.get(k)
            if isinstance(v, str) and v.strip():
                return v
        vals = [str(v) for v in x.values() if isinstance(v, (str, int, float))]
        return " / ".join(vals)
    if x is None:
        return ""
    return str(x)

def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _infer_rq_paths(sparql_root: Optional[Path], rq_kg: Optional[Path], rq_hy: Optional[Path]) -> Tuple[Optional[Path], Optional[Path]]:
    """If explicit files not provided, try common names under sparql_root."""
    if rq_kg is None and sparql_root:
        for name in [
            "cqs_queries_template_filled_in.rq",
            "cqs_queries_template_kg.rq",
            "cqs_queries_template_filled_in_kg.rq",
        ]:
            cand = sparql_root / name
            if cand.exists():
                rq_kg = cand
                break
    if rq_hy is None and sparql_root:
        for name in [
            "cqs_queries_template_filled_in_hybrid.rq",
            "cqs_queries_template_hybrid.rq",
        ]:
            cand = sparql_root / name
            if cand.exists():
                rq_hy = cand
                break
    return rq_kg, rq_hy

# -------------------------- load CQs (CSV/JSON) -------------------------

@dataclass
class CQRow:
    id: str
    question: str
    beat_title: str
    mode: Optional[str]  # "KG" / "Hybrid" or None

def _detect_id_col(cols: List[str]) -> Optional[str]:
    prefs = ["CQ_ID", "cq_id", "id", "Id", "CQ", "cq", "CQID", "cqid", "CQ-ID", "cq-id"]
    for p in prefs:
        if p in cols:
            return p
    return None

def _rows_from_csv(p: Path, mode_filter: Optional[str]) -> List[CQRow]:
    out: List[CQRow] = []
    with p.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        cols = rdr.fieldnames or []
        id_col = _detect_id_col(cols)
        q_cols = [c for c in cols if c.lower() in ("question", "cq_question", "text")]
        beat_cols = [c for c in cols if c.lower() in ("beats", "beat_title", "beattitle", "beatlabel", "beat_name")]
        mode_col = "RetrievalMode" if "RetrievalMode" in cols else None

        for row in rdr:
            raw = (row.get(id_col) or "").strip() if id_col else ""
            m = CQ_RX.search(raw)
            cid = m.group(0) if m else raw
            if not cid:
                continue

            mode = (row.get(mode_col) or "").strip() if mode_col else None
            if mode_filter and mode and mode != mode_filter:
                continue

            q = ""
            for qc in q_cols:
                if row.get(qc):
                    q = row[qc].strip()
                    if q:
                        break

            bt = ""
            for bc in beat_cols:
                if row.get(bc):
                    bt = _to_text(row[bc]).strip()
                    if bt:
                        break

            out.append(CQRow(id=cid, question=q, beat_title=bt, mode=mode))
    return out

def _rows_from_json(p: Path, mode_filter: Optional[str]) -> List[CQRow]:
    data = _load_json(p)
    if isinstance(data, dict):
        rows = data.get("items") or data.get("data") or data.get("rows") or data.get("cqs") or []
    elif isinstance(data, list):
        rows = data
    else:
        rows = []

    out: List[CQRow] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        raw_id = (r.get("id") or r.get("CQ_ID") or r.get("cq_id") or r.get("CQ") or "")
        m = CQ_RX.search(str(raw_id))
        cid = m.group(0) if m else str(raw_id)
        if not cid:
            continue

        mode = (r.get("RetrievalMode") or r.get("mode") or None)
        # if isinstance(mode, list):
        #     mode = next((x for x in mode if isinstance(x, str) and x.strip()), None)
        if isinstance(mode, str):
            mode = [mode.strip()]
        if mode_filter and mode and mode != mode_filter:
            continue

        q = (r.get("question") or r.get("Question") or "") or ""
        bt_raw = r.get("beat_title") or r.get("Beats") or r.get("beats") or r.get("beatLabel") or r.get("beat_name")
        bt = _to_text(bt_raw).strip()
        out.append(CQRow(id=cid, question=q, beat_title=bt, mode=mode))
    return out

def load_cq_rows(cq_path: Path, mode_filter: Optional[str]) -> List[CQRow]:
    if cq_path.suffix.lower() == ".csv":
        return _rows_from_csv(cq_path, mode_filter)
    return _rows_from_json(cq_path, mode_filter)

# ------------------------- parse filled-in .rq ---------------------------

def parse_filled_rq_with_questions(path: Path) -> Dict[str, Dict[str, str]]:
    """
    Returns mapping:
      { CQ_ID: { "sparql": "...", "question": "..." } }

    Accepts headers:
      '#CQ-ID:<id> <question text>'
    or
      '# CQ-ID: <id>' on one line + '#QUESTION: <text>' on a following line
    The SPARQL block is the non-comment lines until the next '#CQ-ID:' header.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    out: Dict[str, Dict[str, str]] = {}
    current_id: Optional[str] = None
    current_question: str = ""
    block: List[str] = []

    def flush():
        nonlocal current_id, current_question, block, out
        if current_id is None:
            return
        sparql_lines = [ln for ln in block if not ln.strip().startswith("#")]
        sparql = "\n".join(sparql_lines).strip()
        out[current_id] = {
            "sparql": sparql,
            "question": current_question.strip(),
        }

    for ln in lines:
        m = CQ_HEADER_RX.match(ln)
        if m:
            # new section begins
            flush()
            raw_id = m.group(1) or ""
            # sanitize id by extracting the CQ token
            id_match = CQ_RX.search(raw_id)
            current_id = id_match.group(0) if id_match else raw_id.strip()
            current_question = (m.group(2) or "").strip()
            block = []
            continue

        # secondary question line
        if not current_question:
            qm = QUESTION_RX.match(ln)
            if qm:
                current_question = qm.group(1) or ""

        # collect line
        block.append(ln)

    flush()
    return out

# ----------------------- build per-mode metadata ------------------------

def build_metadata_for_mode(
    rows: List[CQRow],
    rq_map: Dict[str, Dict[str, str]],
    mode_name: str,
    beat_map: Dict[str, str],
    prefer_rq_question: bool,
) -> Dict[str, Dict[str, Any]]:
    """
    Returns {CQ_ID: record}
    record: {question, sparql, beat_title, beat_slug, mode}
    """
    meta: Dict[str, Dict[str, Any]] = {}

    for r in rows:
        if mode_name and r.mode and mode_name not in r.mode:
            continue  # row is explicitly another mode

        # SPARQL + question from .rq (if present)
        rq_entry = rq_map.get(r.id) or {}
        rq_sparql = (rq_entry.get("sparql") or "").strip()
        rq_question = (rq_entry.get("question") or "").strip()

        # Question precedence
        if prefer_rq_question and rq_question:
            q = rq_question
        else:
            q = (r.question or rq_question or "").strip()

        # Beat normalization/remap
        bt = (r.beat_title or "").strip()
        src_slug = _slug_text(bt)
        if src_slug in beat_map:
            bt = beat_map[src_slug]

        rec = {
            "question": q,
            "sparql": rq_sparql,
            "beat_title": bt,
            "beat_slug": _slug_text(bt),
            "mode": mode_name or (r.mode or ""),
        }
        meta[r.id] = rec
    return meta

# ----------------------------- embeddings -------------------------------

def embed_questions(meta: Dict[str, Dict[str, Any]], embedder: str, sbert_model: Optional[str]) -> Tuple[List[str], Optional["np.ndarray"]]:
    ids = sorted(meta.keys())
    texts = [meta[i].get("question", "") for i in ids]

    if np is None:
        print("numpy not available; skipping embeddings.")
        return ids, None

    if embedder == "sbert":
        if SentenceTransformer is None:
            print("sentence-transformers not available; skipping embeddings.")
            return ids, None
        model_name = sbert_model or "all-MiniLM-L6-v2"
        print(f"Embedding {len(texts)} questions via SBERT: {model_name}")
        model = SentenceTransformer(model_name)
        X = model.encode(texts, normalize_embeddings=True)
        X = np.asarray(X, dtype="float32")
        return ids, X

    print(f"Unknown embedder '{embedder}'; skipping embeddings.")
    return ids, None

def build_faiss_index(emb: "np.ndarray") -> Optional[Any]:
    if faiss is None or np is None:
        print("faiss or numpy not available; skipping FAISS index.")
        return None
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine if embeddings normalized
    index.add(emb)
    return index

# ------------------------------- validate -------------------------------

def validate_metadata(meta: Dict[str, Dict[str, Any]], mode: str) -> Dict[str, Any]:
    n = len(meta)
    with_sparql = sum(1 for r in meta.values() if (r.get("sparql") or "").strip())
    with_beat = sum(1 for r in meta.values() if (r.get("beat_title") or "").strip())
    with_questions = sum(1 for r in meta.values() if (r.get("question") or "").strip())
    by_beat: Dict[str, int] = {}
    for r in meta.values():
        b = r.get("beat_slug") or _slug_text(r.get("beat_title") or "")
        if b:
            by_beat[b] = by_beat.get(b, 0) + 1
    return {
        "mode": mode,
        "count": n,
        "with_sparql": with_sparql,
        "with_question": with_questions,
        "with_beat_title": with_beat,
        "beats_top": sorted(by_beat.items(), key=lambda x: (-x[1], x[0]))[:10],
    }

# --------------------------------- main ---------------------------------


def _mode_matches(r_mode, build_mode: str) -> bool:
    # no mode on the row → treat as "applies to all"
    if r_mode is None or (isinstance(r_mode, str) and r_mode.strip() == ""):
        return True

    b = build_mode.strip().lower()

    # string cases
    if isinstance(r_mode, str):
        s = r_mode.strip().lower()
        if s in ("both", "kg+hybrid"):     # treat as both
            return True
        return s == b                      # exact match only

    # list/tuple/set cases
    if isinstance(r_mode, (list, tuple, set)):
        vals = [str(x).strip().lower() for x in r_mode]
        if "both" in vals or "kg+hybrid" in vals:
            return True
        return b in vals

    # anything else → allow
    return True


def main():
    ap = argparse.ArgumentParser(description="Build per-mode CQ index (metadata + optional embeddings/FAISS).")
    ap.add_argument("--cq_path", required=True, help="Path to CQ JSON or CSV (has CQ_ID/Question/Beat/RetrievalMode).")
    ap.add_argument("--retrieval_mode", required=True, choices=["KG", "Hybrid", "Both"], help="Which mode to build.")
    ap.add_argument("--out_root", required=True, help="Output root directory (creates <out_root>/<MODE>/).")

    # SPARQL inputs
    ap.add_argument("--sparql_root", default=None, help="Directory containing filled-in .rq files (not recursive).")
    ap.add_argument("--rq_kg", default=None, help="Explicit KG .rq file path.")
    ap.add_argument("--rq_hybrid", default=None, help="Explicit Hybrid .rq file path.")

    # Beat remap (optional)
    ap.add_argument("--beat_map", default=None, help="JSON mapping of source beat slugs -> target narrative titles")

    # Prefer question from .rq header over CSV/JSON question
    ap.add_argument("--prefer_rq_question", action="store_true", help="If set, override CQ question with the '#CQ-ID:' header question.")

    # Embeddings / FAISS
    ap.add_argument("--build_faiss", action="store_true", help="Build embeddings.npy and FAISS index.")
    ap.add_argument("--embedder", default="sbert", choices=["sbert"], help="Embedding backend (default: sbert).")
    ap.add_argument("--sbert_model", default="all-MiniLM-L6-v2", help="SBERT model name when --embedder sbert.")

    # Validation
    ap.add_argument("--validate", action="store_true", help="Print validation summary.")

    args = ap.parse_args()

    cq_path = Path(args.cq_path)
    out_root = Path(args.out_root)
    sparql_root = Path(args.sparql_root) if args.sparql_root else None
    rq_kg = Path(args.rq_kg) if args.rq_kg else None
    rq_hy = Path(args.rq_hybrid) if args.rq_hybrid else None
    rq_kg, rq_hy = _infer_rq_paths(sparql_root, rq_kg, rq_hy)

    if args.retrieval_mode in ("KG", "Both") and rq_kg is None:
        print("ERROR: KG .rq file not found. Provide --rq_kg or --sparql_root with a KG template.", file=sys.stderr)
        sys.exit(2)
    if args.retrieval_mode in ("Hybrid", "Both") and rq_hy is None:
        print("ERROR: Hybrid .rq file not found. Provide --rq_hybrid or --sparql_root with a Hybrid template.", file=sys.stderr)
        sys.exit(2)

    # Beat remap
    beat_map: Dict[str, str] = {}
    if args.beat_map:
        try:
            raw = _load_json(Path(args.beat_map))
            if isinstance(raw, dict):
                beat_map = {k.strip().lower(): v for k, v in raw.items()}
        except Exception as e:
            print(f"WARNING: could not load beat_map {args.beat_map}: {e}")

    # Load rows (once; we'll filter per-mode)
    rows_all = load_cq_rows(cq_path, mode_filter=None)

    # Load SPARQL+question maps
    rq_map_kg: Dict[str, Dict[str, str]] = parse_filled_rq_with_questions(rq_kg) if rq_kg else {}
    rq_map_hy: Dict[str, Dict[str, str]] = parse_filled_rq_with_questions(rq_hy) if rq_hy else {}

    modes_to_build = ["KG", "Hybrid"] if args.retrieval_mode == "Both" else [args.retrieval_mode]

    for mode in modes_to_build:
        print(f"\n== Building mode: {mode} ==")
        # Filter rows for this mode (keep mode-less rows too)

        #print(f"rows_all : {rows_all}")
        for r in rows_all:
            print(f"Row: {r.id} | Question: {r.question} | Beat: {r.beat_title} | Mode: {r.mode}")
        rows = [r for r in rows_all if (r.mode is None or mode in r.mode)]
        print(f"rows : {rows}")
        rq_map = rq_map_kg if mode == "KG" else rq_map_hy

        # Build metadata
        meta = build_metadata_for_mode(
            rows=rows,
            rq_map=rq_map,
            mode_name=mode,
            beat_map=beat_map,
            prefer_rq_question=args.prefer_rq_question,
        )

        # Ensure out dir
        out_dir = out_root / mode
        _ensure_dir(out_dir)

        # Write metadata + id_map
        meta_obj = {"metadata": meta}
        (out_dir / "cq_metadata.json").write_text(json.dumps(meta_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        ids_sorted = sorted(meta.keys())
        id_map = {cid: i for i, cid in enumerate(ids_sorted)}
        (out_dir / "id_map.json").write_text(json.dumps(id_map, ensure_ascii=False, indent=2), encoding="utf-8")

        # Embeddings / FAISS
        if args.build_faiss:
            ids, E = embed_questions(meta, args.embedder, args.sbert_model)
            if E is not None:
                if np is not None:
                    np.save(str(out_dir / "embeddings.npy"), E)
                if faiss is not None:
                    index = build_faiss_index(E)
                    if index is not None:
                        faiss.write_index(index, str(out_dir / "faiss.index"))

        # Validate
        if args.validate:
            stats = validate_metadata(meta, mode)
            print(f"Mode={stats['mode']} | CQs={stats['count']} | with_sparql={stats['with_sparql']} | with_question={stats['with_question']} | with_beat_title={stats['with_beat_title']}")
            print("Top beats:", stats["beats_top"] or "(none)")
            missing = [cid for cid, r in meta.items() if not (r.get("sparql") or "").strip()]
            if missing:
                print(f"Missing SPARQL for {len(missing)} CQs (showing up to 20):")
                for cid in missing[:20]:
                    print(" -", cid)

        print(f"Wrote: {out_dir/'cq_metadata.json'}")
        if args.build_faiss:
            print(f"Embeddings: {'OK' if (out_dir/'embeddings.npy').exists() else 'skipped'} | "
                  f"FAISS: {'OK' if (out_dir/'faiss.index').exists() else 'skipped'}")

    print("\nDone.")

if __name__ == "__main__":
    main()
