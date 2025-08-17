#!/usr/bin/env python3
"""
cq_build_all.py — One script to do it all for your CQ pipeline

What it does
------------
1) Load & normalize your CQ file (CSV/JSON) into a clean in-memory schema.
2) Map SPARQL from .rq templates (per-mode), optionally override Question from header.
3) Build per-mode outputs:
   - cq_index.jsonl
   - cq_metadata.json  (schema: {"order":[...], "metadata": {id -> {...}}})
   - order.txt, ids.json
   - embeddings.npy    (if --build_faiss + embedder chosen)
   - faiss.index       (if FAISS available)
   - validation_report.json
   - sparql_manifest.json
4) Supports: --retrieval_mode KG|Hybrid|Both

Key options
-----------
--embedder {none,sbert,ollama}
--sbert_model all-MiniLM-L6-v2
--ollama_model nomic-embed-text  --auto_ollama --ollama_pull
--embed_text {question,question+answer,question+beats,qa+beats}  (default: question+beats)
--normalize / --no-normalize (default: normalize vectors)
--index_type {flatip,ivfflat} (default: flatip)
--override_question (use header text from .rq for Question when present)
--beats_sep / --mode_sep (if your CSV uses custom separators)

Typical run
-----------
python3 cq_build_all.py \
  --cq_path ../data/WembleyRewindCQs_with_beats_trimmed.json \
  --sparql_root ./sparql \
  --retrieval_mode Both \
  --out_root ./ \
  --build_faiss \
  --embedder sbert \
  --sbert_model all-MiniLM-L6-v2 \
  --override_question \
  --validate
"""
from __future__ import annotations

import argparse, json, csv, re, os, time, atexit, subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# Optional deps
def _try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None

np = _try_import("numpy")
faiss = _try_import("faiss")
requests = _try_import("requests")

# ------------- Helpers: normalization & parsing -------------

_ID_KEYS        = ["CQ_ID", "CQ-ID", "id", "ID"]
_Q_KEYS         = ["Question", "question", "CQ"]
_A_KEYS         = ["Answer", "answer", "ExpectedAnswer", "AnswerTemplate", "TemplateAnswer", "Concrete Answer", "ConcreteAnswer"]
_BEATS_KEYS     = ["Beats", "beat", "beats"]
_MODE_KEYS      = ["RetrievalMode", "retrieval_mode", "mode"]

_DIRECT_SP_KEYS = ["SPARQL", "sparql", "SPARQLText", "sparql_text"]
_FILE_SP_KEYS   = ["SPARQLFile", "SPARQL_Path", "SPARQLFilePath", "sparql_file", "sparql_path"]

RQ_HEADER_RE = re.compile(r'^#CQ-ID\s*:\s*([^\s]+)(?:\s+(.+))?$', re.IGNORECASE)

def _to_list(val, sep=None):
    if val is None: return []
    if isinstance(val, list): return [str(x).strip() for x in val]
    s = str(val).strip()
    if s == "": return []
    if sep: return [x.strip() for x in s.split(sep) if x.strip()]
    if "|" in s: return [x.strip() for x in s.split("|") if x.strip()]
    if "," in s: return [x.strip() for x in s.split(",") if x.strip()]
    return [s]

def _load_any_cq(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.values()) if isinstance(data, dict) else data
    elif path.suffix.lower() in (".csv", ".tsv"):
        delim = "," if path.suffix.lower()==".csv" else "\t"
        rows=[]
        with path.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f, delimiter=delim)
            for row in r:
                # Support JSON arrays stored as strings in CSV
                mr = row.get("RetrievalMode")
                if isinstance(mr,str) and mr.strip().startswith("["):
                    try: row["RetrievalMode"] = json.loads(mr)
                    except Exception: pass
                bt = row.get("Beats")
                if isinstance(bt,str) and bt.strip().startswith("["):
                    try: row["Beats"] = json.loads(bt)
                    except Exception: pass
                rows.append(row)
        return rows
    else:
        raise ValueError(f"Unsupported CQ file type: {path.suffix}")

def _normalize_cqs(items: List[Dict[str, Any]], beats_sep=None, mode_sep=None) -> List[Dict[str, Any]]:
    out=[]; seen=set()
    for rec in items:
        cid = (rec.get("CQ_ID") or rec.get("id") or rec.get("CQ-ID") or rec.get("ID") or "").strip()
        q   = (rec.get("Question") or rec.get("question") or rec.get("CQ") or "").strip()
        ans = (rec.get("Answer") or rec.get("answer") or rec.get("ExpectedAnswer") or "").strip()
        beats = rec.get("Beats") or rec.get("beats") or rec.get("beat") or ""
        modes = rec.get("RetrievalMode") or rec.get("retrieval_mode") or rec.get("mode") or ""
        if isinstance(modes, str) and modes.strip().startswith("["):
            try: modes = json.loads(modes)
            except Exception: pass
        beats = _to_list(beats, sep=beats_sep)
        modes = _to_list(modes, sep=mode_sep) or ["KG","Hybrid"]
        if not cid: cid = f"Q::{abs(hash(q))}"
        if cid in seen: raise ValueError(f"Duplicate CQ_ID: {cid}")
        seen.add(cid)
        out.append({"CQ_ID": cid, "Question": q, "Answer": ans, "Beats": beats, "RetrievalMode": modes,
                    # Preserve any prefilled SPARQL/SPARQLFile so mapper can respect precedence
                    "SPARQL": rec.get("SPARQL") or rec.get("sparql") or "",
                    "SPARQLFile": rec.get("SPARQLFile") or rec.get("sparql_file") or ""})
    return out

# ------------- SPARQL mapping from .rq -------------

def _scan_rq_files(sparql_root: Optional[Path]) -> Tuple[Dict[str, str], Dict[str, str]]:
    rq_by_id: Dict[str, str] = {}; q_from_rq: Dict[str, str] = {}
    if not sparql_root or not sparql_root.exists(): return rq_by_id, q_from_rq
    for p in sparql_root.rglob("*.rq"):
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        cur_id=None; cur_q=None; body=[]
        for line in lines:
            m = RQ_HEADER_RE.match(line.strip())
            if m:
                if cur_id and body:
                    rq_by_id[cur_id] = "\n".join(body).strip()
                    if cur_q: q_from_rq[cur_id]=cur_q
                cur_id = m.group(1).strip()
                cur_q  = (m.group(2) or "").strip()
                body=[]
            else:
                body.append(line)
        if cur_id and body:
            rq_by_id[cur_id] = "\n".join(body).strip()
            if cur_q: q_from_rq[cur_id]=cur_q
    return rq_by_id, q_from_rq

# ------------- Embedding + FAISS -------------

def _l2_normalize(x):
    if np is None: return x
    x = np.asarray(x, dtype="float32")
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

def _embed_sbert(texts: List[str], model_name: str, normalize: bool):
    st = _try_import("sentence_transformers")
    if not st: raise RuntimeError("Install: pip install sentence-transformers")
    model = st.SentenceTransformer(model_name)
    vecs = model.encode(texts, normalize_embeddings=normalize, convert_to_numpy=True)
    if not normalize and np is not None:
        vecs = np.asarray(vecs, dtype="float32")
    dim = int(vecs.shape[1])
    return vecs, dim, "sbert"

def _ollama_up(host: str, timeout: float = 1.0) -> bool:
    if not requests: return False
    try:
        r = requests.get(host.rstrip("/") + "/", timeout=timeout)
        if r.ok and "Ollama is running" in (r.text or ""): return True
    except Exception: pass
    try:
        r = requests.get(host.rstrip("/") + "/api/tags", timeout=timeout)
        return r.ok
    except Exception:
        return False

def _find_free_port() -> int:
    import socket as _s
    with _s.socket(_s.AF_INET, _s.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def _ensure_ollama(host: str, auto: bool, pull_model: bool, model: str) -> str:
    if _ollama_up(host):
        if pull_model: subprocess.run(["ollama","pull",model], check=False)
        return host
    if not auto: return host
    port = _find_free_port()
    auto_host = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    proc = subprocess.Popen(["ollama","serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    def _kill(): 
        try: proc.terminate()
        except Exception: pass
    atexit.register(_kill)
    start = time.time()
    while time.time() - start < 10.0:
        if _ollama_up(auto_host, timeout=0.5):
            if pull_model: subprocess.run(["ollama","pull",model], check=False, env=env)
            return auto_host
        time.sleep(0.25)
    return host

def _embed_ollama(texts: List[str], model: str, host: str, normalize: bool):
    if not requests: raise RuntimeError("Install: pip install requests")
    url = host.rstrip("/") + "/api/embeddings"
    vecs=[]; dim=None
    for t in texts:
        resp = requests.post(url, json={"model": model, "prompt": t}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        emb = data.get("embedding") or (data.get("data", [{}])[0].get("embedding") if isinstance(data.get("data"), list) else None)
        if not emb: raise RuntimeError("Ollama embeddings: missing 'embedding'")
        dim = dim or len(emb)
        vecs.append(emb)
    if np is not None:
        vecs = np.asarray(vecs, dtype="float32")
        if normalize: vecs = _l2_normalize(vecs)
    return vecs, int(dim), "ollama"

def _build_faiss(vectors, out_path: Path, index_type: str = "flatip"):
    if not (faiss and np is not None):
        return False, "faiss or numpy not available"
    xb = np.asarray(vectors, dtype="float32")
    d = xb.shape[1]
    if index_type == "ivfflat":
        n = xb.shape[0]
        nlist = max(16, min(4096, int((n ** 0.5) * 2)))
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(xb)
        index.add(xb)
    else:
        index = faiss.IndexFlatIP(d)
        index.add(xb)
    faiss.write_index(index, str(out_path))
    return True, f"{out_path.name} written (type={index_type}, d={d}, n={xb.shape[0]})"

# ------------- Validation & Manifest -------------

def _validate_metadata(meta_path: Path) -> dict:
    report = {"file": str(meta_path), "ok": True, "errors": [],
              "counts": {"total": 0, "missing_question": 0, "missing_beats": 0, "missing_mode": 0, "missing_sparqlsource": 0}}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        report["ok"] = False
        report["errors"].append(f"parse error: {e}")
        return report
    order = meta.get("order") or []; rows = meta.get("metadata") or {}
    report["counts"]["total"] = len(order)
    for sid in order:
        rec = rows.get(sid) or {}
        q = (rec.get("question") or "").strip()
        beats = rec.get("beats")
        mode = rec.get("retrieval_mode")
        sps = (rec.get("sparql_source") or "").strip()
        if not q:
            report["ok"] = False; report["counts"]["missing_question"] += 1
            report["errors"].append(f"{sid}: missing question")
        if beats is None or (isinstance(beats, list) and len(beats) == 0):
            report["ok"] = False; report["counts"]["missing_beats"] += 1
            report["errors"].append(f"{sid}: beats missing/empty")
        if mode is None or (isinstance(mode, list) and len(mode) == 0):
            report["ok"] = False; report["counts"]["missing_mode"] += 1
            report["errors"].append(f"{sid}: retrieval_mode missing/empty")
        if not sps:
            report["ok"] = False; report["counts"]["missing_sparqlsource"] += 1
            report["errors"].append(f"{sid}: sparql_source missing")
    return report

def _write_manifest(meta_path: Path):
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    order = data.get("order") or []; rows = data.get("metadata") or {}
    counts = {k:0 for k in ["direct","file","rq_scan","fallback_named","missing"]}
    manifest = {}
    for sid in order:
        rec = rows.get(sid, {})
        src = (rec.get("sparql_source") or "missing").strip()
        sparql = rec.get("sparql") or ""
        counts[src] = counts.get(src, 0) + 1
        manifest[sid] = {
            "source": src,
            "has_sparql": bool(sparql.strip()),
            "sparql_len": len(sparql),
            "preview": (sparql.strip().splitlines() or [""])[0][:200],
            "question": rec.get("question","")
        }
    out = {"file": str(meta_path), "total": len(order), "counts": counts, "manifest": manifest}
    out_path = meta_path.with_name("sparql_manifest.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path

# ------------- Core per-mode build -------------

def _compose_embed_text(kind: str, question: str, answer: str, beats) -> str:
    btxt = ", ".join(beats) if isinstance(beats, (list, tuple)) else str(beats or "").strip()
    if kind == "question": return question
    if kind == "question+answer": return f"{question}\nAnswer template: {answer}".strip()
    if kind == "question+beats": return f"{question}\nBeats: {btxt}".strip()
    if kind == "qa+beats":
        parts=[question]
        if answer: parts.append("Answer template: "+answer)
        if btxt: parts.append("Beats: "+btxt)
        return "\n".join(parts)
    return question

def _build_for_mode(norm_items: List[Dict[str, Any]], retrieval_mode: str, out_root: Path, sparql_root: Optional[Path],
                    override_question: bool, build_faiss: bool, embedder: str, sbert_model: str,
                    ollama_model: str, ollama_host: str, auto_ollama: bool, ollama_pull: bool,
                    embed_text: str, normalize: bool, index_type: str, validate: bool) -> Dict[str, Any]:

    # scope to chosen mode
    items = [r for r in norm_items if retrieval_mode in {m.strip() for m in (r.get("RetrievalMode") or [])}]
    if not items:
        raise RuntimeError(f"No CQs for mode '{retrieval_mode}'.")

    # scan SPARQL
    rq_by_id, rq_questions = _scan_rq_files(sparql_root) if sparql_root else ({}, {})

    # outputs
    mode_dir = out_root / retrieval_mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    out_index = mode_dir / "cq_index.jsonl"
    out_meta  = mode_dir / "cq_metadata.json"
    out_order = mode_dir / "order.txt"
    out_ids   = mode_dir / "ids.json"
    out_emb   = mode_dir / "embeddings.npy"
    out_faiss = mode_dir / "faiss.index"

    index_lines=[]; metadata={}; order=[]; ids_for_embed=[]; texts_for_embed=[]
    stats={"direct":0,"file":0,"rq_scan":0,"fallback_named":0,"missing":0,"q_from_rq":0}

    for rec in sorted(items, key=lambda r: r["CQ_ID"]):
        cid = rec["CQ_ID"]
        q   = rec.get("Question","").strip()
        a   = rec.get("Answer","").strip()
        beats = rec.get("Beats") or []
        modes = rec.get("RetrievalMode") or []

        # SPARQL precedence: direct field > file pointer > rq scan > fallback by name
        sparql = (rec.get("SPARQL") or "").strip()
        sparql_src = ""
        if sparql:
            sparql_src="direct"; stats["direct"]+=1
        else:
            spfile = rec.get("SPARQLFile") or ""
            if spfile:
                p = Path(spfile)
                if not p.is_absolute() and sparql_root: p = sparql_root / p
                try:
                    txt = p.read_text(encoding="utf-8").strip()
                except Exception:
                    txt = ""
                if txt:
                    sparql = txt; sparql_src="file"; stats["file"]+=1
            if not sparql and cid in rq_by_id:
                sparql = rq_by_id[cid]; sparql_src="rq_scan"; stats["rq_scan"]+=1
                if override_question and cid in rq_questions and rq_questions[cid]:
                    q = rq_questions[cid]
                    stats["q_from_rq"]+=1
            if not sparql and sparql_root:
                p = sparql_root / f"{cid}.rq"
                if p.exists():
                    try: txt = p.read_text(encoding="utf-8").strip()
                    except Exception: txt = ""
                    if txt:
                        sparql = txt; sparql_src="fallback_named"; stats["fallback_named"]+=1
        if not sparql:
            sparql_src="missing"; stats["missing"]+=1

        # index line & metadata
        embed_txt = _compose_embed_text(embed_text, q, a, beats)
        index_lines.append(json.dumps({"id": cid, "text": f"{cid} :: {q}"}, ensure_ascii=False))
        metadata[cid] = {
            "CQ_ID": cid,
            "question": q,
            "answer": a,
            "beats": beats,
            "retrieval_mode": modes,
            "sparql": sparql,
            "sparql_source": sparql_src,
            "embed_text_recipe": embed_text
        }
        order.append(cid)
        ids_for_embed.append(cid)
        texts_for_embed.append(embed_txt)

    # write core files
    out_index.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    out_meta.write_text(json.dumps({"order": order, "metadata": metadata}, ensure_ascii=False, indent=2), encoding="utf-8")
    out_order.write_text("\n".join(order) + "\n", encoding="utf-8")
    out_ids.write_text(json.dumps(ids_for_embed, ensure_ascii=False, indent=2), encoding="utf-8")

    # embeddings/FAISS
    faiss_msg="skipped"
    if build_faiss:
        try:
            if embedder == "sbert":
                if np is None: raise RuntimeError("numpy is required")
                vecs, dim, _ = _embed_sbert(texts_for_embed, sbert_model, normalize)
                np.save(out_emb, vecs.astype("float32"))
                ok, msg = _build_faiss(vecs, out_faiss, index_type=index_type)
                faiss_msg = msg if ok else f"FAISS not built: {msg}"
            elif embedder == "ollama":
                if np is None: raise RuntimeError("numpy is required")
                host = _ensure_ollama(ollama_host, auto_ollama, ollama_pull, ollama_model)
                vecs, dim, _ = _embed_ollama(texts_for_embed, ollama_model, host, normalize)
                np.save(out_emb, vecs.astype("float32"))
                ok, msg = _build_faiss(vecs, out_faiss, index_type=index_type)
                faiss_msg = msg if ok else f"FAISS not built: {msg}"
            else:
                faiss_msg = "skipped (embedder=none)"
        except Exception as e:
            faiss_msg = f"error during embedding/index: {e}"

    # validation
    val_status="skipped"; out_rep=None
    if validate:
        rep = _validate_metadata(out_meta)
        out_rep = out_meta.with_name("validation_report.json")
        out_rep.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        val_status = "OK" if rep.get("ok") else "ISSUES FOUND"

    # manifest
    manifest_path = _write_manifest(out_meta)

    return {
        "mode": retrieval_mode,
        "count": len(order),
        "out_index": str(out_index),
        "out_meta": str(out_meta),
        "out_order": str(out_order),
        "out_ids": str(out_ids),
        "out_embeddings": str(out_emb),
        "out_faiss": str(out_faiss),
        "faiss": faiss_msg,
        "validation": val_status,
        "validation_report": str(out_rep) if out_rep else "",
        "manifest": str(manifest_path),
        "stats": stats
    }

# ------------- CLI -------------

def main():
    ap = argparse.ArgumentParser(description="All-in-one CQ builder: normalize, map SPARQL, embed/FAISS, validate, manifest.")
    ap.add_argument("--cq_path", required=True, help="Path to source CQ CSV/JSON")
    ap.add_argument("--sparql_root", required=False, default="", help="Folder with .rq templates. For Both, use a root that contains KG/ and Hybrid/ subfolders.")
    ap.add_argument("--retrieval_mode", required=True, choices=["KG","Hybrid","Both"], help="Which mode(s) to build")
    ap.add_argument("--out_root", default="data/indexes", help="Root folder for outputs (mode subfolders will be created)")

    # normalization options
    ap.add_argument("--beats_sep", default=None, help="Explicit separator for Beats in CSV (default: auto-detect '|' or ',')")
    ap.add_argument("--mode_sep", default=None, help="Explicit separator for RetrievalMode in CSV")
    ap.add_argument("--override_question", action="store_true", help="Override Question from .rq header trailing text when present")

    # embeddings/index
    ap.add_argument("--build_faiss", action="store_true", help="Build embeddings + FAISS index")
    ap.add_argument("--embedder", default="sbert", choices=["none","sbert","ollama"])
    ap.add_argument("--sbert_model", default="all-MiniLM-L6-v2")
    ap.add_argument("--ollama_model", default="nomic-embed-text")
    ap.add_argument("--ollama_host", default="http://localhost:11434")
    ap.add_argument("--auto_ollama", action="store_true")
    ap.add_argument("--ollama_pull", action="store_true")
    ap.add_argument("--embed_text", default="question+beats", choices=["question","question+answer","question+beats","qa+beats"])
    ap.add_argument("--normalize", dest="normalize", action="store_true")
    ap.add_argument("--no-normalize", dest="normalize", action="store_false")
    ap.set_defaults(normalize=True)
    ap.add_argument("--index_type", default="flatip", choices=["flatip","ivfflat"])

    # validation
    ap.add_argument("--validate", action="store_true", help="Write validation_report.json")

    args = ap.parse_args()

    src_path = Path(args.cq_path)
    out_root = Path(args.out_root)
    sparql_root = Path(args.sparql_root) if args.sparql_root else None

    # 1) normalize input CQs
    raw = _load_any_cq(src_path)
    norm = _normalize_cqs(raw, beats_sep=args.beats_sep, mode_sep=args.mode_sep)

    # 2) per-mode build(s)
    modes = [args.retrieval_mode] if args.retrieval_mode != "Both" else ["KG","Hybrid"]
    results = []
    for mode in modes:
        mode_root = sparql_root
        if sparql_root and args.retrieval_mode == "Both":
            # Expect subfolders KG/ and Hybrid/ under sparql_root
            mode_root = sparql_root / mode
        info = _build_for_mode(
            norm_items=norm,
            retrieval_mode=mode,
            out_root=out_root,
            sparql_root=mode_root,
            override_question=args.override_question,
            build_faiss=args.build_faiss,
            embedder=args.embedder,
            sbert_model=args.sbert_model,
            ollama_model=args.ollama_model,
            ollama_host=args.ollama_host,
            auto_ollama=args.auto_ollama,
            ollama_pull=args.ollama_pull,
            embed_text=args.embed_text,
            normalize=args.normalize,
            index_type=args.index_type,
            validate=args.validate
        )
        results.append(info)
        # console summary
        s = info["stats"]
        print(f"\n=== {mode} DONE ===")
        print(f"Built {info['count']} items")
        print(f"SPARQL sources: direct={s['direct']}, file={s['file']}, rq_scan={s['rq_scan']}, fallback_named={s['fallback_named']}, missing={s['missing']} (q_from_rq={s['q_from_rq']})")
        print(f"Index:   {info['out_index']}")
        print(f"Meta:    {info['out_meta']}")
        print(f"Order:   {info['out_order']}  |  IDs: {info['out_ids']}")
        print(f"Embeds:  {info['out_embeddings']}")
        print(f"FAISS:   {info['faiss']} -> {info['out_faiss']}")
        print(f"Valid:   {info['validation']} -> {info['validation_report']}")
        print(f"Manifest:{info['manifest']}")

if __name__ == "__main__":
    main()
