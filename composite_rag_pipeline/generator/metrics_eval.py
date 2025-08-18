#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

CITE_RX = re.compile(r"\[(\d+)\]")
NUM_RX  = re.compile(r"\b\d[\d,\.]*\b")
YEAR_RX = re.compile(r"\b(19|20)\d{2}\b")

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            out.append(json.loads(line))
    return out

def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
    return [p.strip() for p in parts if p.strip()]

def evidence_key(ev: Dict[str, Any]) -> str:
    t = ev.get("type") or ("web" if "url" in ev else "kg")
    if t == "web":
        u = (ev.get("url") or "").strip()
        if u:
            return f"web|{u}"
        return f"web|{(ev.get('domain') or '').strip()}|{(ev.get('title') or '').strip()}"
    src = json.dumps({
        "row": ev.get("row", {}),
        "executed_query": ev.get("executed_query","")
    }, sort_keys=True)
    h = hashlib.sha1(src.encode("utf-8")).hexdigest()
    return f"kg|{h}"

def text_stats(text: str) -> Dict[str, float]:
    words = re.findall(r"\b\w+\b", text)
    n_words = max(1, len(words))
    nums = len(NUM_RX.findall(text))
    years = len(YEAR_RX.findall(text))
    return {
        "words": len(words),
        "numbers_per_100w": 100.0 * nums / n_words,
        "years_per_100w": 100.0 * years / n_words,
    }

def compute_metrics(story_path: Path, claims_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    story = story_path.read_text(encoding="utf-8") if story_path.exists() else ""
    claims = read_jsonl(claims_path)

    sents = split_sentences(story)
    cited = [s for s in sents if CITE_RX.search(s)]
    coverage = len(cited) / max(1, len(sents))

    by_beat = defaultdict(int)
    for c in claims:
        key = f"{c.get('beat_index')}|{c.get('beat_title')}"
        by_beat[key] += 1

    ref_counter = Counter()
    web_count = 0
    kg_count = 0
    domains = Counter()

    for c in claims:
        for ev in c.get("evidence", []):
            ref_counter[evidence_key(ev)] += 1
            t = ev.get("type") or ("web" if "url" in ev else "kg")
            if t == "web":
                web_count += 1
                d = (ev.get("domain") or "").strip().lower()
                if d: domains[d] += 1
            else:
                kg_count += 1

    total_ev = sum(ref_counter.values())
    uniq_ev = len(ref_counter)
    most_common = ref_counter.most_common(3)
    top3_share = (sum(n for _, n in most_common) / total_ev) if total_ev else 0.0

    stats = text_stats(story)

    summary = {
        "story_path": str(story_path),
        "claims_path": str(claims_path),
        "sentences_total": len(sents),
        "sentences_cited": len(cited),
        "coverage_overall": coverage,
        "claims_total": len(claims),
        "evidence_total": total_ev,
        "evidence_unique": uniq_ev,
        "evidence_diversity": (uniq_ev / total_ev) if total_ev else 0.0,
        "evidence_web": web_count,
        "evidence_kg": kg_count,
        "top_domains": domains.most_common(10),
        "top3_ref_share": top3_share,
        "numbers_per_100w": stats["numbers_per_100w"],
        "years_per_100w": stats["years_per_100w"],
        "words": stats["words"],
        "claims_per_beat": [{"beat": k, "count": v} for k, v in sorted(by_beat.items())],
    }

    rows = []
    for c in claims:
        rows.append({
            "beat_index": c.get("beat_index"),
            "beat_title": c.get("beat_title"),
            "mode": c.get("mode"),
            "cq_id": c.get("cq_id"),
            "sentence": c.get("sentence"),
            "citations": ";".join(str(n) for n in c.get("citations", [])),
            "evidence_count": len(c.get("evidence", [])),
        })

    return summary, rows

def main():
    ap = argparse.ArgumentParser(description="Compute evaluation metrics from story + claims.jsonl")
    ap.add_argument("--story", required=True)
    ap.add_argument("--claims", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    summary, rows = compute_metrics(Path(args.story), Path(args.claims))

    Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else \
                     ["beat_index","beat_title","mode","cq_id","sentence","citations","evidence_count"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"metrics → {args.out_json} ; rows → {args.out_csv}")

if __name__ == "__main__":
    main()
