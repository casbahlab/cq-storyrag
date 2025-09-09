
#!/usr/bin/env python3
import re, json, argparse, sys, os
from typing import Dict, List, Tuple, Any

# -------- Sentence splitting (Markdown-aware) --------
def split_sentences(text: str) -> List[str]:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # Split on headings and blank lines first
    heading_blank_split = re.split(r"^#+\s.*$|^\s*$", t, flags=re.M)
    sents: List[str] = []
    for blk in heading_blank_split:
        blk = blk.strip()
        if not blk:
            continue
        start = 0
        for m in re.finditer(r"[^.!?]*[.!?]", blk, flags=re.S):
            sents.append(m.group(0).strip())
            start = m.end()
        tail = blk[start:].strip()
        if tail:
            sents.append(tail)
    return [s for s in sents if s]

# -------- Canonicalization helpers --------
def word_boundary_replace(text: str, target: str, replacement: str):
    pattern = re.compile(rf"\b{re.escape(target)}\b")
    new_text, n = pattern.subn(replacement, text)
    return new_text, n

def canonicalize_names_in_text(text: str, display_canon: Dict[str, List[str]]) -> Tuple[str, int]:
    ops = 0
    for canonical, variants in display_canon.items():
        for v in variants:
            if v == canonical: 
                continue
            text, n = word_boundary_replace(text, v, canonical)
            ops += n
    return text, ops

def canonicalize_predicates_in_text(text: str, predicate_canon: Dict[str, List[str]]) -> Tuple[str, int]:
    ops = 0
    for canonical, variants in predicate_canon.items():
        for v in sorted(variants, key=len, reverse=True):
            if v == canonical:
                continue
            if v in text:
                text = text.replace(v, canonical)
                ops += 1
    return text, ops

YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")
NUM_RE  = re.compile(r"\b\d+(?:[.,]\d+)?\b")

def extract_years_nums(text: str):
    years_full = YEAR_RE.findall(text or "")
    nums = NUM_RE.findall(text or "")
    return set(years_full), set(nums)

# ---------- Context loader ----------
def parse_context_line(line: str) -> Dict[str,str]:
    # ': KG: key: "val"; eventName: "Live Aid 1985"; ...' -> dict
    if ": KG:" in line:
        _, after = line.split(": KG:", 1)
    else:
        after = line
    parts = [p.strip() for p in after.split(";") if p.strip()]
    out = {}
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        k = k.strip()
        v = v.strip().strip('"')
        out[k] = v
    return out

def load_context(context_path: str) -> Dict:
    triples: List[Dict] = []
    display_canon: Dict[str, List[str]] = {}
    predicate_canon: Dict[str, List[str]] = {}

    def add_triple(s, p, o):
        if not (s and p and o):
            return
        s = str(s).strip().strip('"')
        p = str(p).strip()
        o = str(o).strip().strip('"')
        triples.append({"subject": s, "predicate": p, "object": o})
        display_canon.setdefault(s, [s])
        display_canon.setdefault(o, [o])
        predicate_canon.setdefault(p, [p])

    text = open(context_path, "r", encoding="utf-8").read()

    # Try JSON parse; if fails, treat as JSONL
    try:
        obj = json.loads(text)
        if isinstance(obj.get("triples"), list):
            for t in obj["triples"]:
                add_triple(t.get("subject"), t.get("predicate"), t.get("object"))
        elif all(k in obj for k in ("subject","predicate","object")):
            add_triple(obj["subject"], obj["predicate"], obj["object"])
        # Optional canon maps
        for name, variants in obj.get("display_canon", {}).items():
            display_canon[name] = variants or [name]
        for pred, variants in obj.get("predicate_canon", {}).items():
            predicate_canon[pred] = variants or [pred]
    except json.JSONDecodeError:
        # JSONL path
        with open(context_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                # Extract from references_by_cq rows
                if "references_by_cq" in rec and isinstance(rec["references_by_cq"], dict):
                    for _, entries in rec["references_by_cq"].items():
                        for e in entries:
                            if e.get("type") == "kg" and isinstance(e.get("row"), dict):
                                row = e["row"]
                                subj = row.get("eventName") or row.get("artistName") or row.get("performanceStartTime") or row.get("musicGroup")
                                if not subj:
                                    continue
                                for k, v in row.items():
                                    if k in ("__cq_id", "eventName", "artistName", "performanceStartTime", "musicGroup"):
                                        continue
                                    add_triple(subj, k, v)
                # Also parse context_lines if present (stringified KG lines)
                if "context_lines" in rec and isinstance(rec["context_lines"], list):
                    for l in rec["context_lines"]:
                        if isinstance(l, str) and "KG:" in l:
                            row = parse_context_line(l)
                            subj = row.get("eventName") or row.get("artistName") or row.get("musicGroup")
                            for k, v in row.items():
                                if k in ("eventName", "artistName", "musicGroup", "__cq_id"):
                                    continue
                                if subj:
                                    add_triple(subj, k, v)

    return {"triples": triples, "display_canon": display_canon, "predicate_canon": predicate_canon}

# ---------- Story canonicalization ----------
def find_target_sentence(sentences: List[str], subj: str, pred: str, obj: str) -> int:
    # 1) subj + pred (order)
    for i, s in enumerate(sentences):
        if subj in s and pred in s and s.find(subj) <= s.find(pred):
            return i
    # 2) subj + object
    for i, s in enumerate(sentences):
        if subj in s and obj in s:
            return i
    # 3) subj only
    for i, s in enumerate(sentences):
        if subj in s:
            return i
    # 4) fallback
    return 0 if sentences else -1

def inject_years_nums(sentences: List[str], triples: List[Dict], display_canon: Dict[str, List[str]]):
    ops = 0
    for t in triples:
        subj = t.get("subject","")
        pred = t.get("predicate","")
        obj  = t.get("object","")
        years, nums = extract_years_nums(obj)
        if not years and not nums:
            continue
        idx = find_target_sentence(sentences, subj, pred, obj)
        if idx == -1:
            continue
        s = sentences[idx]
        changed = False
        for y in years:
            if y not in s:
                s = (s[:-1] + f" in {y}" + s[-1]) if s and s[-1] in ".!?" else (s + f" in {y}")
                ops += 1; changed = True
        for n in [n for n in nums if n not in years]:
            if n not in s:
                s = (s[:-1] + f" ({n})" + s[-1]) if s and s[-1] in ".!?" else (s + f" ({n})")
                ops += 1; changed = True
        if changed:
            sentences[idx] = s
    return sentences, ops

def canonicalize_story(text: str, context: Dict, inject_numbers: bool=True):
    sentences = split_sentences(text)
    name_ops_total = 0
    pred_ops_total = 0
    for i, s in enumerate(sentences):
        s2, nops = canonicalize_names_in_text(s, context["display_canon"])
        name_ops_total += nops
        s3, pops = canonicalize_predicates_in_text(s2, context["predicate_canon"])
        pred_ops_total += pops
        sentences[i] = s3
    inject_ops = 0
    insert_ops = 0
    if True:  # always on; or gate by a CLI flag
        sentences, insert_ops = insert_missing_predicates(sentences, context["triples"], context["predicate_canon"])
    out_text = " ".join(sentences).strip()
    return out_text, {
        "name_replacements": name_ops_total,
        "predicate_replacements": pred_ops_total,
        "year_number_injections": inject_ops,
        "predicate_insertions": insert_ops
    }


# ---------- Answers canonicalization ----------
def canonicalize_field_value(val: Any, display_canon: Dict[str, List[str]]) -> Any:
    if not isinstance(val, str):
        return val
    # Normalize quotes
    v = val.strip().strip('"')
    # Replace aliases with canonical names
    for canonical, variants in display_canon.items():
        for alt in variants:
            if alt == canonical:
                continue
            if v == alt:
                v = canonical
    return v

def process_answers_jsonl(in_path: str, out_path: str, display_canon: Dict[str, List[str]]) -> Dict[str,int]:
    ops = {"name_value_rewrites": 0, "records": 0}
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            changed = False
            # Fix references_by_cq rows
            if isinstance(rec.get("references_by_cq"), dict):
                for _, entries in rec["references_by_cq"].items():
                    for e in entries:
                        if e.get("type") == "kg" and isinstance(e.get("row"), dict):
                            row = e["row"]
                            # normalize all string fields
                            for k, v in list(row.items()):
                                if k == "__cq_id":
                                    continue
                                nv = canonicalize_field_value(v, display_canon)
                                # also strip extra quotes from keys we often see
                                if isinstance(nv, str):
                                    nv2 = nv.strip().strip('"')
                                else:
                                    nv2 = nv
                                if nv2 != v:
                                    row[k] = nv2
                                    changed = True
                                    ops["name_value_rewrites"] += 1
            # Optionally, normalize context_lines too (non-essential for your eval)
            # Write out
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ops["records"] += 1
    return ops

# ---------- CLI ----------
def backup_and_replace(path: str, new_text: str):
    bak_path = path + ".bak"
    if not os.path.exists(bak_path):
        os.rename(path, bak_path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_text)
    return bak_path

def insert_missing_predicates(sentences, triples, predicate_canon):
    ops = 0
    pred_canon_set = set(predicate_canon.keys()) | {v for vs in predicate_canon.values() for v in vs}
    for t in triples:
        s, p, o = t.get("subject",""), t.get("predicate",""), t.get("object","")
        if not (s and p and o):
            continue
        # look for a sentence with subject + object but missing any variant of predicate
        for i, sent in enumerate(sentences):
            if s in sent and o in sent and not any(v in sent for v in ([p] + predicate_canon.get(p, []))):
                # append a clean clause "Subject [predicate] Object."
                suffix = f"{s} {p} {o}"
                if sent.endswith(('.', '!', '?')):
                    sentences[i] = sent[:-1] + f"; {suffix}" + sent[-1]
                else:
                    sentences[i] = sent + f"; {suffix}"
                ops += 1
                break
    return sentences, ops


def main():
    ap = argparse.ArgumentParser(description="Unified canonicalizer for story (text) and answers (JSONL).")
    ap.add_argument("--story", required=True, help="Path to generated story text (Markdown accepted)")
    ap.add_argument("--context", required=True, help="Path to context JSON or JSONL (used to build canon maps)")
    ap.add_argument("--out", help="Path to write canonicalized story (if omitted, overwrite input with .bak backup)")
    ap.add_argument("--answers", help="Path to answers JSONL to canonicalize")
    ap.add_argument("--answers_out", help="Path to write canonicalized answers JSONL (if omitted, overwrite input with .bak backup)")
    ap.add_argument("--report", help="Path to write JSON report of ops")
    ap.add_argument("--no_inject", action="store_true", help="Disable year/number injection into story")
    ap.add_argument("--debug", action="store_true", help="Print summary of loaded triples and canon maps")
    ap.add_argument("--canon", help="Optional JSON with display_canon and predicate_canon to merge")

    args = ap.parse_args()

    if args.answers and not args.answers_out:
        # default answers_out = overwrite
        args.answers_out = args.answers

    # Build canon maps
    ctx = load_context(args.context)
    if args.debug:
        print(f"[debug] triples: {len(ctx['triples'])}", file=sys.stderr)
        print(f"[debug] names: {len(ctx['display_canon'])}, predicates: {len(ctx['predicate_canon'])}", file=sys.stderr)

    if args.canon:
        overrides = json.loads(open(args.canon, "r", encoding="utf-8").read())
        for k, v in overrides.get("display_canon", {}).items():
            ctx["display_canon"][k] = v or [k]
        for k, v in overrides.get("predicate_canon", {}).items():
            ctx["predicate_canon"][k] = v or [k]

    # Canonicalize story
    story_text = open(args.story, "r", encoding="utf-8").read()
    fixed_story, story_ops = canonicalize_story(story_text, ctx, inject_numbers=not args.no_inject)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(fixed_story)
    else:
        bak = backup_and_replace(args.story, fixed_story)
        if args.debug:
            print(f"[debug] backed up original story to {bak}", file=sys.stderr)

    report = {"story_ops": story_ops}

    # Canonicalize answers
    if args.answers:
        if args.answers_out == args.answers:  # in-place
            bak = args.answers + ".bak"
            if not os.path.exists(bak):
                os.rename(args.answers, bak)
            ans_ops = process_answers_jsonl(bak, args.answers, ctx["display_canon"])
            if args.debug:
                print(f"[debug] backed up original answers to {bak}", file=sys.stderr)
        else:
            ans_ops = process_answers_jsonl(args.answers, args.answers_out, ctx["display_canon"])
        report["answers_ops"] = ans_ops

    if args.report:
        open(args.report, "w", encoding="utf-8").write(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
