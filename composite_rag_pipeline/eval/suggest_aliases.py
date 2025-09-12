#!/usr/bin/env python3
import sys, re, json, pandas as pd
from collections import Counter

PROPER_RX = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")

def proper_set(s): return {m.group(1).lower() for m in PROPER_RX.finditer(str(s) or "")}

def main(near_path):
    df = pd.read_csv(near_path)
    cand = Counter()
    for _, r in df.iterrows():
        ps = proper_set(r.get("sentence",""))
        pe = proper_set(r.get("best_evidence",""))
        # asym diff suggests aliasing candidates
        for a in sorted(ps - pe):
            for b in sorted(pe - ps):
                if len(a) >= 3 and len(b) >= 3:
                    cand[(a,b)] += 1
    print("# Suggested alias pairs (left≈sentence variant → right≈evidence variant):")
    out = {}
    for (a,b), n in cand.most_common(80):
        out.setdefault(b, [])
        if a not in out[b]: out[b].append(a)
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python3 suggest_aliases.py path/to/near_misses.csv", file=sys.stderr); sys.exit(1)
    main(sys.argv[1])
