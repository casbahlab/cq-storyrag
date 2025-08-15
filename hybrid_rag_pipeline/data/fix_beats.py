#!/usr/bin/env python3
import json, re, sys
from ast import literal_eval
from pathlib import Path

SMART_QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})

DEFAULT_FIELDS = [
    "Beats",
    "Retrieval Mode",
    "RetrievalMode",
    "retrieval_mode",
    "retrievalMode",
    "Retrieval",          # just in case your data used this
]

def _uniq(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def parse_listlike(v):
    """Return list from str | list | tuple | set | None. Cleans quotes/brackets, dedupes."""
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        raw = [str(x) for x in v]
    else:
        s = str(v).strip().translate(SMART_QUOTE_MAP)
        # JSON array?
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                if isinstance(arr, (list, tuple, set)):
                    return _uniq([str(x).strip().strip('"\'' ).strip("[]()") for x in arr if str(x).strip()])
            except Exception:
                pass
        # Python literal list/tuple?
        try:
            lit = literal_eval(s)
            if isinstance(lit, (list, tuple, set)):
                raw = [str(x) for x in lit]
            else:
                raw = [s]
        except Exception:
            # Fallback: split on common delimiters
            s = s.strip("[]()")
            raw = re.split(r"[;,|\n]+", s)
    cleaned = []
    for t in raw:
        t = re.sub(r"\s+", " ", str(t)).strip().strip('"\'' ).strip("[]()")
        if t:
            cleaned.append(t)
    return _uniq(cleaned)

def normalize_object_fields(obj: dict, target_keys_lower: set[str]) -> None:
    """Mutate obj: for any key whose lowercase is in target set, parse to list."""
    for k in list(obj.keys()):
        if k.lower() in target_keys_lower:
            obj[k] = parse_listlike(obj[k])

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_fields.py <in.json> [out.json] [--fields=Beats,\"Retrieval Mode\",tags]")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") \
              else in_path.with_name(in_path.stem + "_fixed.json")

    # parse optional --fields=...
    fields = None
    for arg in sys.argv[2:]:
        if arg.startswith("--fields="):
            csv = arg.split("=", 1)[1]
            fields = [x.strip() for x in re.split(r"[,\s]+", csv) if x.strip()]

    target_keys = fields if fields else DEFAULT_FIELDS
    target_lower = {k.lower() for k in target_keys}

    data = json.loads(in_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                normalize_object_fields(row, target_lower)
    elif isinstance(data, dict):
        # support top-level dict with those keys or nested items list
        if any(k.lower() in target_lower for k in data.keys()):
            normalize_object_fields(data, target_lower)
        if "items" in data and isinstance(data["items"], list):
            for row in data["items"]:
                if isinstance(row, dict):
                    normalize_object_fields(row, target_lower)
    else:
        print("Input JSON should be an array of objects, or an object (optionally with 'items').")

    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Wrote {out_path}  (normalized fields: {', '.join(sorted(target_lower))})")

if __name__ == "__main__":
    main()
