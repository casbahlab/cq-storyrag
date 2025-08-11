#!/usr/bin/env python3
import argparse
from pathlib import Path
from utils_normalize import normalize_schema_prefixes_in_file

def main():
    ap = argparse.ArgumentParser(description="Normalize schema prefixes (schema1->schema, https->http) in a TTL file")
    ap.add_argument("--file", required=True, help="Path to TTL file to normalize (in place)")
    args = ap.parse_args()
    out = normalize_schema_prefixes_in_file(Path(args.file), inplace=True)
    print(f"[normalize] Updated: {out}")

if __name__ == "__main__":
    main()
