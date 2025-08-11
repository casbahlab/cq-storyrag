#!/usr/bin/env python3
import re
from pathlib import Path

SCHEMA_HTTP = "http://schema.org/"
SCHEMA_HTTPS = "https://schema.org/"

def normalize_schema_prefixes_in_text(text: str) -> str:
    text = text.replace(SCHEMA_HTTPS, SCHEMA_HTTP)
    text = re.sub(r"(?im)^\s*@?prefix\s+schema1:\s*<https?://schema\.org/>\s*\.\s*$", "", text)
    if "@prefix schema:" not in text and "prefix schema:" not in text:
        lines = text.splitlines()
        inserted = False
        for i, line in enumerate(lines[:20]):
            if line.strip().startswith("@prefix"):
                lines.insert(i, f"@prefix schema: <{SCHEMA_HTTP}> .")
                inserted = True
                break
        if not inserted:
            lines.insert(0, f"@prefix schema: <{SCHEMA_HTTP}> .")
        text = "\n".join(lines)
    text = text.replace("schema1:", "schema:")
    return text

def normalize_schema_prefixes_in_file(ttl_path: Path, inplace: bool = True) -> Path:
    ttl_path = Path(ttl_path)
    txt = ttl_path.read_text(encoding="utf-8")
    norm = normalize_schema_prefixes_in_text(txt)
    if inplace:
        ttl_path.write_text(norm, encoding="utf-8")
        return ttl_path
    out = ttl_path.with_suffix(".normalized.ttl")
    out.write_text(norm, encoding="utf-8")
    return out
