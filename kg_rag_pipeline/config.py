import json
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None


def load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        if p.suffix in {".yaml", ".yml"}:
            if not yaml:
                raise RuntimeError("PyYAML not installed. Run: pip install pyyaml")
            cfg = yaml.safe_load(p.read_text())
        elif p.suffix == ".json":
            cfg = json.loads(p.read_text())
        else:
            raise ValueError("Unsupported config format (use .yaml/.yml/.json)")
    return cfg


def deep_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
