from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import copy, yaml

# --- Dataclasses (nice for IDEs; we still pass dicts around) ---

@dataclass
class RetrieverCfg:
    seed_strategy: List[str] = field(default_factory=lambda: ["cq_entities", "sparql_seed"])
    k_hops: int = 2
    max_nodes: int = 300
    community: str = "label_propagation"
    edge_types_include: Optional[List[str]] = field(default_factory=lambda: ["performedAt","performedSong","influencedBy"])
    edge_types_exclude: Optional[List[str]] = field(default_factory=list)
    summarise_subgraph: bool = True

@dataclass
class GenerationCfg:
    max_context_chars: int = 6000
    max_triples: int = 350
    max_facts: int = 200
    citation_style: str = "cqid"
    enforce_citation_each_sentence: bool = True

@dataclass
class GraphCfg:
    retrieval: RetrieverCfg = field(default_factory=RetrieverCfg)
    generation: GenerationCfg = field(default_factory=GenerationCfg)

_DEFAULT = asdict(GraphCfg())

def _deep_merge(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_graph_config(path: Path | str | None, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load YAML (if exists), apply overrides, and return a resolved dict."""
    cfg = copy.deepcopy(_DEFAULT)
    if path:
        p = Path(path)
        if p.exists():
            file_cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            cfg = _deep_merge(cfg, file_cfg)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg
