#!/usr/bin/env python3
"""
Clean non-song items (intros, interviews, speeches, etc.) from setlists and
recordings/works, or tag them for query-time filtering.

Defaults:
- Detects non-songs using conservative rules (exact/bracketed descriptors and known localnames)
- Removes them from:
  * 33_recordings_works.ttl  (drops those works & their recordings; removes mm:hasRecording to dropped recs)
  * 40_setlists_songs.ttl    (drops ListItems to those works; mm:performedWork to those works; mm:recordedAs to dropped recs)
- Writes *.cleaned.ttl alongside inputs, plus a JSON report

Options:
- --mode tag         : keep nodes but add ex:isNonSong true (no deletions)
- --aggressive       : widen detection (match titles containing intro/interview/outro/etc.)
- --inplace          : overwrite inputs (writes .bak first)
- --report-out FILE  : path for JSON report
- --pattern REGEX    : add extra regex(es) (repeatable)

Usage:
  python clean_non_song_items.py \
      --setlists-in kg/40_setlists_songs.ttl \
      --recworks-in kg/33_recordings_works.ttl

  # Tag instead of delete:
  python clean_non_song_items.py --mode tag

  # Aggressive matching:
  python clean_non_song_items.py --aggressive

  # In-place cleanup:
  python clean_non_song_items.py --inplace
"""
from rdflib import Graph, Namespace, URIRef, RDF, RDFS, Literal, BNode
from rdflib.namespace import OWL
from pathlib import Path
import argparse, re, json, shutil

SCHEMA = Namespace("http://schema.org/")
MM     = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
EX     = Namespace("http://wembrewind.live/ex#")

def load_graph(path: Path) -> Graph:
    g = Graph()
    g.parse(path, format="turtle")
    # bind usual prefixes
    g.bind("schema", SCHEMA); g.bind("mm", MM); g.bind("ex", EX); g.bind("rdfs", RDFS); g.bind("owl", OWL)
    return g

def save_graph(g: Graph, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(path), format="turtle")

def localname(u: URIRef) -> str:
    s = str(u)
    if "#" in s: return s.split("#", 1)[-1]
    return s.rsplit("/", 1)[-1]

def label_of(g: Graph, s: URIRef) -> str:
    for o in g.objects(s, SCHEMA.name):  return str(o)
    for o in g.objects(s, RDFS.label):   return str(o)
    return localname(s)

# Known non-song localname tokens (lowercase)
KNOWN_NON_SONG_NAMES = {
    "introduction","intro","outro","audienceparticipation","audience","applause",
    "interview","mtvinterview","abcinterview","bbcinterview","message","liveaidmessage",
    "liveaidmessage2","liveaidmessage3","introductionfromwembley","introductionspeech",
    "hollandintroduction","rehearsals","rehearsal","tuning","banter","montage","credits",
    "introductingthefaminevideo" # typo in some data
}

# Conservative title patterns: exact/bracketed descriptors like “[introduction]”
TITLE_EXACT = re.compile(
    r"""^\s*\[?\s*(intro(?:duction)?|outro|audience(?:\s+participation)?|applause|interview
        |speech|appeal|presentation|rehearsal|tuning|banter|montage|credits)\s*\]?\s*$""",
    re.I | re.X
)

# Aggressive title patterns: anywhere in the label
TITLE_CONTAINS = re.compile(
    r"""(?i)\b(intro|introduction|outro|audience(?: participation)?|applause|interview
        |speech|appeal|presentation|rehearsal|tuning|banter|montage|credits)\b""",
    re.X
)

def detect_non_song_works(g_set: Graph, g_rw: Graph, aggressive: bool, extra_patterns: list[str]) -> set[URIRef]:
    extra_res = [re.compile(p, re.I) for p in extra_patterns or []]
    works = set(g_rw.subjects(RDF.type, SCHEMA.MusicComposition)) | set(g_set.subjects(RDF.type, SCHEMA.MusicComposition))
    non_song = set()

    for w in works:
        if not isinstance(w, URIRef):
            continue
        ln = localname(w).lower()
        if ln in KNOWN_NON_SONG_NAMES:
            non_song.add(w); continue

        title = label_of(g_rw, w) or label_of(g_set, w)
        if not title: continue

        # Conservative first
        if TITLE_EXACT.search(title):
            non_song.add(w); continue

        # Optional aggressive match
        if aggressive and TITLE_CONTAINS.search(title):
            non_song.add(w); continue

        # User-provided extras
        if any(p.search(title) for p in extra_res):
            non_song.add(w); continue

    return non_song

def tag_non_songs(g_set: Graph, g_rw: Graph, works: set[URIRef], recs: set[URIRef]) -> tuple[Graph, Graph]:
    for w in works:
        g_rw.add((w, EX.isNonSong, Literal(True)))
        # keep type & labels as-is
    for r in recs:
        g_rw.add((r, EX.isNonSong, Literal(True)))
    return g_set, g_rw

def remove_non_songs(g_set: Graph, g_rw: Graph, works: set[URIRef], recs: set[URIRef]) -> tuple[Graph, Graph, int, int]:
    g_rw_out = Graph(); [g_rw_out.bind(pfx, ns) for pfx, ns in g_rw.namespaces()]
    g_set_out = Graph(); [g_set_out.bind(pfx, ns) for pfx, ns in g_set.namespaces()]

    removed_rw = removed_set = 0

    # recordings_works: drop non-song works & their recordings; drop mm:hasRecording to removed recs
    for s, p, o in g_rw:
        if s in works or s in recs:
            removed_rw += 1; continue
        if p == MM.hasRecording and isinstance(o, URIRef) and o in recs:
            removed_rw += 1; continue
        g_rw_out.add((s, p, o))

    # setlists: drop ListItems whose schema:item is non-song work; drop mm:performedWork to those works; drop mm:recordedAs to removed recs
    listitem_bnodes_to_drop = set()
    for li in g_set.subjects(RDF.type, SCHEMA.ListItem):
        if isinstance(li, BNode):
            tgt = next(g_set.objects(li, SCHEMA.item), None)
            if isinstance(tgt, URIRef) and tgt in works:
                listitem_bnodes_to_drop.add(li)

    for s, p, o in g_set:
        if isinstance(s, BNode) and s in listitem_bnodes_to_drop:
            removed_set += 1; continue
        if p == SCHEMA.itemListElement and isinstance(o, BNode) and o in listitem_bnodes_to_drop:
            removed_set += 1; continue
        if p == MM.performedWork and isinstance(o, URIRef) and o in works:
            removed_set += 1; continue
        if p == MM.recordedAs and isinstance(o, URIRef) and o in recs:
            removed_set += 1; continue
        g_set_out.add((s, p, o))

    return g_set_out, g_rw_out, removed_set, removed_rw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setlists-in",  default="kg/40_setlists_songs.ttl")
    ap.add_argument("--recworks-in",  default="kg/33_recordings_works.ttl")
    ap.add_argument("--setlists-out", default=None, help="Output for setlists (default: *.cleaned.ttl or *.tagged.ttl)")
    ap.add_argument("--recworks-out", default=None, help="Output for recordings/works (default: *.cleaned.ttl or *.tagged.ttl)")
    ap.add_argument("--mode", choices=["remove","tag"], default="remove", help="Remove nodes or tag with ex:isNonSong true")
    ap.add_argument("--aggressive", action="store_true", help="Match titles containing intro/interview/outro/etc.")
    ap.add_argument("--pattern", action="append", help="Extra regex pattern(s) to treat as non-song (repeatable)")
    ap.add_argument("--inplace", action="store_true", help="Overwrite inputs (writes .bak first)")
    ap.add_argument("--report-out", default=None, help="Where to write JSON report (default: alongside outputs)")
    args = ap.parse_args()

    set_in  = Path(args.setlists_in)
    rw_in   = Path(args.recworks_in)

    g_set = load_graph(set_in)
    g_rw  = load_graph(rw_in)

    # Detect non-song works
    non_song_works = detect_non_song_works(g_set, g_rw, args.aggressive, args.pattern or [])

    # Recordings tied to those works
    non_song_recs = set()
    for r in g_rw.subjects(RDF.type, MM.Recording):
        if not isinstance(r, URIRef): continue
        if any((w in non_song_works) for w in g_rw.objects(r, SCHEMA.recordingOf)):
            non_song_recs.add(r)

    # Outputs
    suffix = ".tagged.ttl" if args.mode == "tag" else ".cleaned.ttl"
    set_out = Path(args.setlists_out) if args.setlists_out else set_in.with_suffix(set_in.suffix + suffix)
    rw_out  = Path(args.recworks_out) if args.recworks_out else rw_in.with_suffix(rw_in.suffix + suffix)

    removed_set = removed_rw = 0
    if args.mode == "tag":
        g_set_out, g_rw_out = tag_non_songs(g_set, g_rw, non_song_works, non_song_recs)
    else:
        g_set_out, g_rw_out, removed_set, removed_rw = remove_non_songs(g_set, g_rw, non_song_works, non_song_recs)

    # Write (inplace or side-by-side)
    report_path = Path(args.report_out) if args.report_out else set_out.with_suffix(".json")

    if args.inplace:
        # backups
        set_bak = set_in.with_suffix(set_in.suffix + ".bak")
        rw_bak  = rw_in.with_suffix(rw_in.suffix + ".bak")
        shutil.copyfile(set_in, set_bak)
        shutil.copyfile(rw_in, rw_bak)
        save_graph(g_set_out, set_in)
        save_graph(g_rw_out,  rw_in)
        set_out_path, rw_out_path = set_in, rw_in
    else:
        save_graph(g_set_out, set_out)
        save_graph(g_rw_out,  rw_out)
        set_out_path, rw_out_path = set_out, rw_out

    report = {
        "mode": args.mode,
        "aggressive": args.aggressive,
        "extra_patterns": args.pattern or [],
        "non_song_work_count": len(non_song_works),
        "non_song_recording_count": len(non_song_recs),
        "removed_triples_setlists": removed_set,
        "removed_triples_recordings_works": removed_rw,
        "non_song_samples": [
            f"{w} :: {label_of(g_rw, w) or label_of(g_set, w)}"
            for w in list(non_song_works)[:20]
        ],
        "outputs": {
            "setlists": str(set_out_path),
            "recworks": str(rw_out_path),
        }
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[non-song works] {len(non_song_works)}")
    print(f"[non-song recs ] {len(non_song_recs)}")
    if args.mode == "remove":
        print(f"[removed] setlists triples: {removed_set}, rec/works triples: {removed_rw}")
    print(f"[write] setlists → {set_out_path}")
    print(f"[write] rec/works → {rw_out_path}")
    print(f"[report] {report_path}")

if __name__ == "__main__":
    main()
