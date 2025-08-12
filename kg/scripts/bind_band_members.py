#!/usr/bin/env python3
import time, random, re
from pathlib import Path
from typing import Optional, Tuple, List, Dict

from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD

# Namespaces
SCHEMA = Namespace("http://schema.org/")
MM = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
EX_BASE = "http://wembrewind.live/ex#"
EX = Namespace(EX_BASE)

# Files
KG_DIR = Path("kg")
ARTISTS_TTL = KG_DIR / "20_artists.ttl"
BAND_MEMBERS_TTL = KG_DIR / "22_band_members.ttl"
INSTR_TTL = KG_DIR / "50_instruments.ttl"

APP_NAME = "WembleyRewind"
APP_VER = "0.1.0"
CONTACT = "mailto:you@example.com"  # <-- set your contact

# -------- config for attributes/instruments --------
ORIGINAL_TOKENS = {"original", "founder", "founding"}

INSTRUMENT_KEYWORDS = {
    "vocal", "vocals", "lead vocals", "background vocals", "backing vocals",
    "guitar", "rhythm guitar", "lead guitar",
    "bass", "bass guitar",
    "drums", "drum", "percussion",
    "keyboard", "synth", "synthesizer", "piano",
    "sax", "saxophone", "trumpet", "violin", "cello", "harmonica"
}

# canonical local names for common instruments; extend as needed
INSTRUMENT_CANON = {
    "vocals": "Voice",
    "lead vocals": "Voice",
    "background vocals": "Voice",
    "backing vocals": "Voice",
    "guitar": "Guitar",
    "rhythm guitar": "Guitar",
    "lead guitar": "Guitar",
    "bass": "BassGuitar",
    "bass guitar": "BassGuitar",
    "drums": "DrumKit",
    "percussion": "Percussion",
    "keyboard": "Keyboard",
    "piano": "Piano",
    "synth": "Synthesizer",
    "synthesizer": "Synthesizer",
    "sax": "Saxophone",
    "saxophone": "Saxophone",
    "trumpet": "Trumpet",
    "violin": "Violin",
    "cello": "Cello",
    "harmonica": "Harmonica",
}

EX_PLAYS_INSTR = URIRef(EX_BASE + "playsInstrument")
EX_IS_ORIGINAL = URIRef(EX_BASE + "isOriginalMember")

# ---------- helpers ----------
def pascal_slug(text: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", (text or "").strip())
    slug = "".join(p.capitalize() for p in parts if p)
    if not slug:
        slug = "Entity"
    if slug[0].isdigit():
        slug = "A" + slug
    return slug

def date_literal(s: Optional[str]) -> Optional[Literal]:
    """Return a typed Literal for date-like strings (YYYY, YYYY-MM, YYYY-MM-DD)."""
    if not s:
        return None
    s = s.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return Literal(s, datatype=XSD.date)
    if re.fullmatch(r"\d{4}-\d{2}", s):
        return Literal(s, datatype=XSD.gYearMonth)
    if re.fullmatch(r"\d{4}", s):
        return Literal(s, datatype=XSD.gYear)
    return Literal(s)  # fallback

def iri(s: str) -> URIRef:
    return URIRef(EX_BASE + s[3:]) if s.startswith("ex:") else URIRef(s)

def load_graph(p: Path) -> Graph:
    g = Graph()
    if p.exists():
        g.parse(p, format="turtle")
    # normalize https->http schema.org just in case
    to_del, to_add = [], []
    for s, p_, o in g:
        s2, p2, o2 = s, p_, o
        if isinstance(p_, URIRef) and str(p_).startswith("https://schema.org/"):
            p2 = URIRef(str(p_).replace("https://", "http://"))
        if isinstance(o, URIRef) and str(o).startswith("https://schema.org/"):
            o2 = URIRef(str(o).replace("https://", "http://"))
        if (s2, p2, o2) != (s, p_, o):
            to_del.append((s, p_, o)); to_add.append((s2, p2, o2))
    for t in to_del: g.remove(t)
    for t in to_add: g.add(t)
    g.bind("ex", EX); g.bind("schema", SCHEMA); g.bind("mm", MM); g.bind("owl", OWL); g.bind("xsd", XSD)
    return g

def save_graph(g: Graph, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(p), format="turtle")
    print(f"[write] {p} (triples: {len(g)})")

def extract_artist_mbids(g: Graph) -> List[Tuple[URIRef, str]]:
    """Subjects that have ex:mbid literal OR owl:sameAs MusicBrainz URL."""
    pairs: List[Tuple[URIRef, str]] = []
    ex_mbid = URIRef(EX_BASE + "mbid")
    for s, _, v in g.triples((None, ex_mbid, None)):
        if isinstance(v, Literal):
            pairs.append((s, str(v)))
    for s, _, u in g.triples((None, OWL.sameAs, None)):
        if isinstance(u, URIRef) and "musicbrainz.org/artist/" in str(u):
            mbid = str(u).rstrip("/").rsplit("/", 1)[-1]
            if (s, ex_mbid, None) not in g:
                pairs.append((s, mbid))
    # dedup by subject
    seen: Dict[URIRef, str] = {}
    for s, mbid in pairs:
        seen.setdefault(s, mbid)
    return list(seen.items())

def is_group(g: Graph, subj: URIRef) -> bool:
    return (subj, RDF.type, SCHEMA.MusicGroup) in g

# ---------- node reuse / minting ----------
PERSON_TYPES_DEFAULT = [SCHEMA.Person, SCHEMA.PropertyValue, MM.Musician]
GROUP_TYPES_DEFAULT  = [SCHEMA.MusicGroup, SCHEMA.PropertyValue, MM.MusicArtist]  # adjust if you prefer

def infer_type_template(g: Graph, for_person: bool) -> List[URIRef]:
    """Try to copy the dominant type pattern from existing artists, else fall back."""
    candidates = []
    if for_person:
        for s, _, _ in g.triples((None, RDF.type, SCHEMA.Person)):
            candidates.append(s)
    else:
        for s, _, _ in g.triples((None, RDF.type, SCHEMA.MusicGroup)):
            candidates.append(s)
    if candidates:
        s0 = candidates[0]
        types = [o for _, _, o in g.triples((s0, RDF.type, None))]
        return types
    return PERSON_TYPES_DEFAULT if for_person else GROUP_TYPES_DEFAULT

def ensure_artist_node(artists_g: Graph, name: str, mbid: str, is_person: bool) -> URIRef:
    ex_mbid = URIRef(EX_BASE + "mbid")
    # Try by mbid literal
    for s, _, _ in artists_g.triples((None, ex_mbid, Literal(mbid))):
        return s
    # Try by owl:sameAs
    mb_url = URIRef(f"https://musicbrainz.org/artist/{mbid}")
    for s, _, _ in artists_g.triples((None, OWL.sameAs, mb_url)):
        return s

    # Mint new node following your style (ex:Sting, etc.)
    base = pascal_slug(name)
    node = URIRef(EX_BASE + base)
    # avoid collision
    i = 2
    while any(True for _ in artists_g.triples((node, None, None))):
        node = URIRef(EX_BASE + f"{base}{i}")
        i += 1

    # Types: copy template from existing similar nodes if possible; else your defaults
    types = infer_type_template(artists_g, is_person)
    for t in types:
        artists_g.add((node, RDF.type, t))

    # Basic annotations (align with your artist file)
    artists_g.add((node, RDFS.label, Literal(name)))
    artists_g.add((node, ex_mbid, Literal(mbid)))
    artists_g.add((node, OWL.sameAs, mb_url))

    return node

# ---------- instruments (lazy under ex:, stored ONLY in 50_instruments.ttl) ----------
def looks_instrument(term: str) -> bool:
    t = term.strip().lower()
    return any(k in t for k in INSTRUMENT_KEYWORDS)

def canon_instrument(term: str) -> Tuple[str, str]:
    """Return (local_name, display_label) for an instrument term."""
    t = term.strip().lower()
    base = (t.replace("lead ", "")
              .replace("background ", "")
              .replace("backing ", "")
              .strip())
    local = INSTRUMENT_CANON.get(base, None) or pascal_slug(base or t)
    label = (INSTRUMENT_CANON.get(base) or base).replace("-", " ").title()
    return local, label

def ensure_instrument_node(instruments_g: Graph, raw_term: str) -> Tuple[URIRef, str]:
    local, label = canon_instrument(raw_term)
    node = EX[local]
    if (node, None, None) not in instruments_g:
        # Tag as ex:MusicInstrument (custom class) and store ONLY in instruments_g
        instruments_g.add((node, RDF.type, EX.MusicInstrument))
        instruments_g.add((node, RDFS.label, Literal(label)))
        instruments_g.add((node, SCHEMA.name, Literal(label)))
    return node, label

# ---------- membership emitter ----------
def add_membership(out_g: Graph, artists_g: Graph, instruments_g: Graph,
                   band: URIRef, member: URIRef,
                   role_name: Optional[str], begin: Optional[str], end: Optional[str],
                   is_original: Optional[bool], inst_terms: List[str]):
    mem = BNode()
    out_g.add((mem, RDF.type, MM.MusicEnsembleMembership))
    out_g.add((mem, RDF.type, SCHEMA.Role))
    out_g.add((mem, MM.involvesMusicEnsemble, band))
    out_g.add((mem, MM.involvesMemberOfMusicEnsemble, member))

    if role_name:
        out_g.add((mem, SCHEMA.roleName, Literal(role_name)))

    lit = date_literal(begin)
    if lit:
        out_g.add((mem, SCHEMA.startDate, lit))
    lit = date_literal(end)
    if lit:
        out_g.add((mem, SCHEMA.endDate, lit))

    if is_original is not None:
        out_g.add((mem, EX_IS_ORIGINAL, Literal(bool(is_original), datatype=XSD.boolean)))

    # Instruments: link on membership; instrument nodes live in instruments_g
    for term in sorted(set(t for t in inst_terms if t)):
        inst_iri, _ = ensure_instrument_node(instruments_g, term)
        out_g.add((mem, SCHEMA.instrument, inst_iri))
        # artist-level capability
        if (member, EX_PLAYS_INSTR, inst_iri) not in artists_g:
            artists_g.add((member, EX_PLAYS_INSTR, inst_iri))

    # Convenience link
    out_g.add((band, SCHEMA.member, member))

# ---------- MB client ----------
def mb_client():
    import musicbrainzngs as mb
    mb.set_useragent(APP_NAME, APP_VER, CONTACT)
    mb.set_hostname("musicbrainz.org")
    mb.set_rate_limit(True)
    return mb

def mb_call(fn, *args, **kwargs):
    import musicbrainzngs as mbx
    for attempt in range(6):
        try:
            return fn(*args, **kwargs)
        except mbx.NetworkError:
            time.sleep(min(30, 2 ** attempt + random.random()))
    raise RuntimeError("MusicBrainz kept failing")

# ---------- main ----------
def main():
    artists_g = load_graph(ARTISTS_TTL)
    out_g = load_graph(BAND_MEMBERS_TTL)
    instruments_g = load_graph(INSTR_TTL)

    pairs = extract_artist_mbids(artists_g)
    groups = [(s, mbid) for (s, mbid) in pairs if is_group(artists_g, s)]
    print(f"[scan] groups found: {len(groups)}")

    if not groups:
        print("[done] no schema:MusicGroup with MBIDs in artists file.")
        # still write instruments_g/out_g in case of small edits
        save_graph(instruments_g, INSTR_TTL)
        save_graph(out_g, BAND_MEMBERS_TTL)
        return

    mb = mb_client()

    for band_iri, band_mbid in groups:
        data = mb_call(mb.get_artist_by_id, band_mbid, includes=["artist-rels"])
        rels = data.get("artist", {}).get("artist-relation-list", []) or []
        count = 0
        for rel in rels:
            rtype = (rel.get("type") or "").lower()
            if rtype not in {"member of band", "band member"}:
                continue

            a = rel.get("artist") or {}
            member_mbid = a.get("id")
            member_name = a.get("name") or "Unknown"
            member_kind = (a.get("type") or "Person").lower()  # "person" or "group"
            if not member_mbid:
                continue

            # Reuse or mint the member node in artists graph
            member_iri = ensure_artist_node(
                artists_g,
                name=member_name,
                mbid=member_mbid,
                is_person=(member_kind == "person")
            )

            # Attributes: original flag + instruments; role_name keeps all non-"original" terms
            attrs = rel.get("attribute-list") or []
            is_original = any(a.strip().lower() in ORIGINAL_TOKENS for a in attrs)
            inst_terms = [a for a in attrs if looks_instrument(a)]
            role_terms = [a for a in attrs if a.strip().lower() not in ORIGINAL_TOKENS]
            role_name = ", ".join(role_terms) if role_terms else None

            begin = rel.get("begin")
            end = rel.get("end")

            add_membership(out_g, artists_g, instruments_g,
                           band_iri, member_iri,
                           role_name, begin, end,
                           is_original=is_original, inst_terms=inst_terms)
            count += 1

        print(f"[band] {band_iri.split('#')[-1]} members added: {count}")

    # Save updates
    save_graph(artists_g, ARTISTS_TTL)          # artists + ex:playsInstrument links
    save_graph(out_g, BAND_MEMBERS_TTL)         # memberships (references instruments)
    save_graph(instruments_g, INSTR_TTL)        # all instrument nodes under ex:
    print("[done] membership export complete.")

if __name__ == "__main__":
    main()
