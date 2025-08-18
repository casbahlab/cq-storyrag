#!/usr/bin/env python3
import requests
from typing import List, Optional, Set, Dict, Any, Union
from urllib.parse import urlsplit, urlunsplit, unquote
from bs4 import BeautifulSoup, Tag, NavigableString

# ---------------- Core utils ----------------

def _bs(html: bytes) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")

def _clean_wiki_bits(soup: Tag) -> None:
    if not soup:
        return
    for x in soup.select(".mw-editsection, .mw-editsection-visualeditor, .mw-empty-elt, sup.reference, sup[role='note']"):
        x.decompose()

def _variants(key: str) -> Set[str]:
    """Lenient variants for id/text matching (spaces, underscores, comma encodings)."""
    if key is None:
        return set()
    k = unquote(key).strip()
    cand = {
        k,
        k.replace("_", " "),
        k.replace(" ", "_"),
        k.lower(),
        k.replace(",", ".2C"),
        k.replace(".2C", ","),
        k.replace("%2C", ","),
        k.replace(",", "%2C"),
    }
    return {c for c in cand if c}

def _text_matches(txt: str, cand: Set[str]) -> bool:
    t = (txt or "").strip()
    return t in cand or t.lower() in cand

def _mw_heading_level(node: Tag) -> Optional[int]:
    if not isinstance(node, Tag):
        return None
    classes = " ".join(node.get("class", []))
    for n in ("1","2","3","4","5","6"):
        if f"mw-heading{n}" in classes:
            return int(n)
    return None

def _is_divider_at_or_above(node: Tag, level: int) -> bool:
    """Is this node a new section at level <= target?"""
    if not isinstance(node, Tag):
        return False
    if node.name in {"h1","h2","h3","h4","h5","h6"}:
        return int(node.name[1]) <= level
    if "mw-heading" in " ".join(node.get("class", [])):
        n = _mw_heading_level(node)
        if n is not None:
            return n <= level
        h = node.find(["h1","h2","h3","h4","h5","h6"])
        if h and int(h.name[1]) <= level:
            return True
    return False

def _section_wrapper_for_h(container: Tag, level: int, key: str) -> Optional[Tag]:
    """Find heading (hN) by id or text; return its mw-heading wrapper if present."""
    cand = _variants(key)

    # <hN id=...>
    for h in container.find_all(f"h{level}"):
        hid = (h.get("id") or "").strip()
        if hid and (hid in cand or hid.lower() in cand):
            p = h.parent
            if isinstance(p, Tag) and "mw-heading" in " ".join(p.get("class", [])):
                return p
            return h

    # <hN><span id=...>
    for h in container.find_all(f"h{level}"):
        span = h.find("span", id=True)
        if span:
            sid = (span.get("id") or "").strip()
            if sid and (sid in cand or sid.lower() in cand):
                p = h.parent
                if isinstance(p, Tag) and "mw-heading" in " ".join(p.get("class", [])):
                    return p
                return h

    # visible text
    for h in container.find_all(f"h{level}"):
        if _text_matches(h.get_text(strip=True), cand):
            p = h.parent
            if isinstance(p, Tag) and "mw-heading" in " ".join(p.get("class", [])):
                return p
            return h

    return None

def _collect_nodes_from_siblings(start_wrapper: Tag, stop_level: int, wanted: Set[str]) -> List[Tag]:
    """Collect nodes of wanted tag names after start_wrapper until next section divider."""
    out: List[Tag] = []
    for sib in start_wrapper.next_siblings:
        if isinstance(sib, Tag) and _is_divider_at_or_above(sib, stop_level):
            break

        if isinstance(sib, NavigableString):
            # turn meaningful stray text into a <p>
            if sib.strip() and "p" in wanted:
                p = BeautifulSoup(f"<p>{sib.strip()}</p>", "html.parser").p
                out.append(p)
            continue

        if not isinstance(sib, Tag):
            continue

        _clean_wiki_bits(sib)

        # If this sibling itself is a wanted node, keep it
        if sib.name in wanted:
            out.append(sib)

        # Also search recursively for nested wanted nodes (e.g., table inside div)
        for node in sib.find_all(list(wanted), recursive=True):
            out.append(node)

    return out

# ---------------- Table parsing ----------------

def _table_to_dict(table: Tag) -> Dict[str, Any]:
    """Convert a <table> to a dict with caption, headers, and rows (text only)."""
    data: Dict[str, Any] = {"caption": "", "headers": [], "rows": []}
    cap = table.find("caption")
    if cap:
        data["caption"] = cap.get_text(" ", strip=True)

    # headers: prefer thead; else first row with ths
    headers: List[str] = []
    thead = table.find("thead")
    if thead:
        tr = thead.find("tr")
        if tr:
            headers = [c.get_text(" ", strip=True) for c in tr.find_all(["th","td"])]
    if not headers:
        tr = table.find("tr")
        if tr:
            ths = tr.find_all("th")
            if ths:
                headers = [th.get_text(" ", strip=True) for th in ths]
            else:
                headers = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
    data["headers"] = headers

    # rows: tbody if present; else all trs after header
    body_rows = []
    tbodies = table.find_all("tbody") or [table]
    for tbody in tbodies:
        for tr in tbody.find_all("tr"):
            cells = tr.find_all(["td","th"])
            if not cells:
                continue
            row = [c.get_text(" ", strip=True) for c in cells]
            # skip exact header row duplicate
            if headers and row == headers:
                continue
            body_rows.append(row)
    data["rows"] = body_rows
    return data

# ---------------- Public APIs ----------------

def content_under_h2(
    url: str,
    h2_key: str,
    content: str = "p",               # "p" or "table"
    parse_tables: bool = False        # if content=="table", parse to dict instead of raw HTML
) -> List[Union[str, Dict[str, Any]]]:
    """
    Collect paragraphs or tables under an H2 section until the next H2/H1.
    - content="p" => returns List[str] (paragraph texts)
    - content="table" => returns List[html strings] or List[dict] if parse_tables=True
    """
    parts = urlsplit(url)
    base = urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))

    r = requests.get(base, headers={"User-Agent": "section-scraper/1.0"}, timeout=20)
    r.raise_for_status()
    soup = _bs(r.content)

    container = soup.select_one("#mw-content-text .mw-parser-output") \
             or soup.select_one("#mw-content-text") \
             or soup

    h2_wrapper = _section_wrapper_for_h(container, 2, h2_key)
    if not h2_wrapper:
        raise ValueError(f"H2 section '{h2_key}' not found on {base}")

    wanted = {"p"} if content == "p" else {"table"}
    nodes = _collect_nodes_from_siblings(h2_wrapper, stop_level=2, wanted=wanted)

    if content == "p":
        return [n.get_text(" ", strip=True) for n in nodes if isinstance(n, Tag)]
    else:
        if parse_tables:
            return [_table_to_dict(n) for n in nodes if isinstance(n, Tag)]
        return [str(n) for n in nodes if isinstance(n, Tag)]

def content_under_h3(
    url: str,
    h2_key: str,
    h3_key: str,
    content: str = "p",               # "p" or "table"
    parse_tables: bool = False
) -> List[Union[str, Dict[str, Any]]]:
    """
    Within a given H2, collect paragraphs or tables under an H3 until next H3/H2/H1.
    """
    parts = urlsplit(url)
    base = urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))

    r = requests.get(base, headers={"User-Agent": "section-scraper/1.0"}, timeout=20)
    r.raise_for_status()
    soup = _bs(r.content)

    container = soup.select_one("#mw-content-text .mw-parser-output") \
             or soup.select_one("#mw-content-text") \
             or soup

    # Find the H2
    h2_wrapper = _section_wrapper_for_h(container, 2, h2_key)
    if not h2_wrapper:
        raise ValueError(f"H2 section '{h2_key}' not found on {base}")

    # Within H2 region, find the H3 wrapper/tag
    h3_wrapper: Optional[Tag] = None
    cand = _variants(h3_key)
    for sib in h2_wrapper.next_siblings:
        if isinstance(sib, Tag) and _is_divider_at_or_above(sib, 2):
            break  # reached next H2
        if not isinstance(sib, Tag):
            continue

        lvl = _mw_heading_level(sib)
        if lvl == 3:
            h = sib.find("h3")
            if h:
                hid = (h.get("id") or "").strip()
                txt = h.get_text(strip=True)
                span = h.find("span", id=True)
                sid = (span.get("id") or "").strip() if span else ""
                if (hid and (hid in cand or hid.lower() in cand)) \
                   or (sid and (sid in cand or sid.lower() in cand)) \
                   or _text_matches(txt, cand):
                    h3_wrapper = sib
                    break
        elif sib.name == "h3":
            hid = (sib.get("id") or "").strip()
            txt = sib.get_text(strip=True)
            span = sib.find("span", id=True)
            sid = (span.get("id") or "").strip() if span else ""
            if (hid and (hid in cand or hid.lower() in cand)) \
               or (sid and (sid in cand or sid.lower() in cand)) \
               or _text_matches(txt, cand):
                h3_wrapper = sib
                break

    if not h3_wrapper:
        raise ValueError(f"H3 '{h3_key}' not found under H2 '{h2_key}' on {base}")

    wanted = {"p"} if content == "p" else {"table"}
    nodes = _collect_nodes_from_siblings(h3_wrapper, stop_level=3, wanted=wanted)

    if content == "p":
        return [n.get_text(" ", strip=True) for n in nodes if isinstance(n, Tag)]
    else:
        if parse_tables:
            return [_table_to_dict(n) for n in nodes if isinstance(n, Tag)]
        return [str(n) for n in nodes if isinstance(n, Tag)]

# ---------------- Example ----------------
if __name__ == "__main__":
    # Examples:
    # 1) All paragraphs under H2 "Background"
    print("\n\n".join(content_under_h2("https://en.wikipedia.org/wiki/Live_Aid#Background", "Background", content="p")))

    # 2) All tables under H2 "Performances" (as parsed dicts)
    #print(content_under_h2("https://en.wikipedia.org/wiki/Live_Aid#Performances", "Performances", content="table", parse_tables=True))

    # 3) Paragraphs under H3 "Wembley, London" within H2 "Performances"
    #print("\n\n".join(content_under_h3("https://en.wikipedia.org/wiki/Live_Aid#Performances", "Performances", "London, Wembley Stadium", content="p")))

    # 4) Tables under H3 "Wembley, London" within H2 "Performances", parsed
    print(content_under_h3("https://en.wikipedia.org/wiki/Live_Aid#Performances", "Performances", "Philadelphia, John F. Kennedy Stadium", content="table", parse_tables=True))
    pass
