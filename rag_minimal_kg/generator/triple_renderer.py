# generator/triple_renderer.py

from typing import List, Tuple

def triples_to_story(triples: List[Tuple[str, str, str]], persona: str = "Emma") -> str:
    predicate_labels = {
        "schema:name": "is called",
        "schema:performer": "was performed by",
        "schema:startDate": "started on",
        "schema:location": "was held at",
        "schema:subEvent": "included segment",
        "schema:superEvent": "was part of",
        "ex:performedSong": "included the song",
        "ex:segmentOf": "was part of",
        "schema:sameAs": "can also be found at",
        "schema:url": "is linked here",
    }

    lines = []
    for s, p, o in triples:
        p_label = predicate_labels.get(p, p.split("/")[-1])
        o_text = o.split("/")[-1] if "http" in o else o
        s_text = s.split("/")[-1] if "http" in s else s
        sentence = f"'{s_text}' {p_label} '{o_text}'."
        lines.append(sentence)

    story = "\n".join(lines)

    if persona.lower() == "emma":
        return f"Here's a simple overview:\n{story}"
    elif persona.lower() == "luca":
        return f"The following structured information has been retrieved:\n{story}"
    else:
        return story
