import json
from collections import defaultdict, Counter

# Load triples
with open("triples_from_gpt.json") as f:
    data = json.load(f)

# Helper: extract URI fragment
def extract_fragment(uri):
    if not isinstance(uri, str):
        return "Literal"
    if uri.startswith("http"):
        return uri.split("/")[-1]
    return "Literal"

# Trace structure
predicate_traces = defaultdict(lambda: {
    "subject_samples": [],
    "object_samples": [],
    "subject_counter": Counter(),
    "object_counter": Counter()
})

# Collect raw fragments
for item in data:
    print("item")
    print(item)
    for triple in item.get("Triples", []):
        pred = triple.get("predicate")
        print("pred")
        print(pred)
        subj = extract_fragment(triple.get("subject"))
        obj = extract_fragment(triple.get("object"))

        predicate_traces[pred]["subject_samples"].append(subj)
        predicate_traces[pred]["object_samples"].append(obj)
        predicate_traces[pred]["subject_counter"][subj] += 1
        predicate_traces[pred]["object_counter"][obj] += 1

# Inference step: naive generalisation by shared suffix or naming conventions
def infer_class(fragments):
    candidates = []
    for frag, count in fragments.most_common():
        if frag.lower().endswith("performance"):
            candidates.append("Performance")
        elif frag.lower().endswith("stadium"):
            candidates.append("Stadium")
        elif frag.lower().endswith("audience"):
            candidates.append("Audience")
        elif frag.lower().endswith("review"):
            candidates.append("Review")
        elif frag.lower() in {"queen", "u2", "davidbowie", "eltonjohn"}:
            candidates.append("MusicArtist")
        elif frag.lower() in {"liveaid"}:
            candidates.append("MusicEvent")
        elif frag.lower() in {"bbc", "abc"}:
            candidates.append("Organization")
        elif frag.lower() in {"literal", "english", "2018"}:
            candidates.append("Literal")
        else:
            candidates.append(frag)  # fallback to fragment
    return list(dict.fromkeys(candidates))  # unique, in order

# Build final mapping
predicate_inference = {}

for pred, trace in predicate_traces.items():
    subj_classes = infer_class(trace["subject_counter"])
    obj_classes = infer_class(trace["object_counter"])
    predicate_inference[pred] = {
        "predicate": pred,
        "subject_fragments": dict(trace["subject_counter"]),
        "object_fragments": dict(trace["object_counter"]),
        "inferred_subject_classes": subj_classes,
        "inferred_object_classes": obj_classes
    }

# Save traceable output
with open("predicate_domain_range_trace.json", "w") as f:
    json.dump(predicate_inference, f, indent=2)

print("Traceable predicate domain/range inference saved to predicate_domain_range_trace.json")
