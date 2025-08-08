#!/usr/bin/env python3
import json
import re
import os
import ollama

# ===== CONFIG =====
INPUT_FILE = "output/deduplicated_facts.json"
OUTPUT_FILE = "output/refined_triples.json"
FLAGGED_FILE = "output/flagged_triples.json"

OLLAMA_MODEL = "llama3.1"

# Load facts
with open(INPUT_FILE, "r") as f:
    facts = json.load(f)

refined_triples = []
flagged_triples = []

def rule_based_split(fact_text):
    """
    Very simple validator: returns (subject, object, predicate_hint)
    """
    text = fact_text.strip()
    # Basic keyword-based split for subject-object
    lower = text.lower()
    if " at " in lower:
        parts = text.split(" at ", 1)
        subj, obj = parts[0].strip(), parts[1].strip()
        pred_hint = "schema:location"
    elif " in " in lower:
        parts = text.split(" in ", 1)
        subj, obj = parts[0].strip(), parts[1].strip()
        pred_hint = "schema:location"
    elif " on " in lower:
        parts = text.split(" on ", 1)
        subj, obj = parts[0].strip(), parts[1].strip()
        pred_hint = "schema:date"
    elif "perform" in lower or "sang" in lower or "song" in lower:
        subj, obj = text, ""
        pred_hint = "schema:performer"
    else:
        subj, obj, pred_hint = text, "", "ex:relatedFact"
    return subj, obj, pred_hint

def is_suspicious(ollama_triple, rule_subject, rule_object, rule_pred):
    """
    Determine if the Ollama triple is suspicious using simple checks
    """
    subj = ollama_triple.get("subject","").strip()
    obj = ollama_triple.get("object","").strip()
    pred = ollama_triple.get("predicate","").strip()

    # Suspicious if subject is single word and rule-based subject is longer
    if len(subj.split()) == 1 and len(rule_subject.split()) > 1:
        return True

    # Suspicious if object is empty
    if obj == "":
        return True

    # Suspicious if object == subject
    if obj.lower() == subj.lower():
        return True

    # Suspicious if rule_pred indicates location/date but LLaMA picked something else
    if rule_pred.startswith("schema") and pred and not pred.startswith(rule_pred.split(":")[0]):
        return True

    return False

for idx, fact in enumerate(facts):
    canonical = fact["canonical_fact"]
    aliases = fact.get("aliases", [])
    sources = fact.get("source", [])
    cq_ids = fact.get("cq_ids", [])

    print(f"Processing fact {idx+1}/{len(facts)}: {canonical}")

    # 1. Generate triple using Ollama
    prompt = f"""Extract subject, predicate (schema.org or Music Meta if possible), and object as a triple
from the following fact. Return JSON with keys: subject, predicate, object.
Fact: "{canonical}" """
    response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role":"user","content":prompt}])
    content = response["message"]["content"].strip()

    # Try parsing JSON from response
    try:
        triple = json.loads(content)
    except:
        # fallback: simple triple
        triple = {"subject": canonical, "predicate": "ex:relatedFact", "object": ""}

    triple["source_fact"] = canonical
    triple["aliases"] = aliases
    triple["source"] = sources
    triple["cq_ids"] = cq_ids
    triple["method"] = "ollama"

    # 2. Validate using rule-based heuristic
    rule_subject, rule_object, rule_pred = rule_based_split(canonical)
    if is_suspicious(triple, rule_subject, rule_object, rule_pred):
        triple["validator_status"] = "flagged"
        flagged_triples.append(triple)
    else:
        triple["validator_status"] = "pass"

    refined_triples.append(triple)

# Save outputs
with open(OUTPUT_FILE, "w") as f:
    json.dump(refined_triples, f, indent=2)

with open(FLAGGED_FILE, "w") as f:
    json.dump(flagged_triples, f, indent=2)

print(f"Processing complete! {len(refined_triples)} triples saved to {OUTPUT_FILE}")
print(f"{len(flagged_triples)} triples flagged for review -> {FLAGGED_FILE}")
