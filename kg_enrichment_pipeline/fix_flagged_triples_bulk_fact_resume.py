#!/usr/bin/env python3
import json
import os

# ===== CONFIG =====
INPUT_FILE = "output/flagged_triples.json"
OUTPUT_FILE = "output/refined_triples_fixed.json"
IGNORED_FILE = "output/ignored_triples.json"
FIX_LOG_FILE = "output/fix_log.json"
PROGRESS_FILE = "output/progress.json"

# Load existing outputs or initialize
refined_triples = json.load(open(OUTPUT_FILE)) if os.path.exists(OUTPUT_FILE) else []
ignored_triples = json.load(open(IGNORED_FILE)) if os.path.exists(IGNORED_FILE) else []
fix_log = json.load(open(FIX_LOG_FILE)) if os.path.exists(FIX_LOG_FILE) else []

# Load flagged triples
with open(INPUT_FILE, "r") as f:
    flagged_triples = json.load(f)

# Load progress
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r") as f:
        progress = json.load(f)
        processed_facts = progress.get("facts_processed", 0)
else:
    processed_facts = 0

print(f"Loaded {len(flagged_triples)} flagged triples (facts). {processed_facts} facts already processed.")

def save_progress():
    with open(OUTPUT_FILE, "w") as f:
        json.dump(refined_triples, f, indent=2)
    with open(IGNORED_FILE, "w") as f:
        json.dump(ignored_triples, f, indent=2)
    with open(FIX_LOG_FILE, "w") as f:
        json.dump(fix_log, f, indent=2)
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"facts_processed": processed_facts}, f, indent=2)

def parse_bulk_triples(input_block):
    """
    Parses a block of triples in multiple formats into JSON triples.
    Supported formats:
    1. subject → predicate → object
    2. subject -> predicate -> object
    3. subject predicate object .  (Turtle style, period optional)
    """
    triples = []
    for line in input_block.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        # Remove trailing period
        if line.endswith("."):
            line = line[:-1].strip()

        # Check for arrows first
        if "→" in line:
            parts = [p.strip() for p in line.split("→")]
        elif "->" in line:
            parts = [p.strip() for p in line.split("->")]
        else:
            # Split by whitespace (Turtle style)
            parts = line.split()
            if len(parts) < 3:
                print(f"⚠ Skipping malformed line: {line}")
                continue
            subj, pred = parts[0], parts[1]
            obj = " ".join(parts[2:])
            parts = [subj, pred, obj]

        if len(parts) != 3:
            print(f"⚠ Skipping malformed line: {line}")
            continue

        subj, pred, obj = parts
        triples.append((subj, pred, obj))
    return triples

# Interactive loop
while processed_facts < len(flagged_triples):
    triple = flagged_triples[processed_facts]
    idx = processed_facts

    canonical = triple.get("source_fact","")
    sources = triple.get("source", [])
    cq_ids = triple.get("cq_ids", [])
    subj = triple.get("subject","")
    pred = triple.get("predicate","")
    obj = triple.get("object","")

    print("\n" + "="*60)
    print(f"Flagged Triple (Fact) #{idx+1}/{len(flagged_triples)}")
    print(f"Fact: {canonical}")
    if sources:
        print(f"Sources: {', '.join(sources)}")
    if cq_ids:
        print(f"CQ IDs: {', '.join(cq_ids)}")
    print(f"Current Triple: subject={subj}, predicate={pred}, object={obj}")

    choice = input("1=✅ Approve as-is, 2=✏ Edit (bulk paste), 3=❌ Reject (Enter=Skip): ").strip()
    old_triple = triple.copy()

    if choice == "1":
        triple["status"] = "approved"
        triple.pop("aliases", None)
        refined_triples.append(triple)

    elif choice == "2":
        print("\nPaste triples (supported formats):")
        print("  subject → predicate → object")
        print("  subject -> predicate -> object")
        print("  subject predicate object .   (Turtle style)")
        print("End with an empty line and press Enter twice:")

        lines = []
        while True:
            line = input()
            if not line.strip():
                break
            lines.append(line)

        bulk_block = "\n".join(lines)
        parsed_triples = parse_bulk_triples(bulk_block)

        if not parsed_triples:
            print("⚠ No valid triples entered. Skipping this fact.")
            continue

        for subj_val, pred_val, obj_val in parsed_triples:
            new_triple = {
                "subject": subj_val,
                "predicate": pred_val,
                "object": obj_val,
                "source_fact": canonical,
                "source": sources,
                "cq_ids": cq_ids,
                "status": "edited"
            }
            refined_triples.append(new_triple)
            fix_log.append({"old": old_triple, "new": new_triple})

    elif choice == "3":
        triple["status"] = "rejected"
        ignored_triples.append(triple)

    else:
        print("Skipped (pending).")
        continue

    # Mark fact processed and save
    processed_facts += 1
    save_progress()

print("\nInteractive universal bulk triple fixing complete!")
print(f"Approved/edited triples saved to {OUTPUT_FILE}")
print(f"Rejected triples saved to {IGNORED_FILE}")
print(f"Progress saved to {PROGRESS_FILE}")
