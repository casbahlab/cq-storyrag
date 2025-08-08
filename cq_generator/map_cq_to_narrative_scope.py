import csv

def parse_fact_file(filepath):
    records = []
    current_scope = None
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith(":"):
                current_scope = line[:-1].strip()
            elif current_scope:
                records.append({"Fact": line, "Narrative_Scope": current_scope})
    return records

def main():
    emma_facts = parse_fact_file("llama_outputs/emma_facts_output.txt")
    luca_facts = parse_fact_file("llama_outputs/luca_facts_output.txt")

    all_facts = emma_facts + luca_facts

    output_file = "output/categorized_facts.csv"
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Fact", "Narrative_Scope"])
        writer.writeheader()
        writer.writerows(all_facts)

    print(f"Categorized facts saved to: {output_file}")

if __name__ == "__main__":
    main()
