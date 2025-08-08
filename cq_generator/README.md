# Wembley Rewind: Competency Question Generation Pipeline

This project implements a full pipeline to extract factual knowledge from scenario narratives, generate competency questions (CQs), decompose and deduplicate them.
---

## CQ Generation Pipeline – Run Instructions

### Prerequisites

Ensure you have the following:

- Python 3.9+
- [`ollama`](https://ollama.com/) installed with the `llama3` model available locally
- Virtual environment setup (e.g., using `venv`)
- All dependencies from `requirements.txt` installed
- Input scenario files:
  - `scenarios/emma_scenario.txt`
  - `scenarios/luca_scenario.txt`

---

### Project Structure

```
cq_generator/
├── scenarios/
│   ├── emma_scenario.txt
│   └── luca_scenario.txt
├── output/
├── generate_fact_extraction_prompt.py
├── generate_cqs_from_facts.py
├── combine_cqs_with_ids.py
├── decompose_to_fact_based_cqs.py
├── deduplicate_cleaned_cqs.py
├── README.md
```

---

## Step-by-Step Workflow

### 1. Extract factual statements from scenarios

```bash
python generate_fact_extraction_prompt.py scenarios/emma_scenario.txt
python generate_fact_extraction_prompt.py scenarios/luca_scenario.txt
```

Output:
- `llama_outputs/emma_facts_output.txt`
- `llama_outputs/luca_facts_output.txt`

---

### 2. Generate initial CQs from factual statements

```bash
python generate_cqs_from_facts.py
```

Output:
- `output/emma_generated_CQs_llama_facts.csv`
- `output/luca_generated_CQs_llama_facts.csv`

---

### 3. Combine generated CQ files and assign IDs

```bash
python combine_cqs_with_ids.py
```

Output:
- `output/combined_cqs_with_ids.csv`

---

### 4. Decompose procedural, temporal, and causal CQs into fact-based ones

```bash
python decompose_to_fact_based_cqs.py
```

Output:
- `output/combined_CQs_with_expanded_facts.csv`

---

### 5. Deduplicate and group semantically similar CQs

```bash
python deduplicate_cleaned_cqs.py
```

Output:
- `output/unique_cleaned_CQs.csv`

---

## Notes

- All generated and intermediate files are stored in the `output/` directory.
- Ensure `ollama` is running before executing model-dependent scripts.
- CQs are grouped, deduplicated, and traceable to their original scenario and type.
- Each script prints output file locations after successful execution.

---