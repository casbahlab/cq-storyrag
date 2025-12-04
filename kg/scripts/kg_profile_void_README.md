# kg_profile_void.py

`kg_profile_void.py` generates a simple [VoID](https://www.w3.org/TR/void/)-style profile for a Knowledge Graph (KG) dataset.

Given an input RDF file (for example, `liveaid_instances_master.ttl`), the script computes basic statistics and writes them out as a separate VoID description, which you can publish alongside your KG or use for sanity checks and documentation.

---

## Features

- Creates a `void:Dataset` description for your KG
- Uses a custom dataset URI provided via CLI
- Computes basic dataset statistics, such as:
  - Total number of triples
  - Number of distinct classes
  - Number of distinct properties
  - Number of distinct subjects / entities (depending on implementation)

> Note: The exact set of statistics depends on the implementation of `kg_profile_void.py`. This README describes the intended behaviour based on the CLI.

---

## Usage

Basic example:

```bash
python kg_profile_void.py \
  --in ../liveaid_instances_master.ttl \
  --out-dir out/profile \
  --dataset-uri http://wembrewind.live/ex/dataset/liveaid
