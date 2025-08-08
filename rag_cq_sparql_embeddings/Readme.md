# Wembley Rewind – Hybrid RAG Pipeline (Quick Guide)

This pipeline generates **persona‑specific narratives** from a **Live Aid 1985 KG** using a **Hybrid RAG** approach:

- **Symbolic**: SPARQL queries over KG  
- **Semantic**: Embeddings + FAISS retrieval  
- **External**: Wikipedia, Wikidata, MusicBrainz digests  

Two personas:  
- **Emma (Curious Novice)** → Light, empathetic story  
- **Luca (Informed Enthusiast)** → Artist & media‑rich narrative  

---

## 1️⃣ Query KG
```bash
python scripts/sparql_to_json.py
```
- Input: `cqs_queries.rq` + KG in GraphDB  
- Output: `cq_results.json`

---

## 2️⃣ Enrich with External Sources
```bash
python scripts/enrich_external_digest.py
```
- Adds Wikipedia, Wikidata, MusicBrainz digests  
- Output: `cq_results_with_enhanced_digests.json`

---

## 3️⃣ Generate Embeddings & FAISS Index
```bash
python scripts/generate_faiss_embeddings.py
```
- Embeds facts with **Ollama: nomic‑embed‑text**  
- Builds FAISS index for semantic retrieval  
- Outputs:
  - `cq_results_with_vectors.json`  
  - `cq_results_faiss.index` + `faiss_metadata.json`

---

## 4️⃣ Generate Narratives

**Option A: Raw Streaming**
```bash
python scripts/generate_semantic_narratives_streamlined.py
```
- Output: `persona_full_narratives_streaming_8.json` + `.md`

**Option B: Cleaned Persona Narratives**
```bash
python scripts/generate_emma_clean_narrative.py
python scripts/generate_luca_clean_narrative.py
```
- Outputs:
  - `persona_emma_cleaned.json` + `.md`  
  - `persona_luca_cleaned.json` + `.md`

---

## 5️⃣ Merge Final Narratives
```bash
python scripts/merge_persona_narratives.py
```
- Output:
  - `cleaned_persona_narratives.json`  
  - `cleaned_persona_narratives.md` (for dissertation appendix)

---

## Key Outputs
- Raw facts → `cq_results.json`  
- Enriched facts → `cq_results_with_enhanced_digests.json`  
- FAISS embeddings → `cq_results_faiss.index`  
- Raw narratives → `persona_full_narratives_streaming_8.json`  
- Final cleaned narratives → `cleaned_persona_narratives.json`

---

This is all you need to **rerun the full pipeline** from KG → Narrative in one sitting.
