# # suggestor_app.py
# import streamlit as st
# import json
#
# # Load your ontology JSON (same as your modeling app)
# with open("ontology.json") as f:
#     ontology = json.load(f)
#
# st.title("🔎 Semantic Triple Suggestor")
# st.markdown("Suggest semantic triple mappings (Subject → Predicate → Object) for a fact using Music Meta and schema.org.")
#
# fact = st.text_area("📝 Enter your fact sentence:")
#
# if st.button("💡 Suggest Mapping"):
#     with st.spinner("Querying model and generating suggestions..."):
#
#         # Build label dictionaries for grounding
#         class_labels = ontology["class_labels"]
#         property_labels = ontology["property_labels"]
#
#         # Build your grounding prompt
#         prompt = f"""
# You are a semantic web expert.
#
# Given this fact: "{fact}"
#
# Suggest the most appropriate semantic triple mapping using **Music Meta Ontology** and **schema.org**.
#
# Use only existing class and property labels. Choose from:
#
# - Subject Classes: {', '.join(class_labels[:15])}...
# - Properties: {', '.join(property_labels[:15])}...
# - Object Classes: {', '.join(class_labels[:15])}...
#
# Output this as JSON:
#
# {{
#   "subject_label": "...",
#   "predicate_label": "...",
#   "object_label": "...",
#   "rationale": "Explain your reasoning for these choices."
# }}
# """
#
#             model="gpt-4",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.4
#         )
#         try:
#             suggestion = json.loads(response.choices[0].message["content"])
#             st.success("✅ Suggestion Ready")
#             st.markdown(f"**Subject**: `{suggestion['subject_label']}`")
#             st.markdown(f"**Predicate**: `{suggestion['predicate_label']}`")
#             st.markdown(f"**Object**: `{suggestion['object_label']}`")
#             st.markdown("**Rationale:**")
#             st.code(suggestion['rationale'])
#         except Exception as e:
#             st.error("❌ Could not parse model response. Raw output:")
#             st.code(response.choices[0].message["content"])


import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-flash")


def suggest_semantic_triples(fact: str) -> str:
    prompt = f"""
You are a semantic web expert tasked with converting natural language facts into RDF-style triples.

Each triple must:
- Follow subject, predicate, object structure.
- Use class and property labels from the Music Meta Ontology (prefix: mm:) or schema.org (prefix: schema:).
- Return 1–3 good suggestions.
- Clearly label all components using the format below.

Input fact:
"{fact}"

Return suggestions in this format:

1. Subject: mm:Something  
   Predicate: schema:someProperty  
   Object: mm:SomethingElse

Use only real terms from the ontologies.
    """

    response = model.generate_content(prompt)
    return response.text


# fact = "The concert was broadcast live via satellite uplinks."
# print(suggest_semantic_triples(fact))


import streamlit as st

st.title("🔎 Semantic Triple Suggester (Gemini)")

fact_input = st.text_area("Enter a fact", placeholder="e.g., Queen performed at Wembley Stadium in 1985.")

if st.button("Suggest Triples"):
    with st.spinner("Thinking..."):
        suggestions = suggest_semantic_triples(fact_input)
    st.code(suggestions, language="markdown")