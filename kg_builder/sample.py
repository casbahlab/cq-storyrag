import streamlit as st
import uuid

# Sample schema vocab
classes = ["Artist", "Event", "Location", "Song"]
properties = ["performedAt", "heldIn", "wroteSong", "raisedFundsFor"]
objects = ["Wembley", "USA for Africa", "Bohemian Rhapsody", "Ethiopia"]

# Session state init
if "facts_table" not in st.session_state:
    st.session_state["facts_table"] = []

st.title("🎛️ Triple Modeling (Manual Editable Table)")

# Add new fact row
with st.form("add_fact_form", clear_on_submit=True):
    col1, col2, col3 = st.columns(3)
    subj = col1.selectbox("Subject", options=classes)
    pred = col2.selectbox("Predicate", options=properties)
    obj = col3.selectbox("Object", options=objects)
    submitted = st.form_submit_button("➕ Add Triple")

    if submitted:
        st.session_state["facts_table"].append({
            "id": str(uuid.uuid4()),
            "subject": subj,
            "predicate": pred,
            "object": obj
        })

# Display and edit table
for i, fact in enumerate(st.session_state["facts_table"]):
    with st.expander(f"Triple {i+1}", expanded=True):
        col1, col2, col3, col4 = st.columns([3, 3, 3, 1])

        fact["subject"] = col1.text_input("Subject", fact["subject"], key=f"subj_{fact['id']}")
        fact["predicate"] = col2.text_input("Predicate", fact["predicate"], key=f"pred_{fact['id']}")
        fact["object"] = col3.text_input("Object", fact["object"], key=f"obj_{fact['id']}")

        if col4.button("🗑️", key=f"del_{fact['id']}"):
            st.session_state["facts_table"].pop(i)
            st.experimental_rerun()

# Debug
st.markdown("### 📦 Stored Triples:")
st.json(st.session_state["facts_table"])
