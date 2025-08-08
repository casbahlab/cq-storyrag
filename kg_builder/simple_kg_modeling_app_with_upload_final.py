import streamlit as st
import json
from collections import defaultdict

# -------------------- Ontology Loader --------------------


import rdflib
from collections import defaultdict
import pickle

def extract_class_hierarchy(ontology_graph):
    """
    Extract class hierarchy as {child_class_uri: set(parent_class_uris)}
    """
    hierarchy = defaultdict(set)
    for s, p, o in ontology_graph.triples((None, rdflib.RDFS.subClassOf, None)):
        if isinstance(s, rdflib.URIRef) and isinstance(o, rdflib.URIRef):
            hierarchy[str(s)].add(str(o))
    return hierarchy


def build_inherited_subject_predicate_map(subject_predicate_map, class_hierarchy):
    memo = {}

    def get_all_properties(class_uri, visited=None):
        if class_uri in memo:
            return memo[class_uri]

        visited = visited or set()
        if class_uri in visited:
            return set()  # Prevent cycles

        visited.add(class_uri)
        own_props = subject_predicate_map.get(class_uri, set())
        inherited_props = set()

        for parent_uri in class_hierarchy.get(class_uri, []):
            inherited_props |= get_all_properties(parent_uri, visited)

        total_props = own_props | inherited_props
        memo[class_uri] = total_props
        return total_props

    # Expand full map
    return {class_uri: get_all_properties(class_uri) for class_uri in subject_predicate_map}


@st.cache_data
def load_ontology():
    from ontology_loader import load_music_meta_ontology, load_schema_org_ontology, merge_ontologies

    mm = load_music_meta_ontology()
    schema = load_schema_org_ontology()
    combined_ontology = merge_ontologies(mm, schema)

    classes = combined_ontology["classes"]
    properties = combined_ontology["properties"]

    class_labels = sorted({label for label in classes.values() if isinstance(label, str)})
    property_labels = sorted({
        label
        for value in properties.values()
        for label in (
            [value] if isinstance(value, str)
            else value.get("label", []) if isinstance(value.get("label", []), list)
            else [value.get("label")] if isinstance(value.get("label"), str)
            else []
        )
    })



    with open("ontology_maps_prefixed_keys_full.pkl", "rb") as f:
        maps = pickle.load(f)

    subject_predicate_map = maps["subject_predicate_map"]
    predicate_object_map = maps["predicate_object_map"]

    # 🔁 Build Label → URI maps for dropdown usage
    label_to_class = {
        label: label for uri, label in classes.items()
        if isinstance(label, str)
    }

    label_to_property = {}
    for uri, prop in properties.items():
        if isinstance(prop, str):
            label_to_property[prop] = uri
        elif isinstance(prop, dict):
            if isinstance(prop.get("label"), list):
                for lbl in prop["label"]:
                    label_to_property[lbl] = uri
            elif isinstance(prop.get("label"), str):
                label_to_property[prop["label"]] = uri

    return {
        "classes": classes,
        "properties": properties,
        "class_labels": class_labels,
        "property_labels": property_labels,
        "subject_predicate_map": subject_predicate_map,
        "predicate_object_map": predicate_object_map,
        "label_to_class": label_to_class,
        "label_to_property": label_to_property,
    }

def encode_cq_id(cq_id: str) -> int:
    """
    Encodes CQ IDs like 'CQ-E1', 'CQ-E3b', or 'CQ-L4a' into traceable unique integers.
    Letter → 1-based A-Z index × 1000
    Number → numeric part
    Suffix → optional a/b/c mapped to 1/2/3 and added as ones digit
    """
    try:
        _, letter_num = cq_id.split("-")  # e.g., 'E3b'
        letter = letter_num[0].upper()

        # Extract numeric part and optional suffix
        digits = ''.join(filter(str.isdigit, letter_num[1:]))
        suffix = ''.join(filter(str.isalpha, letter_num[1:])).lower()

        base = (ord(letter) - ord("A") + 1) * 1000
        number = int(digits) * 10
        suffix_val = ord(suffix) - ord('a') + 1 if suffix else 0

        return base + number + suffix_val
    except Exception as e:
        print(f"[CQ-ID Conversion Error] {cq_id} → {e}")
        return -1

# -------------------- File Loader --------------------

def load_cq_data(file):
    try:
        return json.load(file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return []

# -------------------- Triple UI --------------------

def render_cq_editor(ontology, cq_item):
    cq_id = cq_item["CQ_ID"]
    st.markdown(f"### {cq_id}")
    st.markdown(f"**Question**: {cq_item['CQ_Text']}")
    st.markdown(f"**Answer**: {cq_item['Answer']}")

    classes = ontology["classes"]

    encoded_cq_id = encode_cq_id(cq_id)

    current = {
            "fact": cq_item["Answer"],
            "subject": "",
            "predicate": "",
            "object": ""
        }


    if "facts" not in st.session_state:
        st.session_state["facts"] = {}

    if cq_id not in st.session_state["facts"]:
        st.session_state["facts"][cq_id] = cq_item['facts'] or []

    if cq_item['facts'] is not None:
        print(f"cq_item facts : {cq_item['facts']}")
        st.session_state["facts"][cq_id] = cq_item['facts']

    if "current_fact" not in st.session_state:
        st.session_state["current_fact"] = {}

    if encoded_cq_id not in st.session_state["current_fact"]:
        st.session_state["current_fact"][encoded_cq_id] = {
        "fact": "",
        "subject": "",
        "predicate": "",
        "object": ""
    }

    answer_text = st.text_input("📝 Fact", value=cq_item["Answer"], key=f"fact_text_{cq_id}")
    current_obj = st.session_state["current_fact"][encoded_cq_id]

    st.markdown("##### 🧩 Map this fact into a triple:")

    #with st.form(key=f"form_{encoded_cq_id}"):
    fact_text = st.text_input("✏️ Fact Sentence", value=current["fact"], key=f"fact_text_{encoded_cq_id}")

    custom_mode = st.checkbox("🛠️ Enable Custom Mapping", key=f"custom_custom_mode_{encoded_cq_id}")

    if custom_mode:
        subj = st.text_input("🔹 Custom Subject Class", key=f"custom_subj_free_text_{encoded_cq_id}")
    else:
        subj = st.selectbox("🔹 Subject Class", [""] + ontology["class_labels"],
                        index=ontology["class_labels"].index(current["subject"]) if current["subject"] in ontology[
                            "class_labels"] else 0, key=f"subj_{encoded_cq_id}")

    current_obj["subject"] = subj
    st.session_state["current_fact"][encoded_cq_id] = current_obj
    subj_uri = ontology["label_to_class"].get(subj, "")
    st.write(f"**Subject URI:** `{subj_uri}`")

    pred_options = [
        ontology["properties"][p]["label"]
        for p in ontology["subject_predicate_map"].get(subj_uri, [])
        if p in ontology["properties"]
    ]

    if custom_mode:
        pred = st.text_input("🔸 Custom Predicate", key=f"custom_pred_free_text_{encoded_cq_id}")
    else:
        pred = st.selectbox("🔸 Predicate", [""] + pred_options,
                        index=pred_options.index(current["predicate"]) + 1 if current[
                                                                                  "predicate"] in pred_options else 0,
                        key=f"pred_{encoded_cq_id}")
    current_obj["predicate"] = pred
    st.session_state["current_fact"][encoded_cq_id] = current_obj
    pred_uri = ontology["label_to_property"].get(pred, "")
    pred_label = next((k for k, v in ontology["label_to_property"].items() if v == pred_uri), pred_uri)

    st.write(f"**Predicate URI:** `{pred_label}`")

    obj_class_uris = ontology["predicate_object_map"].get(pred_label, set())
    obj_labels = [ontology["classes"].get(o, o) for o in obj_class_uris]

    if custom_mode:
        obj = st.text_input("🔹 Custom Object Class", key=f"custom_obj_free_text_{encoded_cq_id}")
    else:
        obj = st.selectbox("🔹 Object Class", [""] + obj_labels,
                       index=obj_labels.index(current["object"]) + 1 if current["object"] in obj_labels else 0,
                       key=f"obj_{encoded_cq_id}")

    if custom_mode:
        for new_label in [subj, obj]:
            if new_label not in ontology["class_labels"]:
                ontology["class_labels"].append(new_label)
                ontology["label_to_class"][new_label] = f"custom:{new_label}"

        if pred not in ontology["property_labels"]:
            ontology["property_labels"].append(pred)
            ontology["label_to_property"][pred] = f"custom:{pred}"

    current_obj["object"] = obj
    st.session_state["current_fact"][encoded_cq_id] = current_obj
    obj_uri = ontology["label_to_class"].get(obj, "")
    st.write("**Object URI:**", obj_uri)

    if st.button("➕ Add Triple", key=f"add_triple_{encoded_cq_id}"):
        st.write("**new Fact id:**", len(st.session_state["facts"][cq_id]))
        new_fact = {
            "id": len(st.session_state["facts"][cq_id]),
            "fact": fact_text,
            "subject": subj,
            "predicate": pred,
            "object": obj
        }
        st.session_state["facts"][cq_id].append(new_fact)
        st.success("✅ Triple added!")
        #st.rerun()

    if st.session_state["facts"][cq_id]:
        st.markdown("#### ✅ Mapped Triples:")
        for i, triple in enumerate(st.session_state["facts"][cq_id]):
            col1, col2 = st.columns([10, 1])
            with col1:
                subj_lbl = ontology["classes"].get(triple['subject'], triple['subject'])
                pred_data = ontology["properties"].get(triple['predicate'], {})
                pred_lbl = pred_data.get("label", triple['predicate'])
                if isinstance(pred_lbl, list):
                    pred_lbl = pred_lbl[0]
                obj_lbl = ontology["classes"].get(triple['object'], triple['object'])

                st.markdown(f"- **{i + 1}.** {triple['fact']} — `{subj_lbl} → {pred_lbl} → {obj_lbl}`")
            with col2:
                delete_clicked = st.button("❌", key=f"delete_{encoded_cq_id}_{i}")
                if delete_clicked:
                    all_triples = st.session_state["facts"][cq_id]
                    new_triples = [t for j, t in enumerate(all_triples) if j != i]
                    st.session_state["facts"][cq_id] = new_triples
                    st.success("🗑️ Triple deleted")
                    #st.rerun()


# -------------------- App Entry --------------------

def main():
    st.set_page_config(page_title="KG Fact Triple Annotator", layout="wide")
    st.title("🎼 Simple KG Triple Mapper")

    ontology = load_ontology()

    uploaded = st.file_uploader("📂 Upload CQ JSON File", type=["json"])
    cq_data = load_cq_data(uploaded) if uploaded else []

    if st.button("💾 Export Triples to JSON"):
        output_data = cq_data
        for item in output_data:
            cq_id = item["CQ_ID"]
            item["facts"] = st.session_state["facts"].get(cq_id, [])
        st.download_button(
            "📥 Download Enhanced JSON",
            data=json.dumps(output_data, indent=2),
            file_name="enhanced_triples.json",
            mime="application/json"
        )

    st.session_state.setdefault("facts", {})
    st.session_state.setdefault("modeled_triples", {})
    st.session_state.setdefault("fact_widget_map", {})

    if "edit_mode" not in st.session_state:
        st.session_state["edit_mode"] = {}

    if not cq_data:
        st.info("Upload a JSON file containing CQs with fields: `CQ_ID`, `CQ_Text`, and `Answer`.")
        return

    if "fact_triples" not in st.session_state:
        st.session_state["fact_triples"] = {}

    st.markdown("---")
    for cq_item in cq_data:
        render_cq_editor(ontology, cq_item)
        st.markdown("---")



if __name__ == "__main__":
    main()
