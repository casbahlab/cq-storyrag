import ollama
import yaml
from pathlib import Path
from jinja2 import Template


# Load personas from YAML
def load_personas(yaml_path="config/personas.yaml"):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


PERSONAS = load_personas()

OLLAMA_PROMPT_TEMPLATE = """
You are a storytelling assistant generating a narrative from RDF triples.

Persona Description:
{{ persona_description }}

Instructions:
- Generate a coherent narrative based on the triples below.
- Follow the persona's tone and information preference.
- Do not just list the triples. Tell a story.

Triples:
{% for s, p, o in triples %}
- {{ s }} {{ p }} {{ o }}
{% endfor %}

Narrative:
"""


def generate_story_with_ollama(triples, persona="Emma", model="llama3"):
    persona_data = PERSONAS.get(persona, {})
    persona_description = persona_data.get("description", "A general music audience.")

    prompt = Template(OLLAMA_PROMPT_TEMPLATE).render(
        triples=triples,
        persona_description=persona_description
    )

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']
