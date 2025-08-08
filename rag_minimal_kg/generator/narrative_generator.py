import ollama
import yaml
from pathlib import Path
from jinja2 import Template, Environment, FileSystemLoader


# Load personas from YAML
def load_personas(yaml_path="config/personas.yaml"):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


PERSONAS = load_personas()

template_dir = Path(__file__).parent / "prompts"
env = Environment(loader=FileSystemLoader(str(template_dir)))
template = env.get_template("story_prompt_template.jinja2")

def generate_prompt(facts, persona_name, profile):
    return template.render(
        facts=facts,
        persona=persona_name,
        description=profile["description"],
        tone=profile["tone"],
        detail_level=profile["detail_level"],
        key_words=profile["key_words"]
    )


def generate_story_with_ollama(triples, persona="Emma", model="llama3"):
    persona_data = PERSONAS.get(persona, {})
    prompt = generate_prompt(triples, persona, persona_data)

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']
