import os
import sys
import subprocess


def extract_persona_from_filename(filepath: str) -> str:
    return os.path.basename(filepath).split("_")[0].capitalize()


def generate_fact_prompt(text: str) -> str:
    return f"""
You will be given a scenario describing an interaction with a digital exhibit. Your task is to extract **only factual statements** from the scenario. Avoid subjective judgments, personas, or opinions. Return each fact as a single sentence.

Respond with a **numbered list** of factual statements only.

Scenario:
{text.strip()}
""".strip()


def call_ollama(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", "llama3", prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return result.stdout.strip()


def write_output(persona: str, prompt: str, response: str):
    os.makedirs("llama_outputs", exist_ok=True)

    prompt_file = f"llama_outputs/{persona.lower()}_facts_prompt.txt"
    output_file = f"llama_outputs/{persona.lower()}_facts_output.txt"

    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(response)

    print(f"Prompt saved to: {prompt_file}")
    print(f"Response saved to: {output_file}")


def main(filepath: str):
    persona = extract_persona_from_filename(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        scenario_text = f.read()

    prompt = generate_fact_prompt(scenario_text)
    response = call_ollama(prompt)
    write_output(persona, prompt, response)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_facts_with_llama.py scenarios/emma_scenario.txt")
        sys.exit(1)

    main(sys.argv[1])
