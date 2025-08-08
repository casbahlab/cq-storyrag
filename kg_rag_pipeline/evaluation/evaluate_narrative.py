import json
from textstat.textstat import textstat
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import nltk
nltk.download('vader_lexicon')


# 1. Plan Categorization Validation
def validate_plan(generated_narrative: str, plan: dict):
    plan_validation = {
        "entry_section_covered": False,
        "core_section_covered": False,
        "exit_section_covered": False,
    }

    # Entry section check (keywords)
    entry_keywords = ['introduction', 'beginning', 'start']
    plan_validation["entry_section_covered"] = any(keyword in generated_narrative.lower() for keyword in entry_keywords)

    # Core section check (keywords)
    core_keywords = ['performance', 'audience', 'broadcast']
    plan_validation["core_section_covered"] = any(keyword in generated_narrative.lower() for keyword in core_keywords)

    # Exit section check (keywords)
    exit_keywords = ['impact', 'legacy', 'conclusion']
    plan_validation["exit_section_covered"] = any(keyword in generated_narrative.lower() for keyword in exit_keywords)

    return plan_validation


# 2. Query Result Validation (checking if facts are correctly reflected in the narrative)
import re


# A helper function to extract keywords (last part of the URL)
def extract_keyword_from_url(url: str):
    return url.split('#')[-1] if '#' in url else url.split('/')[-1]


# A function to validate query results based on keyword presence in the narrative
def validate_query_results(generated_narrative: str, query_facts: list):
    missing_facts = []

    # Iterate over the facts in the query_facts
    for fact in query_facts:
        fact_question = fact['question']
        found_fact = False

        # Check if the question text contains any keyword from the rows
        for row in fact['rows']:
            # Iterate through all keys in the row (assuming they are URL or descriptive text)
            for key, value in row.items():
                if isinstance(value, str):  # Ensure the value is a string before processing
                    # Extract the keyword from the value (URL or other text)
                    keyword = extract_keyword_from_url(value)

                    # Check if the keyword exists in the generated narrative
                    if re.search(r'\b' + re.escape(keyword) + r'\b', generated_narrative):
                        found_fact = True
                        break

        # If no relevant keyword found in the narrative, add it to missing facts
        if not found_fact:
            missing_facts.append(fact_question)

    return missing_facts


# 3. Facts Representation in the Narrative (check if facts are in the right sections)
def validate_facts_in_sections(generated_narrative: str, facts_by_section: dict):
    fact_section_validation = {
        "Entry": [],
        "Core": [],
        "Exit": [],
    }

    for section, facts in facts_by_section.items():
        for fact in facts:
            if fact.lower() in generated_narrative.lower():
                fact_section_validation[section].append(fact)

    return fact_section_validation


# 4. Logical Flow and Language Consistency (Check readability and transitions)
def readability_and_transitions(generated_narrative: str):
    # Check readability score
    readability_score = textstat.flesch_kincaid_grade(generated_narrative)

    # Check transitions (simple approach)
    transition_keywords = ["first", "next", "finally", "to conclude"]
    transitions_ok = any(keyword in generated_narrative.lower() for keyword in transition_keywords)

    # Sentiment analysis for emotional consistency (enthusiastic tone)
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(generated_narrative)

    return {
        "readability_score": readability_score,
        "transitions_ok": transitions_ok,
        "sentiment": sentiment,
    }


# 5. Overall Evaluation of Narrative
def evaluate_narrative(json_file_path: str):
    # 1. Load JSON file
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Extract necessary information from the JSON
    plan = data['plan']  # Get the 'plan' key
    facts_data = data['facts']  # Get the 'facts' key
    facts_by_section = {
        "Entry": [fact['question'] for fact in facts_data['Entry']],
        "Core": [fact['question'] for fact in facts_data['Core']],
        "Exit": [fact['question'] for fact in facts_data['Exit']]
    }
    generated_narrative = data['narrative']  # Get the narrative from the JSON file

    # 1. Plan Categorization Validation
    plan_validation_results = validate_plan(generated_narrative, plan)

    # 2. Query Result Validation (Check missing facts)
    query_facts = facts_data['Entry'] + facts_data['Core'] + facts_data['Exit']
    missing_facts = validate_query_results(generated_narrative, query_facts)

    # 3. Facts Representation in the Narrative (Check section validity)
    fact_section_results = validate_facts_in_sections(generated_narrative, facts_by_section)

    # 4. Logical Flow and Language Consistency (Readability, transitions, sentiment)
    flow_results = readability_and_transitions(generated_narrative)

    return {
        "plan_validation": plan_validation_results,
        "missing_facts": missing_facts,
        "fact_section_validation": fact_section_results,
        "flow_results": flow_results
    }


# Example usage
json_file_path = "../narrative_output_20250808_222519.json"

# Evaluate the narrative
evaluation_results = evaluate_narrative(json_file_path)
print("Evaluation Results:", evaluation_results)
