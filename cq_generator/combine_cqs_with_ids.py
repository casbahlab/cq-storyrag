import pandas as pd
import os

def assign_ids(df, prefix):
    df = df.copy()
    df = df.reset_index(drop=True)
    df['CQ_ID'] = [f"CQ-{prefix}{i+1}" for i in range(len(df))]
    return df

def combine_and_label(emma_file, luca_file, output_file):
    # Read the two CSVs
    emma_df = pd.read_csv(emma_file)
    luca_df = pd.read_csv(luca_file)

    # Assign IDs
    emma_df = assign_ids(emma_df, 'E')
    luca_df = assign_ids(luca_df, 'L')

    # Add persona column if not present
    if 'Persona' not in emma_df.columns:
        emma_df['Persona'] = 'Emma'
    if 'Persona' not in luca_df.columns:
        luca_df['Persona'] = 'Luca'

    # Combine
    combined_df = pd.concat([emma_df, luca_df], ignore_index=True)

    # Optional: reorder columns
    columns = ['CQ_ID', 'Scenario_Sentence', 'Generated_CQ', 'Cleaned_CQ', 'CQ_Type', 'Persona']
    combined_df = combined_df[[col for col in columns if col in combined_df.columns]]

    # Save output
    combined_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Combined CQ file saved to: {output_file}")

if __name__ == "__main__":
    # Input files
    emma_csv = "output/emma_generated_CQs_llama_facts.csv"
    luca_csv = "output/luca_generated_CQs_llama_facts.csv"
    output_csv = "output/combined_CQs_with_ids.csv"

    combine_and_label(emma_csv, luca_csv, output_csv)
