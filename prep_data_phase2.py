import json
import csv
import pandas as pd
from datasets import Dataset

SYSTEM_PROMPT = (
    "You are a helpful Indian legal advisor. Answer user queries based on Indian law, IPC, Constitution, and case judgments. "
    "Be concise and cite sources when available."
)

def build_constitution_chat(csv_path):
    df = pd.read_csv(csv_path)
    examples = []
    for idx, row in df.iterrows():
        article = row["Articles"]
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Explain this provision of the Indian Constitution:\n\n{article}"},
                {"role": "assistant", "content": article}
            ]
        })
    return examples

def build_law_chat(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    examples = []
    for rec in data:
        title = rec.get("title", "")
        desc = rec.get("description", "")
        user_q = f"What does the provision '{title}' state in Indian law?"
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_q},
                {"role": "assistant", "content": desc}
            ]
        })
    return examples

def build_ipc_chat(csv_path):
    df = pd.read_csv(csv_path)
    examples = []
    for idx, row in df.iterrows():
        section = row.get("Section", "Unknown")
        desc = row.get("Description", "")
        punishment = row.get("Punishment", "")
        user_q = f"What is IPC Section {section} about?"
        answer = f"{desc}\n\nPunishment: {punishment}"
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_q},
                {"role": "assistant", "content": answer}
            ]
        })
    return examples

def main():
    const_examples = build_constitution_chat("data/constitution.csv")
    law_examples = build_law_chat("data/indian_law.json")
    ipc_examples = build_ipc_chat("data/ipc.csv")

    all_examples = const_examples + law_examples + ipc_examples
    ds = Dataset.from_list(all_examples)
    ds.save_to_disk("data/phase2_chat")

    print(f"✅ Saved {len(all_examples)} chat examples to data/phase2_chat")

if __name__ == "__main__":
    main()
