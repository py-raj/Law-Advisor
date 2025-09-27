"""
prep_data.py
- Reads JSONL dataset (IndicLegalQA, IPC, Constitution, etc.)
- Converts Q&A into chat message format compatible with Llama 3 chat templates
- Saves as Hugging Face dataset
"""
import json
from datasets import Dataset

SYSTEM_PROMPT = (
    "You are a helpful Indian legal advisor. Answer user queries based on Indian law, IPC, Constitution, and case judgments. "
    "Be concise and cite sources when available."
)


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            # File is JSON array
            data = json.load(f)
            for rec in data:
                yield rec
        else:
            # File is JSONL
            for line in f:
                line = line.strip()
                if not line:  # skip blank lines
                    continue
                yield json.loads(line)



def build_chat_examples(input_path):
    examples = []
    for rec in load_jsonl(input_path):
        user_q = rec.get("question")
        if rec.get("case_name"):
            user_q = f"In {rec['case_name']} ({rec.get('judgement_date','')}), {rec['question']}"
        assistant_a = rec.get("answer", "")
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_q},
                {"role": "assistant", "content": assistant_a}
            ]
        })
    return examples


def main(in_path='data/indiclegalqa.json', out_path='data/indiclegalqa_chat'):
    examples = build_chat_examples(in_path)
    ds = Dataset.from_list(examples)
    ds.save_to_disk(out_path)
    print(f"Saved {len(ds)} chat examples to {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='data/indiclegalqa.json')
    p.add_argument('--output', default='data/indiclegalqa_chat')
    args = p.parse_args()
    main(args.input, args.output)