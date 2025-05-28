import json
import os
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

def parse_multiwoz(input_dir, output_dir):
    for split in ["train", "test"]:
        messages = []
        split_dir = os.path.join(input_dir, split)
        for fname in os.listdir(split_dir):
            if fname.endswith('.json'):
                with open(os.path.join(split_dir, fname), "r") as f:
                    dialogs = json.load(f)
                for dialog in dialogs:
                    msg_seq = []
                    for turn in dialog.get("turns", []):
                        role = turn["speaker"].lower()
                        if role == "user":
                            role = "user"
                        elif role == "system":
                            role = "assistant"
                        else:
                            continue
                        msg_seq.append({
                            "role": role,
                            "content": turn["utterance"].strip()
                        })
                    domains = dialog.get("services", [])
                    messages.append({
                        "dialogue_id": dialog.get("dialogue_id", ""),
                        "domains": domains,
                        "messages": msg_seq
                    })
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{split}_parsed.json")
        with open(out_path, "w") as f:
            json.dump(messages, f, indent=2)
        print(f"Saved {len(messages)} dialogues with domains to {out_path}")

def sft_format(parsed_json_path, sft_jsonl_path, add_domain_system_msg=False):
    with open(parsed_json_path, "r") as f:
        dialogs = json.load(f)
    count = 0
    with open(sft_jsonl_path, "w") as outf:
        for dialog in dialogs:
            messages = dialog["messages"]
            # Optionally prepend domain info as a system message
            if add_domain_system_msg and dialog.get("domains"):
                domain_msg = {"role": "system", "content": f"Domains: {', '.join(dialog['domains'])}"}
                messages = [domain_msg] + messages
            record = {"messages": messages, "domains": dialog.get("domains", [])}
            outf.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} SFT-format dialogues to {sft_jsonl_path}")

def randomforest_classify(parsed_json_path):
    with open(parsed_json_path, "r") as f:
        dialogs = json.load(f)
    texts = []
    labels = []
    for d in dialogs:
        # Use all messages as "document"
        joined = " ".join([m["content"] for m in d["messages"]])
        texts.append(joined)
        labels.append(d.get("domains", []))
    # Convert domains to binary matrix
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(labels)
    # Simple text vectorization
    vect = TfidfVectorizer(max_features=2000)
    X = vect.fit_transform(texts)
    # Split for evaluation
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, Y_train)
    acc = clf.score(X_test, Y_test)
    print(f"RandomForest multi-label domain accuracy: {acc:.3f}")
    # Predict one example
    idx = np.random.choice(len(X_test.indices))
    x_sample = X_test[idx]
    pred = clf.predict(x_sample)
    print("Example prediction:")
    print("Text:", texts[idx][:200])
    print("Predicted domains:", mlb.inverse_transform(pred)[0])
    print("True domains:", mlb.inverse_transform(Y_test[idx])[0])

def show_menu():
    print("\nMultiWOZ Pipeline Menu:")
    print("1. Parse dialogs (with domains)")
    print("2. Convert to SFT JSONL (for Llama 3)")
    print("3. Train/test RandomForest domain classifier")
    print("4. Exit")

if __name__ == "__main__":
    while True:
        show_menu()
        choice = input("Select option (1-4): ").strip()
        if choice == "1":
            in_dir = input("Enter MultiWOZ input dir: ").strip()
            out_dir = input("Enter output dir: ").strip()
            parse_multiwoz(in_dir, out_dir)
        elif choice == "2":
            in_file = input("Parsed JSON file (e.g., train_parsed.json): ").strip()
            out_file = input("Output SFT JSONL file: ").strip()
            domain_msg = input("Add domain info as system message? (y/n): ").strip().lower() == "y"
            sft_format(in_file, out_file, add_domain_system_msg=domain_msg)
        elif choice == "3":
            in_file = input("Parsed JSON file for classification: ").strip()
            randomforest_classify(in_file)
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Try again.")

