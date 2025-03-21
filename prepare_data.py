import json
import re
import uuid
import os
import xml.etree.ElementTree as ET
from underthesea import sent_tokenize
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_DIR = os.path.join(BASE_DIR, "VNLegalText", "dataset")

os.makedirs(DATA_DIR, exist_ok=True)

output_file = os.path.join(DATA_DIR, "tag_sentences.jsonl")
label_file = os.path.join(DATA_DIR, "label_sentences.json")
train_file = os.path.join(DATA_DIR, "train.jsonl")
val_file = os.path.join(DATA_DIR, "val.jsonl")
test_file = os.path.join(DATA_DIR, "test.jsonl")


def clean_text(text: str) -> str:
    text = re.sub(r"\n\s*\n", "\n", text)
    text = text.replace(";", ".")
    text = re.sub(r"\s+", " ", text).strip()
    if not text.endswith(('.', '?', '!', '...')):
        text += '.'
    return text


def contains_entity(sentence, entities):
    return any(ent[0] in sentence for ent in entities)


with open(output_file, "w", encoding="utf-8") as jsonl_file:
    for filename in os.listdir(DATASET_DIR):
        if filename.endswith(".xml"):
            file_path = os.path.join(DATASET_DIR, filename)
            print(f"📂 Processing file: {filename}")

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                print(f"⚠️ File {filename} empty. Skipping...")
                continue

            wrapped_content = f'<?xml version="1.0" encoding="UTF-8"?><root>{content}</root>'

            try:
                tree = ET.ElementTree(ET.fromstring(wrapped_content))
                root = tree.getroot()

                entities = []
                for elem in root.iter():
                    if elem.text and elem.text.strip():
                        entity_text = elem.text.strip()
                        tag = elem.tag
                        relation = elem.attrib.get("rel", "None")
                        entities.append((entity_text, tag, relation))

                content = clean_text(content)
                sentences = sent_tokenize(content)

                for sent in sentences:
                    if contains_entity(sent, entities):
                        json_record = {"filename": filename, "sentence": sent.strip()}
                        jsonl_file.write(json.dumps(json_record, ensure_ascii=False) + "\n")

            except ET.ParseError as e:
                print(f"❌ XML Parse Error in file {filename}: {e}")

print(f"✅ Created {output_file}")

# create label_sentences
data_output = []
pattern = r'<(\w+) rel="[^"]*">\s*(.*?)\s*</\1>'

with open(output_file, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        text = entry["sentence"]
        cleaned_text = re.sub(pattern, r'\2', text).strip()
        matches = list(re.finditer(pattern, text))
        annotations = []
        used_positions = set()

        for match in matches:
            tag = match.group(1)
            entity_text = match.group(2).strip()

            for entity_match in re.finditer(re.escape(entity_text), cleaned_text):
                start, end = entity_match.start(), entity_match.end()
                if (start, end) not in used_positions:
                    used_positions.add((start, end))
                    annotations.append({
                        "id": str(uuid.uuid4()),
                        "value": {
                            "start": start,
                            "end": end,
                            "text": entity_text,
                            "labels": [tag]
                        }
                    })
                    break

        data_output.append({
            "data": {"value": cleaned_text},
            "annotations": [{"result": annotations}]
        })

with open(label_file, "w", encoding="utf-8") as f_out:
    json.dump(data_output, f_out, ensure_ascii=False, indent=4)

print(f"✅ Created {label_file}")

# split train, val, test
random.shuffle(data_output)
train_size = int(len(data_output) * 0.8)
val_size = int(len(data_output) * 0.1)
test_size = len(data_output) - train_size - val_size

save_jsonl = lambda data, filename: open(filename, "w", encoding="utf-8").writelines(
    json.dumps(entry, ensure_ascii=False) + "\n" for entry in data
)

save_jsonl(data_output[:train_size], train_file)
save_jsonl(data_output[train_size:train_size + val_size], val_file)
save_jsonl(data_output[train_size + val_size:], test_file)

print(f"✅ Split data: {train_size} train, {val_size} val, {test_size} test")

# convert JSONL to CoNLL
def jsonl_to_conll(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]

    with open(output_file, "w", encoding="utf-8") as f_out:
        for entry in data:
            sentence = entry["data"]["value"]
            annotations = entry["annotations"][0]["result"]

            words = sentence.split()
            labels = ["O"] * len(words)

            for ann in annotations:
                start, end, label = ann["value"]["start"], ann["value"]["end"], ann["value"]["labels"][0]
                entity_text = ann["value"]["text"]
                entity_words = entity_text.split()

                for i in range(len(words)):
                    if words[i] == entity_words[0] and " ".join(words[i:i+len(entity_words)]) == entity_text:
                        labels[i] = f"B-{label}"
                        for j in range(1, len(entity_words)):
                            labels[i+j] = f"I-{label}"
                        break

            for word, tag in zip(words, labels):
                f_out.write(f"{word} {tag}\n")
            f_out.write("\n")

    print(f"✅ Created {output_file}")

jsonl_to_conll(train_file, os.path.join(DATA_DIR, "train.conll"))
jsonl_to_conll(val_file, os.path.join(DATA_DIR, "val.conll"))
jsonl_to_conll(test_file, os.path.join(DATA_DIR, "test.conll"))
