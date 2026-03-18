# =========================================
# EXPERIMENT 7: NAMED ENTITY RECOGNITION
# =========================================

import spacy
import pandas as pd
import random
import re
from spacy.training import Example
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# =========================================
# STEP 1: DATASET
# =========================================

base_sentences = [
    "Elon Musk is the CEO of Tesla and SpaceX.",
    "Sundar Pichai is the CEO of Google.",
    "Bill Gates founded Microsoft in the United States.",
    "Apple is headquartered in Cupertino, California.",
    "Amazon was founded by Jeff Bezos in Seattle.",
]

extra_sentences = [
    "Mark Zuckerberg founded Facebook in California.",
    "Larry Page co-founded Google in the United States.",
    "Satya Nadella is the CEO of Microsoft.",
    "Tim Cook is the CEO of Apple.",
    "Jeff Bezos founded Amazon in Seattle.",
    "SpaceX is headquartered in the United States.",
    "Tesla operates in California.",
    "Google is based in Mountain View.",
    "Microsoft has offices in Seattle.",
    "Apple was founded by Steve Jobs."
]

text_data = base_sentences + extra_sentences
print("Dataset Prepared.")

# =========================================
# STEP 2: RULE-BASED NER
# =========================================

def rule_based_ner(text):
    entities = []
    
    person_pattern = r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b"
    orgs = ["Google", "Apple", "Microsoft", "Amazon", "Tesla", "SpaceX", "Facebook"]
    locations = ["California", "Seattle", "United States", "Cupertino", "Mountain View"]

    for match in re.finditer(person_pattern, text):
        if match.group() not in locations:
            entities.append((match.group(), "PERSON"))

    for org in orgs:
        if org in text:
            entities.append((org, "ORG"))

    for loc in locations:
        if loc in text:
            entities.append((loc, "GPE"))

    return entities

print("\nRule-Based Output:")
for text in text_data[:3]:
    print(text)
    print(rule_based_ner(text))

# =========================================
# STEP 3: PRETRAINED MODEL
# =========================================

nlp_pretrained = spacy.load("en_core_web_sm")

TRAIN_DATA = []
for text in text_data:
    doc = nlp_pretrained(text)
    entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
    if entities:
        TRAIN_DATA.append((text, {"entities": entities}))

print("\nAuto-labeled dataset size:", len(TRAIN_DATA))

# =========================================
# STEP 4: TRAIN-TEST SPLIT
# =========================================

train_data, test_data = train_test_split(TRAIN_DATA, test_size=0.3, random_state=42)

# =========================================
# STEP 5: TRAIN CUSTOM MODEL
# =========================================

nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")

for _, annotations in train_data:
    for ent in annotations["entities"]:
        ner.add_label(ent[2])

optimizer = nlp.initialize()

print("\nTraining model...")

for epoch in range(30):
    random.shuffle(train_data)
    losses = {}

    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], drop=0.35, losses=losses)

    print(f"Epoch {epoch+1}, Loss: {losses}")

nlp.to_disk("custom_ner_model")
print("\nModel saved.")

# =========================================
# STEP 6: EVALUATION (ALL THREE MODELS 🔥)
# =========================================

from sklearn.metrics import classification_report

y_true = []
y_rule = []
y_pre = []
y_custom = []

print("\nTest Predictions (Comparison):")

for text, annotations in test_data:
    
    # TRUE LABELS
    true_labels = [label for _, _, label in annotations["entities"]]

    # RULE-BASED
    rule_entities = rule_based_ner(text)
    rule_labels = [label for _, label in rule_entities]

    # PRETRAINED
    doc_pre = nlp_pretrained(text)
    pre_labels = [ent.label_ for ent in doc_pre.ents]

    # CUSTOM
    doc_custom = nlp(text)
    custom_labels = [ent.label_ for ent in doc_custom.ents]

    print("\nText:", text)
    print("Actual:", true_labels)
    print("Rule:", rule_labels)
    print("Pretrained:", pre_labels)
    print("Custom:", custom_labels)

    # ALIGN LENGTHS (IMPORTANT)
    min_len = min(len(true_labels), len(rule_labels), len(pre_labels), len(custom_labels))

    y_true.extend(true_labels[:min_len])
    y_rule.extend(rule_labels[:min_len])
    y_pre.extend(pre_labels[:min_len])
    y_custom.extend(custom_labels[:min_len])

# =========================================
# STEP 7: CLASSIFICATION REPORTS 🔥
# =========================================

print("\n\n========== CLASSIFICATION REPORTS ==========")

print("\n--- Rule-Based NER ---")
print(classification_report(y_true, y_rule, zero_division=0))

print("\n--- Pretrained Model ---")
print(classification_report(y_true, y_pre, zero_division=0))

print("\n--- Custom Trained Model ---")
print(classification_report(y_true, y_custom, zero_division=0))

# =========================================
# STEP 8: USER INPUT WITH COMPARISON 
# =========================================

print("\n========== USER INPUT COMPARISON ==========")

while True:
    user_text = input("\nEnter a sentence (or type 'exit'): ")

    if user_text.lower() == "exit":
        break

    # -------- RULE-BASED --------
    rule_entities = rule_based_ner(user_text)

    # -------- PRETRAINED MODEL --------
    doc_pre = nlp_pretrained(user_text)
    pre_entities = [(ent.text, ent.label_) for ent in doc_pre.ents]

    # -------- CUSTOM MODEL --------
    doc_custom = nlp(user_text)
    custom_entities = [(ent.text, ent.label_) for ent in doc_custom.ents]

    # -------- PRINT COMPARISON --------
    print("\n==============================")
    print("TEXT:", user_text)

    print("\nRule-Based NER:")
    print(rule_entities if rule_entities else "No entities found")

    print("\nPretrained Model:")
    print(pre_entities if pre_entities else "No entities found")

    print("\nCustom Trained Model:")
    print(custom_entities if custom_entities else "No entities found")

    print("==============================")

# =========================================
# STEP 9: VISUALIZATION
# =========================================

from spacy import displacy

print("\nLaunching visualization on last input...")

if user_text.lower() != "exit":
    doc = nlp(user_text)
    displacy.serve(doc, style="ent", auto_select_port=True)