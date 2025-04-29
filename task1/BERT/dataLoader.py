import torch
import json
from transformers import BertTokenizer
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from datasets import load_dataset

def load_jsonl(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def prepare_data_for_model(data):
    model_inputs = []
    labels = []
    for item in data:
        sentence1 = item['sentence1']
        sentence2 = item['sentence2']
        label = item['gold_label']
        model_inputs.append((sentence1, sentence2))
        labels.append(label)
    return model_inputs, labels

def split_data(model_inputs, labels, train_size):
    inputs_train, inputs_remaining, labels_train, labels_remaining = train_test_split(
        model_inputs, labels, train_size=train_size, random_state=42
    )

    test_val_size = 0.5
    # In the remaining, half are valuation set, other half are test set
    inputs_val, inputs_test, labels_val, labels_test = train_test_split(
        inputs_remaining, labels_remaining, train_size=test_val_size, random_state=42
    )

    return inputs_train, labels_train, inputs_val, labels_val, inputs_test, labels_test

def load_hallucination_dataset():
    dataset = load_dataset('potsawee/wiki_bio_gpt3_hallucination')
    all_data = []
    for split in dataset.values():
        all_data.extend(split)

    train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)

    train_inputs = []
    train_labels = []
    test_inputs = []
    test_labels = []

    for item in train_data:
        premise = item['wiki_bio_text']
        hypothesis = item['gpt3_sentences']
        label = item['annotation']
        if label == 1 or label == 0.5:  # Major or Minor Inaccurate -> Non-Factual
            binary_label = 1
        else:  # Accurate -> Factual
            binary_label = 0
        train_inputs.append((premise, hypothesis))
        train_labels.append(binary_label)

    for item in test_data:
        premise = item['wiki_bio_text']
        hypothesis = item['gpt3_sentences']
        label = item['annotation']
        if label == 1 or label == 0.5:
            binary_label = 1
        else:
            binary_label = 0
        test_inputs.append((premise, hypothesis))
        test_labels.append(binary_label)

    return train_inputs, train_labels, test_inputs, test_labels

class NLIDataset(Dataset):
    def __init__(self, inputs, labels, tokenizer, max_length):
        filtered_inputs = []
        filtered_labels = []
        for input, label in zip(inputs, labels):
            if label != "-":
                filtered_inputs.append(input)
                filtered_labels.append(label)

        self.inputs = filtered_inputs
        self.labels = filtered_labels
        self.tokenizer: BertTokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_text = f"Sentence1: {self.inputs[index][0]} Sentence2: {self.inputs[index][1]} Relationship:"
        label = self.label_map[self.labels[index]]

        encoding = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return {
            'input_ids': encoding.input_ids[0],
            'attention_mask': encoding.attention_mask[0],
            'token_type_ids': encoding.token_type_ids[0],
            'labels': torch.tensor(label, dtype=torch.long)
        }