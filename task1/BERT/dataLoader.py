import torch
import json
from transformers import BertTokenizer
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


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



# import torch
# import json
# from transformers import BertTokenizer
# from torch.utils.data import Dataset
# from sklearn.model_selection import train_test_split


# def load_jsonl(filename):
#     data = []
#     with open(filename, 'r') as file:
#         for line in file:
#             data.append(json.loads(line))
#     return data


# def prepare_data_for_model(data):
#     model_inputs = []
#     labels = []
#     for item in data:
#         wiki_bio_text = item['wiki_bio_text']
#         gpt3_sentences = item['gpt3_sentences']
#         annotations = item['annotation']
#         for sentence, annotation in zip(gpt3_sentences, annotations):
#             input_text = (wiki_bio_text, sentence)
#             model_inputs.append(input_text)
#             # 调整标签处理逻辑
#             if annotation == "accurate":
#                 label = 0  # 可视为 Factual
#             else:
#                 label = 1  # 可视为 Non-Factual
#             labels.append(label)
#     return model_inputs, labels


# def split_data(model_inputs, labels, train_size):
#     inputs_train, inputs_remaining, labels_train, labels_remaining = train_test_split(
#         model_inputs, labels, train_size=train_size, random_state=42
#     )

#     test_val_size = 0.5
#     inputs_val, inputs_test, labels_val, labels_test = train_test_split(
#         inputs_remaining, labels_remaining, train_size=test_val_size, random_state=42
#     )

#     return inputs_train, labels_train, inputs_val, labels_val, inputs_test, labels_test


# class NLIDataset(Dataset):
#     def __init__(self, inputs, labels, tokenizer, max_length):
#         filtered_inputs = []
#         filtered_labels = []
#         for input, label in zip(inputs, labels):
#             if label != "-":
#                 filtered_inputs.append(input)
#                 filtered_labels.append(label)

#         self.inputs = filtered_inputs
#         self.labels = filtered_labels
#         self.tokenizer: BertTokenizer = tokenizer
#         self.max_length = max_length
#         # 这里标签映射不再使用之前针对 NLI 的逻辑
#         self.label_map = {0: 0, 1: 1} 

#     def __len__(self):
#         return len(self.inputs)

#     def __getitem__(self, index):
#         # print(f"Index: {index}, Label: {self.labels[index]}")
#         input_text = f"Wiki Bio: {self.inputs[index][0]} GPT3 Sentence: {self.inputs[index][1]}"
#         label = self.labels[index]

#         encoding = self.tokenizer.encode_plus(
#             input_text,
#             add_special_tokens=True,
#             max_length=self.max_length,
#             return_tensors="pt",
#             padding="max_length",
#             truncation=True,
#         )
#         return {
#             'input_ids': encoding.input_ids[0],
#             'attention_mask': encoding.attention_mask[0],
#             'token_type_ids': encoding.token_type_ids[0],
#             'labels': torch.tensor(label, dtype=torch.long)
#         }