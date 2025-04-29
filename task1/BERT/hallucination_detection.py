import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import load_dataset


# 定义数据集类
class HallucinationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        premise = self.dataset[idx]['wiki_bio_text']
        hypothesis = self.dataset[idx]['gpt3_sentences']
        encoding = self.tokenizer.encode_plus(
            premise,
            hypothesis,
            add_special_tokens=True,
            max_length=self.max_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )
        label = self.dataset[idx]['annotation']
        if label == 1 or label == 0.5:
            binary_label = 1  # Non-Factual
        else:
            binary_label = 0  # Factual
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(binary_label, dtype=torch.long)
        }


# 评估函数
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return accuracy, precision, recall, f1


def main():
    # 加载数据集
    dataset = load_dataset('potsawee/wiki_bio_gpt3_hallucination')
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # 加载预训练的NLI模型和分词器
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 假设NLI模型有3个类别

    # 定义超参数
    max_length = 512
    batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 创建数据集和数据加载器
    train_dataset = HallucinationDataset(train_dataset, tokenizer, max_length)
    test_dataset = HallucinationDataset(test_dataset, tokenizer, max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 评估模型
    accuracy, precision, recall, f1 = evaluate(model, test_dataloader, device)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()