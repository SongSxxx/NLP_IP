import torch
import argparse
import json
import datetime
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataLoader import NLIDataset
from training import train_epoch, evaluate, test_accuracy

# 定义函数将 NLI 标签转换为幻觉检测标签
def nli_to_hallucination_label(nli_label):
    # 这里假设 entailment 对应 Factual，其他对应 Non-Factual
    if nli_label == 0:  # 假设 0 是 entailment
        return 0  # Factual
    else:
        return 1  # Non-Factual

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train a model with specified hyperparameters.')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    args = parser.parse_args()

    # 超参数设置
    batch_size = args.bs
    learning_rate = args.lr
    num_epochs = 5
    max_length = 512

    # 检查可用设备
    cuda_able = torch.cuda.is_available()
    mps_able = (torch.backends.mps.is_available() and torch.backends.mps.is_built())
    device = 'cuda:0' if cuda_able else 'mps' if mps_able else 'cpu'

    # 加载预训练的 BERT 模型和分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    model.to(device)

    # 加载数据集
    dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")

    # 打印数据集的所有分割名称
    print(dataset.keys())

    # 使用 'evaluation' 分割
    split_name = 'evaluation'

    # 准备数据
    inputs = []
    labels = []
    for entry in dataset[split_name]:
        wiki_bio_text = entry['wiki_bio_text']
        gpt3_sentences = entry['gpt3_sentences']
        annotations = entry['annotation']
        for sentence, annotation in zip(gpt3_sentences, annotations):
            # 将 wiki_bio_text 视为前提，gpt3_sentence 视为假设
            input_text = (wiki_bio_text, sentence)
            inputs.append(input_text)
            # 将注释转换为二进制标签
            if annotation == 0:
                label = 0  # Factual
            else:
                label = 1  # Non-Factual
            labels.append(label)

    # 划分训练集和测试集
    train_size = 0.8
    train_inputs = inputs[:int(train_size * len(inputs))]
    train_labels = labels[:int(train_size * len(labels))]
    test_inputs = inputs[int(train_size * len(inputs)):]
    test_labels = labels[int(train_size * len(labels)):]

    train_dataset = NLIDataset(train_inputs, train_labels, tokenizer, max_length)
    val_dataset = NLIDataset(train_inputs, train_labels, tokenizer, max_length)
    test_dataset = NLIDataset(test_inputs, test_labels, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    best_model_state = None
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}")

        # Check best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            print(f"New best model found at epoch {epoch + 1}")

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = outputs.logits
            _, nli_preds = torch.max(logits, dim=1)

            # 将 NLI 预测标签转换为幻觉检测标签
            hallucination_preds = [nli_to_hallucination_label(pred.item()) for pred in nli_preds]
            all_preds.extend(hallucination_preds)
            all_labels.extend(labels.cpu().tolist())

    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

    # 准备结果数据
    results = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_filename = f'results/results_{timestamp}.json'

    # 将结果写入 JSON 文件
    with open(results_filename, 'w') as f:
        json.dump(results, f)