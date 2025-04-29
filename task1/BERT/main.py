import torch
import json
import logging
import argparse
import json
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataLoader import load_jsonl, prepare_data_for_model, split_data, NLIDataset, load_hallucination_dataset
from training import train_epoch, evaluate, test_accuracy
from transformers import BertTokenizer, BertForSequenceClassification

if __name__ == '__main__':
    logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train a model with specified hyperparameters.')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    args = parser.parse_args()

    # Extract arguments
    batch_size = args.bs
    learning_rate = args.lr
    num_epochs = 5
    max_length = 512

    # check available device
    cuda_able = torch.cuda.is_available()
    mps_able = (torch.backends.mps.is_available() and torch.backends.mps.is_built())
    device = 'cuda:0' if cuda_able else 'mps' if mps_able else 'cpu'

    # Load and prepare data for NLI task
    filename = "task1/dataset/matched.jsonl"
    data = load_jsonl(filename)
    inputs, labels = prepare_data_for_model(data)

    # Split data into train, val, and test sets for NLI task
    inputs_train, labels_train, inputs_val, labels_val, inputs_test, labels_test = split_data(
        inputs, labels, train_size=0.8
    )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

    model.to(device)

    train_dataset = NLIDataset(inputs_train, labels_train, tokenizer, max_length)
    val_dataset = NLIDataset(inputs_val, labels_val, tokenizer, max_length)
    test_dataset = NLIDataset(inputs_test, labels_test, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    best_model_state = None
    # Training loop for NLI task
    model.train()
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}")
        logging.info(
            f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}")

        # Check best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            print(f"New best model found at epoch {epoch + 1}")
            logging.info(f"New best model found at epoch {epoch + 1}")

    model.load_state_dict(best_model_state)

    test_acc = test_accuracy(model, test_loader, device)
    print(f"Test Accuracy for NLI task: {test_acc:.3f}")
    logging.info(f"Test Accuracy for NLI task: {test_acc:.3f}")

    # Load and prepare data for hallucination detection task
    train_inputs_hallucination, train_labels_hallucination, test_inputs_hallucination, test_labels_hallucination = load_hallucination_dataset()

    train_dataset_hallucination = NLIDataset(train_inputs_hallucination, train_labels_hallucination, tokenizer, max_length)
    test_dataset_hallucination = NLIDataset(test_inputs_hallucination, test_labels_hallucination, tokenizer, max_length)

    train_loader_hallucination = DataLoader(train_dataset_hallucination, batch_size=batch_size, shuffle=True)
    test_loader_hallucination = DataLoader(test_dataset_hallucination, batch_size=batch_size, shuffle=False)

    # Training loop for hallucination detection task
    model.train()
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader_hallucination, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss for hallucination detection: {train_loss:.3f}")

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader_hallucination:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Hallucination Detection - Accuracy: {accuracy:.4f}")
    print(f"Hallucination Detection - Precision: {precision:.4f}")
    print(f"Hallucination Detection - Recall: {recall:.4f}")
    print(f"Hallucination Detection - F1 Score: {f1:.4f}")

    # Prepare results data
    results = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'best_val_loss': best_val_loss,
        'test_accuracy_nli': test_acc,
        'hallucination_detection_accuracy': accuracy,
        'hallucination_detection_precision': precision,
        'hallucination_detection_recall': recall,
        'hallucination_detection_f1': f1
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_filename = f'results/results_{timestamp}.json'

    # Write results to JSON file
    with open(results_filename, 'w') as f:
        json.dump(results, f)