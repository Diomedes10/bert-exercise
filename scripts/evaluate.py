import argparse
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

def load_data(filepath):
    print(f"Loading data from {filepath}")
    data = pd.read_csv(filepath)
    print(f"Data loaded successfully. Columns: {data.columns}")
    return data

def tokenize_data(tokenizer, texts, batch_size=32):
    print(f"Tokenizing data with {len(texts)} texts.")
    input_ids, attention_masks = [], []

    # Determine the maximum length of the sequences
    max_length = 128  # Setting a fixed length, or you can dynamically find the max length in your dataset

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encodings = tokenizer(batch, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt")
        input_ids.append(encodings['input_ids'])
        attention_masks.append(encodings['attention_mask'])

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

def evaluate_model(model, tokenizer, texts, labels, device, batch_size=32):
    model.eval()
    model.to(device)
    input_ids, attention_masks = tokenize_data(tokenizer, texts, batch_size)
    labels = torch.tensor(labels).to(device)

    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)

    preds, true_labels = [], []

    for batch in dataloader:
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_input_mask = b_input_mask.to(device)

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits

        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(b_labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data = load_data(args.data_path)

    # Filter test data
    test_data = data[data['split'] == 'test']
    test_texts = test_data['text'].astype(str).tolist()
    test_labels = test_data['label'].tolist()

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir)

    # Evaluate model
    accuracy, precision, recall, f1 = evaluate_model(model, tokenizer, test_texts, test_labels, device)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained BERT model for sentiment analysis")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the evaluation data')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory of the trained model and tokenizer')

    args = parser.parse_args()
    main(args)
