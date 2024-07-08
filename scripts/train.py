import argparse
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

def load_data(filepath):
    print(f"Loading data from {filepath}")
    data = pd.read_csv(filepath)
    print(f"Data loaded successfully. Columns: {data.columns}")
    return data

def tokenize_data(tokenizer, texts):
    print(f"Tokenizing data with {len(texts)} texts.")
    return tokenizer(texts, truncation=True, padding=True, max_length=128)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data = load_data(args.data_path)

    # Split data
    train_data = data[data['split'] == 'train']
    val_data = data[data['split'] == 'dev']
    test_data = data[data['split'] == 'test']

    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # Prepare texts and labels
    train_texts = train_data['text'].astype(str).tolist()
    train_labels = train_data['label'].tolist()
    val_texts = val_data['text'].astype(str).tolist()
    val_labels = val_data['label'].tolist()

    print(f"Number of train texts: {len(train_texts)}")
    print(f"Number of train labels: {len(train_labels)}")
    print(f"Number of val texts: {len(val_texts)}")
    print(f"Number of val labels: {len(val_labels)}")

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize data
    train_encodings = tokenize_data(tokenizer, train_texts)
    val_encodings = tokenize_data(tokenizer, val_texts)

    # Convert labels to tensor
    train_labels = torch.tensor(train_labels)
    val_labels = torch.tensor(val_labels)

    # Create dataset objects
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels
    })

    val_dataset = Dataset.from_dict({
        'input_ids': val_encodings['input_ids'],
        'attention_mask': val_encodings['attention_mask'],
        'labels': val_labels
    })

    # Load model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Train model
    trainer.train()

    # Save model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a BERT model for sentiment analysis")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the processed data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the trained model and tokenizer')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Number of warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay rate')
    parser.add_argument('--logging_dir', type=str, default='./logs', help='Directory for logging')
    parser.add_argument('--logging_steps', type=int, default=10, help='Log every X updates steps')

    args = parser.parse_args()
    main(args)

