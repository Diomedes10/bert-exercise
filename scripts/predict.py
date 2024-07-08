import argparse
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the label dictionary
    label_dict_path = f"{args.model_dir}/label_dict.json"
    with open(label_dict_path, 'r') as f:
        label_dict = json.load(f)
    
    print(f"Loaded label dictionary: {label_dict}")

    # Create a reverse label dictionary
    sentiment_map = {v: k for k, v in label_dict.items()}
    print(f"Reversed label dictionary: {sentiment_map}")

    # Load the model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)

    # Tokenize the input text
    inputs = tokenizer(args.text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

    # Map prediction to label
    label = predictions[0]
    human_readable_label = sentiment_map.get(label, "Unknown")

    # Print the result
    print(f"Text: {args.text}")
    print(f"Prediction: {label} ({human_readable_label})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict sentiment of a given text using a trained BERT model")
    parser.add_argument('--text', type=str, required=True, help='Text to analyze')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory of the trained model and tokenizer')

    args = parser.parse_args()
    main(args)
