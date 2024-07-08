import os
from flask import Flask, request, jsonify, send_from_directory
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

app = Flask(__name__)

# Load model and tokenizer
model_dir = './results'
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)
model.eval()

# Load the label dictionary
with open(f"{model_dir}/label_dict.json", 'r') as f:
    label_dict = json.load(f)
sentiment_map = {v: k for k, v in label_dict.items()}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text')
    title = data.get('title')

    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]
        label = sentiment_map.get(prediction, "Unknown")

    sentiment_terms = {
        0: "Very Negative",
        1: "Negative",
        2: "Neutral",
        3: "Positive",
        4: "Very Positive"
    }

    return jsonify({
        'text': text,
        'title': title,
        'label': label,
        'term': sentiment_terms.get(prediction, "Unknown")
    })

@app.route('/')
def serve_frontend():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
