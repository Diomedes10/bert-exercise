from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Movie Reviews Sentiment Analysis App!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data['review']
    # Your prediction logic here, e.g. loading the model and making a prediction
    prediction = 3  # Example prediction
    return jsonify({"review": review, "prediction": prediction})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7860)
