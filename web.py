from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

model_path = '/hacathon.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)
scaler = StandardScaler()

@app.route('/')
def home():
    return "Welcome to the Machine Failure Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400
    try:
        input_data = pd.DataFrame([data])
        features = ['Rotational speed [rpm]', 'TWF', 'HDF']
        input_data[features] = scaler.fit_transform(input_data[features])
        prediction = model.predict(input_data[features])
        result = {"Machine failure prediction": int(prediction[0])}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main':
    app.run(debug=True)