from flask import Flask, request, jsonify
import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))
app = Flask(__name__)

@app.route('/')
def home():
    return "MPG Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array([data['horsepower'], data['weight'], data['acceleration']]).reshape(1, -1)
    prediction = model.predict(input_data)
    return jsonify({"mpg_prediction": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
