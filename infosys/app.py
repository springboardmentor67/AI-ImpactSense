from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = np.array([[
        data["magnitude"],
        data["depth"],
        data["cdi"],
        data["mmi"],
        data["sig"]
    ]])

    prediction = model.predict(features)[0]
    result = le.inverse_transform([prediction])[0]

    return jsonify({"alert": result})

if __name__ == "__main__":
    app.run(debug=True)
