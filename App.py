from flask import Flask, render_template, request
import pickle
import numpy as np

# Load trained model & scaler
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

app = Flask(__name__)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    try:
        data = [float(request.form[field]) for field in [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]]
        data = np.array(data).reshape(1, -1)
        data = scaler.transform(data)  # Normalize input

        # Predict risk
        prediction = model.predict(data)[0]
        risk = "High Risk of Heart Disease" if prediction == 1 else "Low Risk"

        # Provide recommendations
        recommendations = "Exercise regularly, maintain a healthy diet, and avoid smoking." if prediction == 1 else "Maintain a healthy lifestyle."

        return render_template("index.html", prediction=risk, recommendations=recommendations)

    except:
        return render_template("index.html", error="Invalid Input! Please enter valid numbers.")

if __name__ == "__main__":
    app.run(debug=True)
