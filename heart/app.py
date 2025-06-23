from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Main Gallery Page

@app.route('/predict', methods=['POST'])
def predict():
    result = None
    if request.method == 'POST':
        data = request.form
        
        # Extract user inputs
        age = int(data['Age'])
        resting_bp = int(data['RestingBP'])
        cholesterol = int(data['Cholesterol'])
        fasting_bs = int(data['FastingBS'])
        max_hr = int(data['MaxHR'])
        oldpeak = float(data['Oldpeak'])

        # Encoding categorical features
        sex_male = 1 if data['Sex'] == 'M' else 0
        exercise_angina = 1 if data['ExerciseAngina'] == 'Y' else 0

        # One-hot encoding for categorical variables
        chest_pain_types = ['ATA', 'NAP', 'ASY', 'TA']
        resting_ecg_types = ['Normal', 'ST', 'LVH']
        st_slope_types = ['Up', 'Flat', 'Down']

        # Create encoded feature vectors
        chest_pain = [1 if data['ChestPainType'] == c else 0 for c in chest_pain_types[1:]]
        resting_ecg = [1 if data['RestingECG'] == r else 0 for r in resting_ecg_types[1:]]
        st_slope = [1 if data['ST_Slope'] == s else 0 for s in st_slope_types[1:]]

        # Combine all features into a single array
        input_data = np.array([
            age, resting_bp, cholesterol, fasting_bs, max_hr, oldpeak, sex_male, 
            exercise_angina, *chest_pain, *resting_ecg, *st_slope
        ]).reshape(1, -1)

        # Standardize input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        result = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
