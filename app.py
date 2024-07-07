from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model, encoders, and scaler
with open('model.pkl', 'rb') as file:
    model, label_encoders, scaler = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        disease = request.form['disease']
        fever = request.form['fever']
        cough = request.form['cough']
        fatigue = request.form['fatigue']
        difficulty_breathing = request.form['difficulty_breathing']
        age = int(request.form['age'])
        gender = request.form['gender']
        blood_pressure = request.form['blood_pressure']
        cholesterol_level = request.form['cholesterol_level']

        # Encode and scale input data
        input_data = [
            label_encoders['Disease'].transform([disease])[0],
            label_encoders['Fever'].transform([fever])[0],
            label_encoders['Cough'].transform([cough])[0],
            label_encoders['Fatigue'].transform([fatigue])[0],
            label_encoders['Difficulty Breathing'].transform([difficulty_breathing])[0],
            age,
            label_encoders['Gender'].transform([gender])[0],
            label_encoders['Blood Pressure'].transform([blood_pressure])[0],
            label_encoders['Cholesterol Level'].transform([cholesterol_level])[0],
        ]
        input_data = scaler.transform([input_data])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_label = label_encoders['Outcome Variable'].inverse_transform([prediction])[0]

        return render_template('index.html', prediction=prediction_label)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
