from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('SVC.pkl')
scaler = joblib.load('scalar.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        age = float(data['Age'])
        weight = float(data['Weight'])
        height = float(data['Height'])
        blood_group = int(data['Blood group'])
        periods_gaps = int(data['Periods gaps'])
        weight_gain = int(data['Weight gain'])
        facial_hair = int(data['Facial hair'])
        skin_darkening = int(data['Skin darkening'])
        hair_loss = int(data['Hair loss'])
        pimples = int(data['Pimples'])
        fast_food = int(data['Fast food'])
        exercise = int(data['Exercise'])
        mood_swings = int(data['Mood swings'])
        regular_periods = int(data['Regular'])
        period_length = int(data['Period length'])

        # Features
        features = np.array([[age, weight, height, blood_group, periods_gaps, weight_gain, facial_hair, skin_darkening, hair_loss, pimples, fast_food, exercise, mood_swings, regular_periods, period_length]])

        # Scale the features
        features_scaled = scaler.transform(features)

        # Make a prediction
        prediction = model.predict(features_scaled)[0]

        # Render the result
        return render_template('result.html', prediction=prediction)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
