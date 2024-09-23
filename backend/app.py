from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('gradient_boosting_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    data = {
        'Cycle': float(request.form['Cycle']),
        'Weight_gain': float(request.form['Weight_gain']),
        'Hair_growth': float(request.form['Hair_growth']),
        'Skin_darkening': float(request.form['Skin_darkening']),
        'Pimples': float(request.form['Pimples']),
        'Fast_food': float(request.form['Fast_food']),
        'Follicle_no_L': float(request.form['Follicle_no_L']),
        'Follicle_no_R': float(request.form['Follicle_no_R']),
        'Avg_F_size_L': float(request.form['Avg_F_size_L']),
        'Avg_F_size_R': float(request.form['Avg_F_size_R']),
        'TSH': float(request.form['TSH']),
        'PRL': float(request.form['PRL']),
        'BMI': float(request.form['BMI']),
        'FSH_LH_Ratio': float(request.form['FSH_LH_Ratio']),
        'RBS': float(request.form['RBS'])
    }

    # Prepare the feature array
    features = np.array([[data['Cycle'], data['Weight_gain'], data['Hair_growth'],
                          data['Skin_darkening'], data['Pimples'], data['Fast_food'],
                          data['Follicle_no_L'], data['Follicle_no_R'], data['Avg_F_size_L'],
                          data['Avg_F_size_R'], data['TSH'], data['PRL'], data['BMI'],
                          data['FSH_LH_Ratio'], data['RBS']]])

    # Scale the features
    features_scaled = scaler.transform(features)

    # Predict using the model
    prediction = model.predict(features_scaled)[0]

    # Map prediction to readable result
    result = "The prediction indicates that you might have PCOS. Please consult with your doctor for further advice." if prediction == 1 else "The prediction indicates that you might not have PCOS. However, please consult with your doctor for confirmation."

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
