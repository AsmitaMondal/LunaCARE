import pandas as pd
import joblib
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template, request, redirect, url_for,send_from_directory
import numpy as np
from flask import flash, redirect
import google.generativeai as genai
from flask import jsonify

app = Flask(__name__)

GOOGLE_API_KEY = 'key'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/Lunacare'
db = SQLAlchemy(app)

# Load the model and scaler
model = joblib.load('gradient_boosting_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define your Testimonials model for the database
class Testimonial(db.Model):
    __tablename__ = 'testimonials'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

class PCOSForm(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    cycle = db.Column(db.Float, nullable=False)
    weight_gain = db.Column(db.Boolean, nullable=False)
    hair_growth = db.Column(db.Boolean, nullable=False)
    skin_darkening = db.Column(db.Boolean, nullable=False)
    pimples = db.Column(db.Boolean, nullable=False)
    fast_food = db.Column(db.Boolean, nullable=False)
    follicle_no_l = db.Column(db.Integer, nullable=False)
    follicle_no_r = db.Column(db.Integer, nullable=False)
    avg_f_size_l = db.Column(db.Float, nullable=False)
    avg_f_size_r = db.Column(db.Float, nullable=False)
    tsh = db.Column(db.Float, nullable=False)
    prl = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    fsh_lh_ratio = db.Column(db.Float, nullable=False)
    rbs = db.Column(db.Float, nullable=False)
    result = db.Column(db.String(255), nullable=False)
    
# Load your data
df = pd.read_csv('Final_Amazon.csv')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/periodtrack')
def track():
    return render_template('Period.html')

@app.route('/pcos')
def pcoss():
    return render_template('pcos.html')

@app.route('/blogs')
def blogging():
    return render_template('blog.html')

@app.route('/blog1')
def blogging1():
    return render_template('Blogpage.html')

@app.route('/blog2')
def blogging2():
    return render_template('Blogpage2.html')

@app.route('/blog3')
def blogging3():
    return render_template('Blogpage3.html')

@app.route('/blog4')
def blogging4():
    return render_template('Blogpage1.html')

@app.route('/tableau')
def blogging5():
    return render_template('tableau.html')

@app.route('/productc')
def pro():
    return render_template('pc.html')

@app.route('/testimonials')
def tests():
    return render_template('testimonials.html')

@app.route('/compare', methods=['POST'])
def compare_products():
    asin1 = request.form.get('asin1')
    asin2 = request.form.get('asin2')
    
    # Filter the DataFrame based on ASINs
    product1_df = df[df['ASIN'] == asin1]
    product2_df = df[df['ASIN'] == asin2]
    
    product1 = product1_df.iloc[0].to_dict() if not product1_df.empty else None
    product2 = product2_df.iloc[0].to_dict() if not product2_df.empty else None
    
    return render_template('compare.html', product1=product1, product2=product2)

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

    # Store the form data and result in the database
    new_entry = PCOSForm(
        cycle=data['Cycle'],
        weight_gain=bool(data['Weight_gain']),
        hair_growth=bool(data['Hair_growth']),
        skin_darkening=bool(data['Skin_darkening']),
        pimples=bool(data['Pimples']),
        fast_food=bool(data['Fast_food']),
        follicle_no_l=data['Follicle_no_L'],
        follicle_no_r=data['Follicle_no_R'],
        avg_f_size_l=data['Avg_F_size_L'],
        avg_f_size_r=data['Avg_F_size_R'],
        tsh=data['TSH'],
        prl=data['PRL'],
        bmi=data['BMI'],
        fsh_lh_ratio=data['FSH_LH_Ratio'],
        rbs=data['RBS'],
        result=result
    )

    db.session.add(new_entry)
    db.session.commit()

    return render_template('pcos_result.html', result=result)

@app.route('/formsubmit', methods=['POST'])
def formsubmit():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        rating = int(request.form.get('rating'))
        message = request.form.get('message')

        # Create a new testimonial entry
        new_testimonial = Testimonial(
            name=name,
            email=email,
            rating=rating,
            message=message
        )

        # Add to the session and commit
        db.session.add(new_testimonial)
        db.session.commit()

        alert_message = "Review Submitted Successfully!"

        return render_template('index.html', alert_message=alert_message)

    # return render_template('index.html', alert_message=None)

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/periods')
def periods():
    return send_from_directory('templates', 'periods (1).json')



if __name__ == '__main__':
    app.run(debug=True)
