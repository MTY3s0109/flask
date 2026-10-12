import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load saved artifacts
model = joblib.load('car_purchasing_model.h5')
le = joblib.load('le.pkl')
sc = joblib.load('sc.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract fields directly from form data (strings)
        country = request.form.get('country')
        gender = float(request.form.get('gender'))
        age = float(request.form.get('age'))
        annual_salary = float(request.form.get('annual_salary'))
        credit_card_debt = float(request.form.get('credit_card_debt'))
        net_worth = float(request.form.get('net_worth'))

        # Process input
        country_encoded = le.transform([country])[0]
        
        raw_features = np.array([[
            country_encoded, gender, age, 
            annual_salary, credit_card_debt, net_worth
        ]])
        
        scaled_features = sc.transform(raw_features)
        prediction = model.predict(scaled_features)[0]

        # Return prediction as standard text or JSON
        return jsonify({
            'status': 'success',
            'predicted_car_purchase_amount': float(prediction)
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)