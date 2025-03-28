from flask import Flask, render_template, request, jsonify
from app.models.logistic_regression import LogisticRegressionModel
from app.models.polynomial_regression import PolynomialRegressionModel
import os

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# Initialize models
logistic_model = LogisticRegressionModel()
polynomial_model = PolynomialRegressionModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/logistic')
def logistic():
    return render_template('logistic.html')

@app.route('/polynomial')
def polynomial():
    return render_template('polynomial.html')

@app.route('/predict_logistic', methods=['POST'])
def predict_logistic():
    data = request.json
    car_age = float(data['car_age'])
    mileage = float(data['mileage'])
    brand = data['brand']
    fuel_type = data['fuel_type']
    transmission = data['transmission']
    
    prediction = logistic_model.predict(car_age, mileage, brand, fuel_type, transmission)
    
    return jsonify({
        'prediction': prediction,
        'category': 'High' if prediction == 1 else 'Low'
    })

@app.route('/predict_polynomial', methods=['POST'])
def predict_polynomial():
    data = request.json
    car_age = float(data['car_age'])
    
    prediction = polynomial_model.predict(car_age)
    
    return jsonify({
        'prediction': prediction
    })

@app.route('/train_models', methods=['GET'])
def train_models():
    logistic_success = logistic_model.train()
    polynomial_success = polynomial_model.train()
    
    return jsonify({
        'logistic_success': logistic_success,
        'polynomial_success': polynomial_success
    })

if __name__ == '__main__':
    # Train models on startup
    logistic_model.train()
    polynomial_model.train()
    app.run(debug=True) 