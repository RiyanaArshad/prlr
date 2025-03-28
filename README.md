# Car Price Prediction System

A Flask-based web application that predicts car prices using two different regression models.

## Features

1. **Logistic Regression Model**
   - Classifies car prices as high or low
   - Uses multiple features: Car Age, Mileage, Brand, Fuel Type, and Transmission
   - Provides a simple classification result

2. **Polynomial Regression Model**
   - Predicts the exact car price
   - Uses only Car Age as the independent variable
   - Implements a degree 3 polynomial for better accuracy

## Project Structure

```
car_prediction/
├── app/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── logistic_regression.py
│   │   └── polynomial_regression.py
│   ├── static/
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── logistic.html
│   │   └── polynomial.html
├── data/
│   └── car_price_dataset.csv
├── app.py
├── README.md
└── requirements.txt
```

## Setup and Installation

1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the Flask application:
   ```
   python app.py
   ```

3. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## Models

- **Logistic Regression**: Implemented using scikit-learn's LogisticRegression class with standard scaling for numeric features and one-hot encoding for categorical features.
- **Polynomial Regression**: Implemented using scikit-learn's PolynomialFeatures (degree 3) with LinearRegression.

## Dataset

The dataset `car_price_dataset.csv` includes information about cars:
- Car Age (years)
- Mileage (kilometers)
- Brand (BMW, Ford, Honda, Mercedes, Toyota)
- Fuel Type (Petrol, Diesel, Electric, Hybrid)
- Transmission (Automatic, Manual)
- Price (numeric value)
- Price Category (1 for high, 0 for low)

## Usage

1. Navigate to the home page for an overview of the application
2. Choose either Logistic or Polynomial Regression model
3. Enter the required car details
4. Submit the form to get price predictions #   p r l r  
 