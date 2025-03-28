import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle
import os

class PolynomialRegressionModel:
    def __init__(self):
        self.model = None
        self.model_file = 'app/models/polynomial_model.pkl'
        
        # Load the model if it exists
        if os.path.exists(self.model_file):
            self.load_model()
    
    def train(self):
        try:
            # Load data
            data = pd.read_csv('data/car_price_dataset.csv')
            
            # Features and target
            X = data[['Car_Age']]
            y = data['Price']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create and train the model
            self.model = Pipeline([
                ('poly', PolynomialFeatures(degree=3)),
                ('linear', LinearRegression())
            ])
            
            self.model.fit(X_train, y_train)
            
            # Save the model
            self.save_model()
            
            return True
        except Exception as e:
            print(f"Error training polynomial regression model: {e}")
            return False
    
    def predict(self, car_age):
        if self.model is None:
            self.train()
        
        # Create input data
        input_data = np.array([[car_age]])
        
        # Make prediction
        prediction = self.model.predict(input_data)[0]
        
        # Round to 2 decimal places
        return round(float(prediction), 2)
    
    def save_model(self):
        pickle.dump(self.model, open(self.model_file, 'wb'))
        
    def load_model(self):
        try:
            self.model = pickle.load(open(self.model_file, 'rb'))
            return True
        except Exception as e:
            print(f"Error loading polynomial regression model: {e}")
            return False 