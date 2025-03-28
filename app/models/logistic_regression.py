import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle
import os

class LogisticRegressionModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.model_file = 'app/models/logistic_model.pkl'
        self.preprocessor_file = 'app/models/logistic_preprocessor.pkl'
        
        # Load the model if it exists
        if os.path.exists(self.model_file) and os.path.exists(self.preprocessor_file):
            self.load_model()
    
    def train(self):
        try:
            # Load data
            data = pd.read_csv('data/car_price_dataset.csv')
            
            # Features and target
            X = data.drop(['Price', 'Price_Category'], axis=1)
            y = data['Price_Category']
            
            # Create preprocessor
            numeric_features = ['Car_Age', 'Mileage']
            categorical_features = ['Brand', 'Fuel_Type', 'Transmission']
            
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create and train the model
            self.model = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', LogisticRegression(random_state=42))
            ])
            
            self.model.fit(X_train, y_train)
            
            # Save the model and preprocessor
            self.save_model()
            
            return True
        except Exception as e:
            print(f"Error training logistic regression model: {e}")
            return False
    
    def predict(self, car_age, mileage, brand, fuel_type, transmission):
        if self.model is None:
            self.train()
        
        # Create input data
        input_data = pd.DataFrame({
            'Car_Age': [car_age],
            'Mileage': [mileage],
            'Brand': [brand],
            'Fuel_Type': [fuel_type],
            'Transmission': [transmission]
        })
        
        # Make prediction
        prediction = self.model.predict(input_data)[0]
        
        return int(prediction)
    
    def save_model(self):
        pickle.dump(self.model, open(self.model_file, 'wb'))
        
    def load_model(self):
        try:
            self.model = pickle.load(open(self.model_file, 'rb'))
            return True
        except Exception as e:
            print(f"Error loading logistic regression model: {e}")
            return False 