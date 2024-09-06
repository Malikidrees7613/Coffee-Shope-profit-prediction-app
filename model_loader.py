import pandas as pd
import joblib 

def load_trained_model():
    model = joblib.load("Coffee shope prediction Model.keras")
    print(f"Model loaded successfully")
    return model

def make_prediction(model, data):
   
    predictions = model.predict(data)
    return predictions