import pandas as pd
import joblib 

def load_trained_model(model_path):
    model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")
    return model

def make_prediction(model, data):
   
    predictions = model.predict(data)
    return predictions
