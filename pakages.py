import joblib
import pandas as pd

# Load model and encoder once when module imported
model = joblib.load('best_traffic_rf_model.pkl')
encoder = joblib.load('traffic_encoder.pkl')

def preprocess_input(raw_data: pd.DataFrame) -> pd.DataFrame:
    # Encode categorical columns
    encoded_cat = encoder.transform(raw_data[['Day of the week', 'Traffic Situation']])
    encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out())
    # Combine with numeric columns
    features = pd.concat([raw_data.drop(columns=['Day of the week', 'Traffic Situation']).reset_index(drop=True), encoded_cat_df], axis=1)
    return features

def predict(raw_data: pd.DataFrame) -> pd.Series:
    X = preprocess_input(raw_data)
    preds = model.predict(X)
    return preds
from setuptools import setup, find_packages

setup(
    name='traffic_predictor',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'joblib'
    ],
    author='Your Name',
    description='Traffic Volume Prediction Model Package',
    url='https://github.com/yourusername/traffic_predictor',
)
import pandas as pd
from traffic_predictor.model import predict

# Create example raw data as dataframe
new_raw_data = pd.DataFrame({
    'hour': [15],
    'day_of_week_num': [4],
    'CarCount': [20],
    'BikeCount': [10],
    'BusCount': [3],
    'TruckCount': [2],
    'Day of the week': ['Friday'],
    'Traffic Situation': ['normal']
})

predictions = predict(new_raw_data)
print(predictions)
