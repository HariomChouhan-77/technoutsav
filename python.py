import pandas as pd

# Replace 'yourfilename.csv' with your actual CSV file name
data = pd.read_csv(r'C:\Users\hario\Downloads\archive\TrafficTwoMonth.csv')

# Then continue preprocessing as per earlier instructions


# Check columns and sample data
print(data.columns)
print(data.head())

# Inspect 'Date' column original data
print(data['Date'].head())
print(data['Date'].unique())
# Drop NaN rows from Date column before conversion
data = data.dropna(subset=['Date'])

# Now convert to int safely
data['day'] = data['Date'].round().astype(int)
data['Date'] = data['Date'].fillna(1)  # Filling NaN with 1 as an example
data['day'] = data['Date'].round().astype(int)

data['datetime_str'] = '2025-09-' + data['day'].astype(str) + ' ' + data['Time']
data['datetime'] = pd.to_datetime(data['datetime_str'], format='%Y-%m-%d %I:%M:%S %p', errors='coerce')
data = data.dropna(subset=['datetime'])


# Since 'Date' appears to be day of month or day number, create a proper datetime assuming data is from September 2025
# Convert 'Date' to int (round floats safely)
data['day'] = data['Date'].round().astype(int)

# Combine into a datetime string: "Year-Month-Day Time"
data['datetime_str'] = '2025-09-' + data['day'].astype(str) + ' ' + data['Time']

# Convert to proper datetime format, explicitly specifying format to avoid warnings
data['datetime'] = pd.to_datetime(data['datetime_str'], format='%Y-%m-%d %I:%M:%S %p', errors='coerce')

# Drop rows which failed to convert (NaT datetime)
data = data.dropna(subset=['datetime'])

# Extract more datetime features useful for ML
data['hour'] = data['datetime'].dt.hour
data['day_of_week_num'] = data['datetime'].dt.dayofweek  # Monday=0, Sunday=6

# Print the updated dataframe sample
print(data[['datetime', 'hour', 'day_of_week_num']].head())

# Check info after cleaning
print(data.info())

# You can now proceed with feature selection, encoding categorical variables, train-test split, and model training
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Encode categorical features using OneHotEncoder
categorical_features = ['Day of the week', 'Traffic Situation']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
 
# Fit and transform categorical columns
encoded_cat = encoder.fit_transform(data[categorical_features])

# Get encoded feature names, concatenate to dataframe
encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_features))

# Concatenate encoded features with numeric ones
data_model = pd.concat([data.reset_index(drop=True), encoded_cat_df], axis=1)

# Define input features and target
feature_columns = ['hour', 'day_of_week_num', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount'] + list(encoded_cat_df.columns)
X = data_model[feature_columns]
y = data_model['Total']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae:.2f}')
print(f'Mean Squared Error: {mse:.2f}')
print(f'R² Score: {r2:.2f}')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

# Your dataframe is named data

# Encode categorical features
categorical_features = ['Day of the week', 'Traffic Situation']
import sklearn
version = sklearn.__version__

if version >= "1.2":
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
else:
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_cat = encoder.fit_transform(data[categorical_features])
encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_features))

# Combine encoded categorical with numeric features
data_model = pd.concat([data.reset_index(drop=True), encoded_cat_df], axis=1)

# Features and target
features = ['hour', 'day_of_week_num', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount'] + list(encoded_cat_df.columns)
X = data_model[features]
y = data_model['Total']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f'MAE: {mean_absolute_error(y_test, y_pred):.2f}')
print(f'MSE: {mean_squared_error(y_test, y_pred):.2f}')
print(f'R2 Score: {r2_score(y_test, y_pred):.2f}')

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
import sklearn
print(sklearn.__version__)
import sklearn
version = sklearn.__version__

if version >= "1.2":
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
else:
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

import matplotlib.pyplot as plt

# Plot actual vs predicted traffic volumes
plt.figure(figsize=(10,6))
plt.plot(y_test.values[:100], label='Actual', marker='o')
plt.plot(y_pred[:100], label='Predicted', marker='x')
plt.title('Actual vs Predicted Traffic Volume (first 100 samples)')
plt.xlabel('Sample Index')
plt.ylabel('Traffic Volume (Total)')
plt.legend()
plt.show()
import matplotlib.pyplot as plt

# Plot actual vs predicted traffic volumes (first 100 samples)
plt.figure(figsize=(10,6))
plt.plot(y_test.values[:100], label='Actual', marker='o')
plt.plot(y_pred[:100], label='Predicted', marker='x')
plt.title('Actual vs Predicted Traffic Volume (first 100 samples)')
plt.xlabel('Sample Index')
plt.ylabel('Traffic Volume (Total)')
plt.legend()
plt.show()

import matplotlib.pyplot as plt
importances = model.feature_importances_
feature_names = feature_columns

plt.figure(figsize=(10,6))
plt.barh(feature_names, importances)
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.show()
import matplotlib.pyplot as plt

# Get feature importances from the model
importances = model.feature_importances_
feature_names = feature_columns

# Plot horizontal bar chart
plt.figure(figsize=(10,6))
plt.barh(feature_names, importances)
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Define parameter grid to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Initialize RandomForestRegressor
rf = RandomForestRegressor(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=3, n_jobs=-1, scoring='neg_mean_absolute_error')

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters:", grid_search.best_params_)

# Best estimator
best_rf = grid_search.best_estimator_

# Evaluate best model on test set
y_pred_best = best_rf.predict(X_test)
print(f'Best Model MAE: {mean_absolute_error(y_test, y_pred_best):.2f}')
print(f'Best Model MSE: {mean_squared_error(y_test, y_pred_best):.2f}')
print(f'Best Model R2: {r2_score(y_test, y_pred_best):.2f}')
import joblib

# Save the best model to a file
joblib_file = 'best_traffic_rf_model.pkl'
joblib.dump(best_rf, joblib_file)

print(f'Model saved to {joblib_file}')
loaded_model = joblib.load(joblib_file)
# Use loaded_model to predict, evaluate, or deploy
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
import joblib
import pandas as pd

# Example data dictionary matching your feature columns
sample_data = {
    'hour': [10],
    'day_of_week_num': [2],
    'CarCount': [15],
    'BikeCount': [5],
    'BusCount': [2],
    'TruckCount': [1],
    # plus all OneHotEncoded categorical columns, e.g. 'Day of the week_Tuesday': [1], 'Traffic Situation_normal': [1]
}

# Create DataFrame from sample data
new_data_features = pd.DataFrame(sample_data)

# Then:
predictions = loaded_model.predict(new_data_features)
print(predictions)

loaded_model = joblib.load('best_traffic_rf_model.pkl')
new_data_features = ...  # preprocessed new data
predictions = loaded_model.predict(new_data_features)
print(predictions)
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder

# Load saved model
loaded_model = joblib.load('best_traffic_rf_model.pkl')

# Example new raw input (replace with your actual new data)
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

# Load or recreate the same OneHotEncoder used during training
# For example, refit encoder on training categorical data if you saved it or recreate it
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit encoder on categories from training data (example categories list)
encoder.fit([
    ['Monday', 'normal'],
    ['Tuesday', 'normal'],
    ['Wednesday', 'normal'],
    ['Thursday', 'normal'],
    ['Friday', 'normal'],
    ['Saturday', 'normal'],
    ['Sunday', 'normal'],
    ['Monday', 'heavy'],
    ['Tuesday', 'heavy'],
    ['Wednesday', 'heavy'],
    ['Thursday', 'heavy'],
    ['Friday', 'heavy'],
    ['Saturday', 'heavy'],
    ['Sunday', 'heavy'],
])

# Transform new categorical data
encoded_cat = encoder.transform(new_raw_data[['Day of the week', 'Traffic Situation']])

# Convert to DataFrame with proper column names
encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(['Day of the week', 'Traffic Situation']))

# Concatenate encoded features to numerical features
new_data_features = pd.concat([new_raw_data.reset_index(drop=True).drop(columns=['Day of the week', 'Traffic Situation']), encoded_cat_df], axis=1)

# Use the model to predict
predictions = loaded_model.predict(new_data_features)

print("Predicted traffic volume:", predictions)
import joblib
import pandas as pd

# Load saved model and encoder
loaded_model = joblib.load('best_traffic_rf_model.pkl')
loaded_encoder = joblib.load('traffic_encoder.pkl')

# Prepare new raw input data
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

# Encode categorical features with loaded encoder
encoded_cat = loaded_encoder.transform(new_raw_data[['Day of the week', 'Traffic Situation']])
encoded_cat_df = pd.DataFrame(encoded_cat, columns=loaded_encoder.get_feature_names_out())

# Combine with numeric features
new_data_features = pd.concat([new_raw_data.reset_index(drop=True).drop(columns=['Day of the week', 'Traffic Situation']), encoded_cat_df], axis=1)

# Predict
predictions = loaded_model.predict(new_data_features)
print(predictions)
import joblib
joblib.dump(encoder, 'traffic_encoder.pkl')
encoder = joblib.load('traffic_encoder.pkl')
encoded_cat = encoder.transform(new_raw_data[['Day of the week', 'Traffic Situation']])
encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out())

# Assuming encoder is your fitted OneHotEncoder
joblib.dump(encoder, 'traffic_encoder.pkl')
encoder = joblib.load('traffic_encoder.pkl')
loaded_model = joblib.load('best_traffic_rf_model.pkl')
encoder = joblib.load('traffic_encoder.pkl')
loaded_model = joblib.load('best_traffic_rf_model.pkl')








