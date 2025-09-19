# Traffic Volume Prediction using Random Forest Regression

## Project Overview
This project aims to predict traffic volume (total vehicle counts) based on time-series traffic data collected over several days. The model uses features such as time, day of the week, counts of different vehicle types, and traffic conditions to forecast total traffic.

## Dataset Description
- Date and time-stamped traffic data
- Vehicle counts: Cars, Bikes, Buses, Trucks
- Traffic situation labels (e.g., normal, heavy)
- Data cleaned and preprocessed to handle missing values and extract meaningful datetime features

## Methodology
- Preprocessing:
  - Combined 'Date' and 'Time' into datetime features
  - Extracted ‘hour’ and ‘day of week’ as numerical features
  - Encoded categorical variables with OneHotEncoder
- Modeling:
  - Trained a Random Forest Regressor
  - Hyperparameter tuning conducted using GridSearchCV to optimize model parameters
- Evaluation:
  - Model evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score
  - Achieved R² close to 1, indicating excellent fit to the data

## How to Run
1. Clone the repository
2. Install dependencies:
3. Execute the main script:
4. Model training, evaluation, and saving are done automatically. Visualizations will display if matplotlib is available.

## File Structure
- `python.py`: Main script for data processing, model training, tuning, and evaluation
- `best_traffic_rf_model.pkl`: Saved Random Forest model after training (generated after running the script)

## Future Work
- Deployment of model as a web API for live traffic predictions
- Explore advanced machine learning models such as XGBoost or deep learning approaches
- Perform detailed residual analysis and visualize prediction errors
- Integrate additional features like weather, special events, and holidays
- Improve user interface for model interaction
##requiement
pandas>=1.0.0
scikit-learn>=1.7.2
joblib>=1.2.0
matplotlib>=3.0.0
numpy>=1.20.0


## Contact
For questions or collaboration opportunities, please reach out through GitHub issues or contact hariomchouhan932@gmail.com
