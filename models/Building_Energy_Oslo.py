import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import numpy as np

# Load dataset
file_path = "oslo_building.csv"  # Change this to your local file path
df = pd.read_csv(file_path)

# Define features and target variable
features = ['height', 'construction_year', 'shared_wall_area']
target = 'energy_performance_kWh-per-sqm'

# Get unique energy calculation years
years = df['energy_calculation_year'].unique()

# Store results for each year
results = []

for year in years:
    df_year = df[df['energy_calculation_year'] == year][features + [target]].dropna()

    if df_year.shape[0] < 50:  # Skip if too few data points
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        df_year[features], df_year[target], test_size=0.2, random_state=42
    )

    # Initialize and train XGBoost model
    xgb_reg = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.05, max_depth=6,
                               random_state=42)
    xgb_reg.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = xgb_reg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append({'Year': year, 'MAE': mae, 'RMSE': rmse, 'R2': r2})

    print(f"Year: {year}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R² Score: {r2}\n")

# Convert results to DataFrame and display
results_df = pd.DataFrame(results)
print(results_df)
