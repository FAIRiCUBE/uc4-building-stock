import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load dataset
file_path = "Barcelona_buildings.csv"  # Change this to your local file path
df = pd.read_csv(file_path)
df = df[df['year'] == 2020]

# Selecting features and target variable
features = ['height', 'age', 'shared_wall']
target = 'consum_ene'

df_model = df[features + [target]].dropna()

print(df_model.head())
# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df_model[features], df_model[target], test_size=0.2, random_state=42
)

# Initializing and training the XGBoost regressor
xgb_reg = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.04, max_depth=7, random_state=42)
xgb_reg.fit(X_train, y_train)

# Predicting on the test set
#y_pred = xgb_reg.predict(X_test)
y = df_model['consum_ene']
y_pred = xgb_reg.predict(df_model[features])
# Evaluating model performance
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")

# Scatter plot
def plot_scatter(y_test, y_pred, model_name="XGBoost", year=2020):
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.3, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
    plt.xlabel("Energia Barcelona")
    plt.ylabel("XGRegressor")
    plt.title(f"Year {year}")

    # Add RMSE and R2 score as text
    plt.text(y_test.min() + 10, y_test.max() - 10, f"RMSE: {rmse:.2f}\nR²: {r2:.2f}",
             fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    plt.savefig('Barcelona_2020_article_allData.png')
    plt.show()



plot_scatter(y, y_pred)
