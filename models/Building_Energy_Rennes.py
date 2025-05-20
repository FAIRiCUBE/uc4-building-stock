import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Define database and table
sqlite_file_path = "Rennes_buildings.sqlite"
table_name = "rennes_building_shared_wall_v2"

# Connect to the database and load the table into a DataFrame
conn = sqlite3.connect(sqlite_file_path)
query = f"SELECT * FROM {table_name};"
df = pd.read_sql(query, conn)
conn.close()
df['energy_calculation_year'] = pd.to_datetime(df['date_etablissement_dpe']).dt.year
df = df[df['energy_calculation_year'] == 2020]
# Compute total energy demand (space heating + domestic hot water)
df["total_energy_demand_kWh/m2"] = (df["consommation_energie_finale_1.0"] + df["consommation_energie_finale_2.0"])/df["surface_thermique_lot"]
print(df.columns)
print(df['date_etablissement_dpe'].unique())

# Define relevant features based on prior selection
# features = [
#     "surface_habitable", "surface_thermique_lot", "surface_commerciale_contractuelle",
#     "shon", "nombre_niveaux", "surface_baies_orientees_nord", "surface_baies_orientees_sud",
#     "surface_baies_orientees_est_ouest", "surface_planchers_hauts_deperditifs",
#     "surface_parois_verticales_opaques_deperditives", "tr002_type_batiment_id",
#     "tr013_type_erp_id", "annee_construction", "portee_dpe_batiment", "dpe_vierge",
#     "latitude", "longitude"
# ]

features = [
    "nombre_niveaux", "surface_parois_verticales_opaques_deperditives", "annee_construction"
]

target = "total_energy_demand_kWh/m2"  # Using the new total energy demand column

# Keep only necessary columns and drop missing values
df = df[features + [target]].dropna()

print(df[target])
# Identify categorical and numerical features
#categorical_features = ["tr002_type_batiment_id", "tr013_type_erp_id", "dpe_vierge"]
numerical_features = list(set(features) )#- set(categorical_features))

# Preprocessing pipeline: One-hot encoding for categorical, scaling for numerical
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    #("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Define ML model (Gradient Boosting Regressor)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", xgb.XGBRegressor(objective="reg:squarederror", n_estimators=125, learning_rate=0.15, max_depth=5, random_state=42))
])

# features = [
#     "nombre_niveaux", "surface_parois_verticales_opaques_deperditives", "annee_construction"
# ]
print(f"LENTH = {len(df)}")
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

# Print evaluation results
print(f"Model Performance:\nRoot Mean Squared Error (RMSE): {rmse:.2f}\nR² Score: {r2:.2f}")

# Scatter plot
def plot_scatter(y_test, y_pred, model_name="XGBoost", year=2020):
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.3, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
    plt.xlabel("French OpenGov")
    plt.ylabel("XGRegressor")
    plt.title(f"Year {year}")

    # Add RMSE and R2 score as text
    plt.text(y_test.min() + 10, y_test.max() - 10, f"RMSE: {rmse:.2f}\nR²: {r2:.2f}",
             fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    plt.savefig('Rennes_2020_article.png')
    plt.show()



plot_scatter(y_test, y_pred)
