

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Load the dataset
df = pd.read_csv('data.csv')

# Quick EDA
print("Dataset shape:", df.shape)
print(df.head())
print("Missing values per column:")
print(df.isnull().sum())

# Drop rows with missing values if any
df.dropna(inplace=True)

# FEATURE ENGINEERING
# Convert 'date' to datetime and extract useful time-based features
df['date'] = pd.to_datetime(df['date'])
df['year_sold'] = df['date'].dt.year
df['month_sold'] = df['date'].dt.month

# Compute house age and renovation-related features
df['age'] = df['year_sold'] - df['yr_built']
df['years_since_renovation'] = df['year_sold'] - df['yr_renovated']
# If no renovation (year 0), set years_since_renovation to 0
df.loc[df['yr_renovated'] == 0, 'years_since_renovation'] = 0
df['is_renovated'] = (df['yr_renovated'] > 0).astype(int)

# Drop columns that we won't use (string/address info, etc.)
df.drop(['date','yr_built','yr_renovated','street','city','statezip','country'], axis=1, inplace=True)

# Separate features and target
X = df.drop('price', axis=1)
y = df['price']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for neural network (tree models don't require scaling but NN does)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
rf = RandomForestRegressor(random_state=42)
gbr = GradientBoostingRegressor(random_state=42)
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
lgb_model = lgb.LGBMRegressor(random_state=42)

# Hyperparameter grids for tuning
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20]
}
param_grid_gbr = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1]
}
param_grid_lgb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}

# Perform Grid Search with 3-fold CV for each model
print("Tuning RandomForestRegressor...")
grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, n_jobs=-1, scoring='r2')
grid_rf.fit(X_train, y_train)
print("Best RF params:", grid_rf.best_params_)

print("Tuning GradientBoostingRegressor...")
grid_gbr = GridSearchCV(gbr, param_grid_gbr, cv=3, n_jobs=-1, scoring='r2')
grid_gbr.fit(X_train, y_train)
print("Best GBR params:", grid_gbr.best_params_)

print("Tuning XGBRegressor...")
grid_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=3, n_jobs=-1, scoring='r2')
grid_xgb.fit(X_train, y_train)
print("Best XGB params:", grid_xgb.best_params_)

print("Tuning LGBMRegressor...")
grid_lgb = GridSearchCV(lgb_model, param_grid_lgb, cv=3, n_jobs=-1, scoring='r2')
grid_lgb.fit(X_train, y_train)
print("Best LGBM params:", grid_lgb.best_params_)

# Evaluate each tuned model on the test set
models = {
    "Random Forest": grid_rf.best_estimator_,
    "Gradient Boosting": grid_gbr.best_estimator_,
    "XGBoost": grid_xgb.best_estimator_,
    "LightGBM": grid_lgb.best_estimator_
}
best_model_name = None
best_r2 = -np.inf
best_mae = None

for name, model in models.items():
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name}: MAE = {mae:.2f}, R^2 = {r2:.4f}")
    if r2 > best_r2:
        best_r2 = r2
        best_mae = mae
        best_model_name = name

print(f"Best model: {best_model_name} (MAE = {best_mae:.2f}, R^2 = {best_r2:.4f})")

best_model = models[best_model_name]
best_preds = best_model.predict(X_test)
print("Actual vs Predicted (first 5):")
for actual, pred in zip(y_test.values[:5], best_preds[:5]):
    print(f"{actual} -> {pred}")

# Optional: Neural Network model (simple MLP for comparison)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

mlp = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])
mlp.compile(optimizer='adam', loss='mse')
mlp.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
mlp_preds = mlp.predict(X_test_scaled).flatten()
mae_mlp = mean_absolute_error(y_test, mlp_preds)
r2_mlp = r2_score(y_test, mlp_preds)
print(f"Neural Network: MAE = {mae_mlp:.2f}, R^2 = {r2_mlp:.4f}")
