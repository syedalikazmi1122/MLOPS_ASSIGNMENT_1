import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from google.colab import files

#  Load dataset
df = pd.read_csv("/content/realtor-data.zip.csv")

#  Clean data
df.dropna(inplace=True)

#  Encode categorical variables
label_cols = ['status', 'city', 'state']
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

#  Feature engineering
df['log_price'] = np.log1p(df['price'])
df['log_house_size'] = np.log1p(df['house_size'])
df['log_acre_lot'] = np.log1p(df['acre_lot'])
df['bed_bath'] = df['bed'] * df['bath']
df['zip_status'] = df['zip_code'] * df['status']

features = [
    'status', 'bed', 'bath', 'city', 'state', 'zip_code',
    'log_house_size', 'log_acre_lot', 'bed_bath', 'zip_status'
]
target = 'log_price'

X = df[features]
y = df[target]

#  Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  Train LightGBM Regressor with early stopping
print(" Training LightGBM Regressor with early stopping...")
model = lgb.LGBMRegressor(
    objective='regression',
    boosting_type='gbdt',
    learning_rate=0.1,
    num_leaves=31,
    n_estimators=1000,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    verbosity=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

#  Predict and evaluate
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\n Train MSE: {train_mse:.2f}, R²: {train_r2:.4f}")
print(f" Test  MSE: {test_mse:.2f}, R²: {test_r2:.4f}")

#  Save model and encoders
joblib.dump(model, 'lightgbm_model.pkl')
joblib.dump(label_encoders, 'lightgbm_label_encoders.pkl')

# ⬇ Download files
files.download('lightgbm_model.pkl')
files.download('lightgbm_label_encoders.pkl')

#  Feature importance
lgb.plot_importance(model, max_num_features=10)
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.show()

#  Actual vs Predicted (log price)
predicted_prices = np.expm1(y_pred_test)
actual_prices = np.expm1(y_test)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=actual_prices, y=predicted_prices, alpha=0.3)
plt.plot([actual_prices.min(), actual_prices.max()],
         [actual_prices.min(), actual_prices.max()],
         color='red', linestyle='--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.tight_layout()
plt.show()