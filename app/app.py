import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
from google.colab import files

# Load dataset
df = pd.read_csv("./realtor-data.zip.csv")

# Data cleaning
print("Cleaning data...")
print("\nInitial Dataset Info:")
print(df.info())

df.dropna(inplace=True)

# Encode categorical columns
label_cols = ['status', 'city', 'state']
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Create new features
df["price_per_sqft"] = df["price"] / df["house_size"]
df["total_rooms"] = df["bed"] + df["bath"]

# Select features and target
features = [
    'bed', 'bath', 'acre_lot', 'city', 'state',
    'house_size', 'price_per_sqft', 'total_rooms'
]
target = 'price'

# Sample 2.2 million examples
if len(df) > 2_200_000:
    df = df.sample(n=2_200_000, random_state=42)

X = df[features]
y = df[target]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into Train (70%), Val (20%), Test (10%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=2/9, random_state=42
)

print(f"\nTrain size: {len(X_train)}, "
      f"Val size: {len(X_val)}, "
      f"Test size: {len(X_test)}")

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Evaluate function
def evaluate(split_name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f"{split_name} R² Score: {r2:.4f}, MSE: {mse:.2f}")


print("\nEvaluation:")
evaluate("Train", y_train, model.predict(X_train))
evaluate("Validation", y_val, model.predict(X_val))
evaluate("Test", y_test, model.predict(X_test))

# Save components
joblib.dump(model, 'forest_model.pkl')
joblib.dump(scaler, 'forest_scaler.pkl')
joblib.dump(label_encoders, 'forest_label_encoders.pkl')

# Predictions and Plot
y_pred_test = model.predict(X_test)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.3, color='teal')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices (Test Set)")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)
plt.grid(True)
plt.tight_layout()
plt.show()

# Download files
files.download('forest_model.pkl')
files.download('forest_scaler.pkl')
files.download('forest_label_encoders.pkl')
