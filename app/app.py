# from flask import Flask, jsonify, request
# from flask_cors import CORS
# import traceback
# import joblib
# import pandas as pd
# import numpy as np
# import os
# from difflib import get_close_matches

# app = Flask(__name__)
# CORS(app, resources={r"/api/*": {"origins": "*"}})

# # Load model, encoders, and scaler
# MODEL_PATH = 'server/linear_regression_model.pkl'
# ENCODERS_PATH = 'server/linear_label_encoders.pkl'
# SCALER_PATH = 'server/linear_scaler.pkl'
# FOREST_MODEL_PATH = 'server/forest_model.pkl'
# FOREST_ENCODERS_PATH = 'server/forest_label_encoders.pkl'
# FOREST_SCALER_PATH = 'server/forest_scaler.pkl'
# LIGHTGBM_MODEL_PATH = 'server/lightgbm_model.pkl'
# LIGHTGBM_ENCODERS_PATH = 'server/lightgbm_label_encoders.pkl'
# model = joblib.load(MODEL_PATH)
# label_encoders = joblib.load(ENCODERS_PATH)
# scaler = joblib.load(SCALER_PATH)

# # Categorical columns
# label_cols = ['status', 'city', 'state']

# # Fuzzy matching function for safe encoding
# def fuzzy_encode(value, le, field_name):
#     if not isinstance(value, str) or value.strip() == "":
#         raise ValueError(f"{field_name} is missing or invalid.")

#     value_lower = value.strip().lower()
#     matches = get_close_matches(value_lower, [v.lower() for v in le.classes_], n=1, cutoff=0.5)

#     if matches:
#         original_value = next((v for v in le.classes_ if v.lower() == matches[0]), None)
#         if original_value:
#             return le.transform([original_value])[0]

#     raise ValueError(f"Unrecognized {field_name}: '{value}' (try something like '{le.classes_[0]}')")

# # routes and functions
# @app.route('/api/parameters', methods=['GET'])
# def get_parameters():
#     return jsonify({
#         "status": list(label_encoders['status'].classes_),
#         "cities": list(label_encoders['city'].classes_),
#         "states": list(label_encoders['state'].classes_)
#     })

# # Predict endpoint for linear regression
# @app.route('/api/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#         input_df = pd.DataFrame([data])

#         # Fuzzy encode with helpful errors
#         for col in label_cols:
#             input_df[col] = fuzzy_encode(input_df[col].iloc[0], label_encoders[col], col)

#         # Feature engineering
#         # Convert to numeric, coercing errors into NaN (if any values are non-numeric)
#         input_df['house_size'] = pd.to_numeric(input_df['house_size'], errors='coerce')
#         input_df['acre_lot'] = pd.to_numeric(input_df['acre_lot'], errors='coerce')

#         # Apply log1p on the now numeric values
#         input_df['log_house_size'] = np.log1p(input_df['house_size'])
#         input_df['log_acre_lot'] = np.log1p(input_df['acre_lot'])

#         # Ensure 'bed' and 'bath' columns are numeric, coerce errors to NaN
#         input_df['bed'] = pd.to_numeric(input_df['bed'], errors='coerce')
#         input_df['bath'] = pd.to_numeric(input_df['bath'], errors='coerce')

#         # Fill NaN values with 0 (or another appropriate value)
#         input_df['bed'] = input_df['bed'].fillna(0)
#         input_df['bath'] = input_df['bath'].fillna(0)

#         # Perform the multiplication now that both columns are numeric
#         input_df['bed_bath'] = input_df['bed'] * input_df['bath']

#         features = [
#             'status', 'bed', 'bath', 'city', 'state', 'zip_code',
#             'log_house_size', 'log_acre_lot', 'bed_bath',
#         ]

#         input_data = input_df[features]
#         input_scaled = scaler.transform(input_data)

#         log_price_pred = model.predict(input_scaled)[0]
#         predicted_price = np.expm1(log_price_pred)

#         # Generate synthetic comparable properties with varying bed counts
#         bed_value = int(data['bed'])  # Convert 'bed' value to integer

#         comparable_inputs = []
#         for bed_delta in [-1, 1]:
#             comparable = data.copy()
#             comparable['bed'] = max(1, bed_value + bed_delta)  # Ensure bed is at least 1
#             comparable['bath'] = data['bath']
#             comparable['house_size'] = int(data['house_size'] * (1 + 0.05 * bed_delta))  # Adjust house size proportionally
#             comparable['acre_lot'] = data['acre_lot']
#             comparable['status'] = data['status']
#             comparable['city'] = data['city']
#             comparable['state'] = data['state']
#             comparable['zip_code'] = data['zip_code']
#             comparable_inputs.append(comparable)

#         # Predict comparable property values
#         comparables_result = []
#         for c in comparable_inputs:
#             c_df = pd.DataFrame([c])

#             # Fuzzy encode for comparables
#             for col in label_cols:
#                 c_df[col] = fuzzy_encode(c_df[col].iloc[0], label_encoders[col], col)

#             # Ensure 'house_size' and 'acre_lot' are numeric for log1p
#             c_df['house_size'] = pd.to_numeric(c_df['house_size'], errors='coerce')
#             c_df['acre_lot'] = pd.to_numeric(c_df['acre_lot'], errors='coerce')

#             # Apply log1p after converting to numeric
#             c_df['log_house_size'] = np.log1p(c_df['house_size'])
#             c_df['log_acre_lot'] = np.log1p(c_df['acre_lot'])

#             # Fill NaN values for 'bed' and 'bath'
#             c_df['bed'] = pd.to_numeric(c_df['bed'], errors='coerce').fillna(0)
#             c_df['bath'] = pd.to_numeric(c_df['bath'], errors='coerce').fillna(0)

#             # Perform the multiplication now that both columns are numeric
#             c_df['bed_bath'] = c_df['bed'] * c_df['bath']

#             # Predict with scaled data
#             c_scaled = scaler.transform(c_df[features])
#             log_val = model.predict(c_scaled)[0]
#             val = np.expm1(log_val)

#             # Ensure all values are strings when concatenating
#             comparables_result.append({
#             "location": f"{str(c['city'])} · {str(c['house_size'])} sqft",
#             "value": float(val)
#             })

#         return jsonify({
#             "predicted_price": float(predicted_price),
#             "comparableProperties": comparables_result
#         })

#     except ValueError as ve:
#         return jsonify({"error": str(ve)}), 400
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"error": str(e)}), 500

# # Predict endpoint for Random Forest
# @app.route('/api/forestpredict', methods=['POST'])
# def forest_predict():
#     try:
#         data = request.get_json()
#         input_df = pd.DataFrame([data])

#         # Load forest components if not already loaded (optional safety)
#         forest_model = joblib.load(FOREST_MODEL_PATH)
#         forest_label_encoders = joblib.load(FOREST_ENCODERS_PATH)
#         forest_scaler = joblib.load(FOREST_SCALER_PATH)

#         # Fuzzy encode categorical fields
#         for col in label_cols:
#             input_df[col] = fuzzy_encode(input_df[col].iloc[0], forest_label_encoders[col], col)

#         # Convert numeric fields
#         input_df['house_size'] = pd.to_numeric(input_df['house_size'], errors='coerce')
#         input_df['acre_lot'] = pd.to_numeric(input_df['acre_lot'], errors='coerce')
#         input_df['bed'] = pd.to_numeric(input_df['bed'], errors='coerce').fillna(0)
#         input_df['bath'] = pd.to_numeric(input_df['bath'], errors='coerce').fillna(0)

#         # Feature engineering
#         input_df['price_per_sqft'] = input_df['house_size'] / input_df['house_size']  # equals 1, placeholder
#         input_df['total_rooms'] = input_df['bed'] + input_df['bath']

#         # Select features
#         forest_features = [
#             'bed', 'bath', 'acre_lot', 'city', 'state',
#             'house_size', 'price_per_sqft', 'total_rooms'
#         ]

#         input_scaled = forest_scaler.transform(input_df[forest_features])
#         predicted_price = forest_model.predict(input_scaled)[0]

#         # === Comparables (adjust bed count ±1) ===
#         bed_value = int(data['bed'])

#         comparable_inputs = []
#         for bed_delta in [-1, 1]:
#             comparable = data.copy()
#             comparable['bed'] = max(1, bed_value + bed_delta)
#             comparable['bath'] = data['bath']
#             comparable['house_size'] = int(data['house_size'] * (1 + 0.05 * bed_delta))
#             comparable['acre_lot'] = data['acre_lot']
#             comparable['status'] = data['status']
#             comparable['city'] = data['city']
#             comparable['state'] = data['state']
#             comparable['zip_code'] = data['zip_code']
#             comparable_inputs.append(comparable)

#         # Predict comparables
#         comparables_result = []
#         for c in comparable_inputs:
#             c_df = pd.DataFrame([c])

#             for col in label_cols:
#                 c_df[col] = fuzzy_encode(c_df[col].iloc[0], forest_label_encoders[col], col)

#             c_df['house_size'] = pd.to_numeric(c_df['house_size'], errors='coerce')
#             c_df['acre_lot'] = pd.to_numeric(c_df['acre_lot'], errors='coerce')
#             c_df['bed'] = pd.to_numeric(c_df['bed'], errors='coerce').fillna(0)
#             c_df['bath'] = pd.to_numeric(c_df['bath'], errors='coerce').fillna(0)

#             c_df['price_per_sqft'] = c_df['house_size'] / c_df['house_size']
#             c_df['total_rooms'] = c_df['bed'] + c_df['bath']

#             c_scaled = forest_scaler.transform(c_df[forest_features])
#             val = forest_model.predict(c_scaled)[0]

#             comparables_result.append({
#                 "location": f"{str(c['city'])} · {str(c['house_size'])} sqft",
#                 "value": float(val)
#             })

#         return jsonify({
#             "predicted_price": float(predicted_price),
#             "comparableProperties": comparables_result
#         })

#     except ValueError as ve:
#         return jsonify({"error": str(ve)}), 400
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"error": str(e)}), 500

# # Predict endpoint for LightGBM
# @app.route('/api/lightgbmpredict', methods=['POST'])
# def lightgbm_predict():
#     print("LightGBM predict endpoint called")
#     try:
#         data = request.get_json()
#         input_df = pd.DataFrame([data])

#         # Load model and encoders
#         lightgbm_model = joblib.load(LIGHTGBM_MODEL_PATH)
#         lightgbm_label_encoders = joblib.load(LIGHTGBM_ENCODERS_PATH)

#         # Label columns used in training
#         label_cols = ['status', 'city', 'state']

#         # Encode categorical columns using saved label encoders
#         for col in label_cols:
#             input_df[col] = fuzzy_encode(input_df[col].iloc[0], lightgbm_label_encoders[col], col)

#         # Convert numeric fields and handle missing/invalid data
#         input_df['house_size'] = pd.to_numeric(input_df['house_size'], errors='coerce')
#         input_df['acre_lot'] = pd.to_numeric(input_df['acre_lot'], errors='coerce')
#         input_df['bed'] = pd.to_numeric(input_df['bed'], errors='coerce').fillna(0)
#         input_df['bath'] = pd.to_numeric(input_df['bath'], errors='coerce').fillna(0)
#         input_df['zip_code'] = pd.to_numeric(input_df['zip_code'], errors='coerce').fillna(0)

#         # Feature engineering (must match training!)
#         input_df['log_house_size'] = np.log1p(input_df['house_size'])
#         input_df['log_acre_lot'] = np.log1p(input_df['acre_lot'])
#         input_df['bed_bath'] = input_df['bed'] * input_df['bath']
#         input_df['zip_status'] = input_df['zip_code'] * input_df['status']

#         # Match exact feature set used in training
#         lightgbm_features = [
#             'status', 'bed', 'bath', 'city', 'state', 'zip_code',
#             'log_house_size', 'log_acre_lot', 'bed_bath', 'zip_status'
#         ]

#         input_data = input_df[lightgbm_features]
#         predicted_log_price = lightgbm_model.predict(input_data)[0]
#         predicted_price = float(np.expm1(predicted_log_price))  # reverse log1p

#         # Create comparables (±1 bed)
#         bed_value = int(input_df['bed'].iloc[0])
#         comparables_result = []

#         for bed_delta in [-1, 1]:
#             c = data.copy()
#             c['bed'] = max(1, bed_value + bed_delta)
#             c['house_size'] = int(data['house_size'] * (1 + 0.05 * bed_delta))

#             c_df = pd.DataFrame([c])
#             for col in label_cols:
#                 c_df[col] = fuzzy_encode(c_df[col].iloc[0], lightgbm_label_encoders[col], col)

#             c_df['house_size'] = pd.to_numeric(c_df['house_size'], errors='coerce')
#             c_df['acre_lot'] = pd.to_numeric(c_df['acre_lot'], errors='coerce')
#             c_df['bed'] = pd.to_numeric(c_df['bed'], errors='coerce').fillna(0)
#             c_df['bath'] = pd.to_numeric(c_df['bath'], errors='coerce').fillna(0)
#             c_df['zip_code'] = pd.to_numeric(c_df['zip_code'], errors='coerce').fillna(0)

#             c_df['log_house_size'] = np.log1p(c_df['house_size'])
#             c_df['log_acre_lot'] = np.log1p(c_df['acre_lot'])
#             c_df['bed_bath'] = c_df['bed'] * c_df['bath']
#             c_df['zip_status'] = c_df['zip_code'] * c_df['status']

#             comp_pred_log = lightgbm_model.predict(c_df[lightgbm_features])[0]
#             comp_pred_price = float(np.expm1(comp_pred_log))

#             comparables_result.append({
#                 "location": f"{c['city']} · {c['house_size']} sqft",
#                 "value": comp_pred_price
#             })

#         return jsonify({
#             "predicted_price": predicted_price,
#             "comparableProperties": comparables_result
#         })

#     except ValueError as ve:
#         return jsonify({"error": str(ve)}), 400
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True,port=5001)