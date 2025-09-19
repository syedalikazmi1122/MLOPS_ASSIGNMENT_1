from flask import Flask, jsonify, request
from flask_cors import CORS
import traceback
import joblib
import pandas as pd
import numpy as np
import os
from difflib import get_close_matches

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Resolve artifact paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIGHTGBM_MODEL_PATH = os.path.join(BASE_DIR, 'lightgbm_model.pkl')
LIGHTGBM_ENCODERS_PATH = os.path.join(BASE_DIR, 'lightgbm_label_encoders.pkl')

# Load encoders once for parameters endpoint; model can be lazy-loaded in handler
label_cols = ['status', 'city', 'state']
try:
    lightgbm_label_encoders = joblib.load(LIGHTGBM_ENCODERS_PATH)
except Exception:
    lightgbm_label_encoders = None


def fuzzy_encode(value, le, field_name):
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{field_name} is missing or invalid.")

    value_lower = value.strip().lower()
    matches = get_close_matches(value_lower, [v.lower() for v in le.classes_], n=1, cutoff=0.5)

    if matches:
        original_value = next((v for v in le.classes_ if v.lower() == matches[0]), None)
        if original_value:
            return le.transform([original_value])[0]

    raise ValueError(f"Unrecognized {field_name}: '{value}' (try something like '{le.classes_[0]}')")


@app.route('/api/parameters', methods=['GET'])
def get_parameters():
    if lightgbm_label_encoders is None:
        return jsonify({"error": "Encoders not available"}), 500
    return jsonify({
        "status": list(lightgbm_label_encoders['status'].classes_),
        "cities": list(lightgbm_label_encoders['city'].classes_),
        "states": list(lightgbm_label_encoders['state'].classes_)
    })


@app.route('/api/lightgbmpredict', methods=['POST'])
def lightgbm_predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])

        lightgbm_model = joblib.load(LIGHTGBM_MODEL_PATH)
        encoders = lightgbm_label_encoders or joblib.load(LIGHTGBM_ENCODERS_PATH)

        for col in label_cols:
            input_df[col] = fuzzy_encode(input_df[col].iloc[0], encoders[col], col)

        input_df['house_size'] = pd.to_numeric(input_df['house_size'], errors='coerce')
        input_df['acre_lot'] = pd.to_numeric(input_df['acre_lot'], errors='coerce')
        input_df['bed'] = pd.to_numeric(input_df['bed'], errors='coerce').fillna(0)
        input_df['bath'] = pd.to_numeric(input_df['bath'], errors='coerce').fillna(0)
        input_df['zip_code'] = pd.to_numeric(input_df['zip_code'], errors='coerce').fillna(0)

        input_df['log_house_size'] = np.log1p(input_df['house_size'])
        input_df['log_acre_lot'] = np.log1p(input_df['acre_lot'])
        input_df['bed_bath'] = input_df['bed'] * input_df['bath']
        input_df['zip_status'] = input_df['zip_code'] * input_df['status']

        lightgbm_features = [
            'status', 'bed', 'bath', 'city', 'state', 'zip_code',
            'log_house_size', 'log_acre_lot', 'bed_bath', 'zip_status'
        ]

        input_data = input_df[lightgbm_features]
        predicted_log_price = lightgbm_model.predict(input_data)[0]
        predicted_price = float(np.expm1(predicted_log_price))

        bed_value = int(input_df['bed'].iloc[0])
        comparables_result = []
        for bed_delta in [-1, 1]:
            c = data.copy()
            c['bed'] = max(1, bed_value + bed_delta)
            c['house_size'] = int(data['house_size'] * (1 + 0.05 * bed_delta))

            c_df = pd.DataFrame([c])
            for col in label_cols:
                c_df[col] = fuzzy_encode(c_df[col].iloc[0], encoders[col], col)

            c_df['house_size'] = pd.to_numeric(c_df['house_size'], errors='coerce')
            c_df['acre_lot'] = pd.to_numeric(c_df['acre_lot'], errors='coerce')
            c_df['bed'] = pd.to_numeric(c_df['bed'], errors='coerce').fillna(0)
            c_df['bath'] = pd.to_numeric(c_df['bath'], errors='coerce').fillna(0)
            c_df['zip_code'] = pd.to_numeric(c_df['zip_code'], errors='coerce').fillna(0)

            c_df['log_house_size'] = np.log1p(c_df['house_size'])
            c_df['log_acre_lot'] = np.log1p(c_df['acre_lot'])
            c_df['bed_bath'] = c_df['bed'] * c_df['bath']
            c_df['zip_status'] = c_df['zip_code'] * c_df['status']

            comp_pred_log = lightgbm_model.predict(c_df[lightgbm_features])[0]
            comp_pred_price = float(np.expm1(comp_pred_log))
            comparables_result.append({
                "location": f"{c['city']} · {c['house_size']} sqft",
                "value": comp_pred_price
            })

        return jsonify({
            "predicted_price": predicted_price,
            "comparableProperties": comparables_result
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)
