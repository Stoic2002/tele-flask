import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
from flask_cors import CORS

app = Flask(__name__)

def load_model_and_scaler(target_column):
    try:
        model = tf.keras.models.load_model(f"lstm_model_{target_column.upper()}.h5")
        model.compile(optimizer="adam", loss="mse")
        scaler = joblib.load(f"scaler_{target_column.upper()}.pkl")
        model_path = f"lstm_model_{target_column.upper()}.h5"
        scaler_path = f"scaler_{target_column.upper()}.pkl"
        print(f"Mencoba membuka model dari: {model_path}")
        print(f"Mencoba membuka scaler dari: {scaler_path}")
        return model, scaler
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None, None

def normalize_columns(df):
    """
    Normalize column names to uppercase
    Handles both lowercase and uppercase input
    """
    column_map = {
        'inflow': 'INFLOW',
        'outflow': 'OUTFLOW',
        'tma': 'TMA',
        'beban': 'BEBAN',
        'datetime': 'datetime'
    }
    
    # Rename columns to uppercase
    df = df.rename(columns={col: column_map.get(col.lower(), col) for col in df.columns})
    
    return df

def prepare_input_sequence(df, look_back=24, target_column='INFLOW'):
    # Add time-based features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    feature_columns = ['INFLOW', 'OUTFLOW', 'TMA', 'BEBAN', 'hour', 'day_of_week', 'month']
    missing_columns = [col for col in feature_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing columns in data: {missing_columns}")
        return None
    
    if len(df) < look_back:
        print(f"Not enough data. Need at least {look_back} rows.")
        return None
    
    return df[feature_columns].iloc[-look_back:]

def predict_multiple_steps(model, scaler, input_sequence, target_column, steps=24):
    predictions = []
    current_input = input_sequence.copy()
    
    for _ in range(steps):
        scaled_input = scaler.transform(current_input)
        input_reshaped = scaled_input.reshape(1, scaled_input.shape[0], scaled_input.shape[1])
        prediction_scaled = model.predict(input_reshaped)[0, 0]
        
        full_inverse_array = np.zeros((1, scaler.scale_.shape[0]))
        feature_columns = ['INFLOW', 'OUTFLOW', 'TMA', 'BEBAN', 'hour', 'day_of_week', 'month']
        full_inverse_array[0, feature_columns.index(target_column.upper())] = prediction_scaled
        prediction = scaler.inverse_transform(full_inverse_array)[0, feature_columns.index(target_column.upper())]
        
        predictions.append(prediction)
        
        # Update current input with the new prediction
        next_row = current_input.iloc[-1].copy()
        next_row.loc[target_column.upper()] = prediction
        
        # Update time-related features
        next_datetime = current_input.index[-1] + timedelta(hours=1)
        next_row.loc['hour'] = next_datetime.hour
        next_row.loc['day_of_week'] = next_datetime.dayofweek
        next_row.loc['month'] = next_datetime.month
        
        # Convert next_row to DataFrame before appending
        next_row_df = pd.DataFrame([next_row.values], 
                                   columns=current_input.columns, 
                                   index=[next_datetime])
        
        current_input = pd.concat([current_input.iloc[1:], next_row_df])
    
    return predictions

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data dari JSON request
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Parameter default
    target_column = data.get('target_column', 'INFLOW').upper()
    look_back = data.get('look_back', 24)
    steps = data.get('steps', 12)
    
    # Konversi data ke DataFrame
    df = pd.DataFrame(data['data'])
    
    # Normalize column names
    df = normalize_columns(df)
    
    # Validasi input data
    required_columns = ['INFLOW', 'OUTFLOW', 'TMA', 'BEBAN', 'datetime']
    for col in required_columns:
        if col not in df.columns:
            return jsonify({'error': f'Missing required column: {col}'}), 400
    
    # Set datetime sebagai index
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # Load model dan scaler
    model, scaler = load_model_and_scaler(target_column)
    if model is None or scaler is None:
        return jsonify({'error': 'Could not load model or scaler'}), 500
    
    # Persiapkan input sequence
    input_sequence = prepare_input_sequence(df, look_back=look_back, target_column=target_column)
    if input_sequence is None:
        return jsonify({'error': 'Could not prepare input sequence'}), 400
    
    # Lakukan prediksi
    try:
        predictions = predict_multiple_steps(model, scaler, input_sequence, target_column, steps=steps)
        
        # Generate prediction times
        last_datetime = df.index[-1]
        prediction_times = pd.date_range(start=last_datetime, periods=steps + 1, freq='H')[1:]
        
        # Siapkan response
        response = {
            'target_column': target_column,
            'predictions': [
                {
                    'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'value': float(pred)
                } for time, pred in zip(prediction_times, predictions)
            ]
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    response = jsonify({'status': 'healthy'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response, 200

if __name__ == '__main__':
    CORS(app, resources={r"/predict": {"origins": "*"}})
    app.run(host='0.0.0.0', port=8989, debug=True)

# # Contoh payload untuk prediksi
# payload = {
#     "target_column": "BEBAN",  # Kolom yang ingin diprediksi
#     "look_back": 24,  # Jumlah data historis yang digunakan
#     "steps": 12,  # Jumlah langkah prediksi ke depan
#     "data": [
#         {
#             "datetime": "2024-01-01 00:00:00",
#             "INFLOW": 100.5,
#             "OUTFLOW": 90.2,
#             "TMA": 50.3,
#             "BEBAN": 75.6
#         },
#         # ... tambahkan data historis lainnya minimal sejumlah look_back
#     ]
# }
