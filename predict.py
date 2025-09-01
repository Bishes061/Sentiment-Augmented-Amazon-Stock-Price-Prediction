# predict.py
import os
import argparse
import numpy as np
import tensorflow as tf
from src.data_loader import load_stock_data

# NOTE: This prediction script is designed for models trained with the 'in_window_norm' feature type.
# Extending it for the 'indicators' type would require saving/re-applying the MinMaxScaler.

WINDOW_SIZE = 30

def predict_next_day_movement(model, latest_data):
    """
    Predicts the next day's stock price movement using a trained model.
    """
    if len(latest_data) < WINDOW_SIZE:
        raise ValueError(f"Need at least {WINDOW_SIZE} days of data, but got {len(latest_data)}.")

    data_window = latest_data.tail(WINDOW_SIZE).copy()

    # Preprocess the window using in-window standardization
    o = (data_window['Open'] - data_window['Open'].mean()) / data_window['Open'].std()
    h = (data_window['High'] - data_window['High'].mean()) / data_window['High'].std()
    l = (data_window['Low'] - data_window['Low'].mean()) / data_window['Low'].std()
    c = (data_window['Close'] - data_window['Close'].mean()) / data_window['Close'].std()
    v = (data_window['Volume'] - data_window['Volume'].mean()) / data_window['Volume'].std()

    input_data = np.column_stack((o, h, l, c, v))
    input_data = np.reshape(input_data, (1, WINDOW_SIZE, 5))

    prediction = model.predict(input_data)
    return "UP" if np.argmax(prediction) == 0 else "DOWN"

def main(args):
    """Main prediction pipeline."""
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    print(f"Loading model from {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path)

    print(f"Loading latest data from {args.data_path}...")
    full_data = load_stock_data(args.data_path)
    
    print("\n--- Predicting for the Next Trading Day ---")
    predicted_movement = predict_next_day_movement(model, full_data)
    print(f"\nPredicted Movement for {args.data_path.split('/')[-1].split('.')[0]}: {predicted_movement}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict next-day stock movement.")
    parser.add_argument('--model_path', type=str, default='saved_models/lstm_AMZN_in_window_norm_classifier.h5', help='Path to the saved Keras model file.')
    parser.add_argument('--data_path', type=str, default='data/AMZN.csv', help='Path to the stock data CSV file.')
    
    args = parser.parse_args()
    main(args)