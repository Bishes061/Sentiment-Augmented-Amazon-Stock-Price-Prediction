# train.py
import os
import argparse
from src.data_loader import load_stock_data
from src.feature_engineering import get_technical_indicators
from src.prepare_data import create_classification_dataset_in_window, create_dataset_with_indicators
from src.models import build_lstm_classifier, build_gru_classifier
from src.utils import create_train_test_split, plot_training_history

MODEL_SAVE_PATH = 'saved_models/'
WINDOW_SIZE = 30
FORECAST_HORIZON = 1

def main(args):
    """Main training pipeline."""
    # 1. Load Data
    data_path = f'data/{args.ticker}.csv'
    df = load_stock_data(data_path)

    # 2. Prepare Supervised Learning Dataset based on feature type
    if args.feature_type == 'indicators':
        print("Using technical indicators for features...")
        feature_df = get_technical_indicators(df.copy())
        X, Y = create_dataset_with_indicators(feature_df, window_size=WINDOW_SIZE, forecast_horizon=FORECAST_HORIZON)
    else:
        print("Using in-window normalization for features...")
        X, Y = create_classification_dataset_in_window(df, window_size=WINDOW_SIZE, forecast_horizon=FORECAST_HORIZON)

    # 3. Split Data
    X_train, X_test, Y_train, Y_test = create_train_test_split(X, Y, test_size=args.test_size)
    
    # 4. Build Model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = (build_lstm_classifier if args.model_type == 'lstm' else build_gru_classifier)(
        input_shape=input_shape, neurons=args.neurons
    )

    # 5. Train Model
    print(f"\n--- Starting {args.model_type.upper()} Model Training ---")
    history = model.fit(
        X_train, Y_train,
        epochs=args.epochs, batch_size=args.batch_size,
        validation_data=(X_test, Y_test),
        shuffle=True, verbose=1
    )

    # 6. Evaluate Model & Save
    loss, accuracy = model.evaluate(X_test, Y_test, batch_size=args.batch_size)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    model_filename = f'{args.model_type}_{args.ticker}_{args.feature_type}_classifier.h5'
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.save(os.path.join(MODEL_SAVE_PATH, model_filename))
    print(f"Model saved to {os.path.join(MODEL_SAVE_PATH, model_filename)}")

    # 7. Plot History
    plot_name = f"{args.model_type.upper()}_{args.ticker}_{args.feature_type.capitalize()}"
    plot_training_history(history, model_name=plot_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a stock price movement prediction model.")
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'gru'], help='Type of model to train.')
    parser.add_argument('--feature_type', type=str, default='in_window_norm', choices=['in_window_norm', 'indicators'], help='Feature engineering method.')
    parser.add_argument('--ticker', type=str, default='AMZN', help='Stock ticker for data file.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=256, help='Training batch size.')
    parser.add_argument('--neurons', type=int, default=64, help='Number of neurons in the recurrent layer.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data for testing.')
    
    args = parser.parse_args()
    main(args)