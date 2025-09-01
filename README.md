# Stock Price Movement Forecasting using Deep Learning

This project architects and benchmarks deep learning models (LSTM & GRU) to forecast the next-day price movement (Up/Down) of stocks, using historical OHLCV data. The core methodology reframes the time-series regression problem into a binary classification task, achieving superior performance evaluation and interpretability.

## Key Features

- **End-to-End Pipeline:** Implements a full pipeline from data loading and preprocessing to model training, evaluation, and prediction.
- **Advanced Preprocessing:** Utilizes in-window standardization to handle the non-stationarity of financial time-series data.
- **Model Benchmarking:** Includes easily swappable architectures for both LSTM and GRU networks.
- **Performance:** Achieved **~68% test accuracy** in predicting the next day's price movement for AMZN stock.

## Project Structure

stock-price-prediction/
│
├── data/
│ └── AMZN.csv
├── saved_models/
│ └── lstm_classifier.h5
├── src/
│ ├── data_loader.py
│ ├── models.py
│ ├── prepare_data.py
│ └── utils.py
├── train.py
├── predict.py
├── requirements.txt
└── README.md


## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://your-github-repo-url.git
    cd stock-price-prediction
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Training the Model

To train the model, run the `train.py` script from the root directory. The script will load the data, preprocess it, build the LSTM model, train it, and save the final model to the `saved_models/` directory.

```bash
python train.py

Train a GRU model with technical indicators:
code
Bash
python train.py --model_type gru --feature_type indicators --epochs 100


Train an LSTM with more neurons:
code
Bash
python train.py --model_type lstm --neurons 128 --batch_size 128


2. Making a Prediction
After a model has been trained, run predict.py to forecast the next day's movement.
code
Bash
# Predict using the default saved model
python predict.py

# Predict using a specific model file
python predict.py --model_path saved_models/gru_AMZN_classifier.h5