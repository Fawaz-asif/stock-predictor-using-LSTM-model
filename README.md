<div align="center">

# LSTM Stock Predictor

### Time-Series Forecasting for Financial Markets using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch/Keras](https://img.shields.io/badge/Deep_Learning-LSTM-FF6F00?style=for-the-badge&logo=keras&logoColor=white)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

### ðŸŒ **[Live Demo: Try the Streamlit App Here!](https://stock-predictor-using-lstm-model-km4vnsnhvstukrral4ra4e.streamlit.app/)**

A machine learning application that uses Long Short-Term Memory (LSTM) neural networks to predict future stock prices based on historical market data. Designed to demonstrate competency in time-series forecasting, sequential data processing, and financial data analysis.

</div>

---

## Why LSTMs?

Standard neural networks evaluate data points independently. Financial markets, however, are sequential - today's price is heavily influenced by yesterday's price. 

**Long Short-Term Memory (LSTM)** networks are a specialized type of Recurrent Neural Network (RNN) that can maintain an "internal state" or memory over long sequences, making them the industry standard architecture for time-series forecasting like stock price prediction.

---

## Features

- **Data Fetching & Preprocessing**: Automatically pulls historical stock data and scales it for neural network consumption.
- **Sequence Generation**: Converts flat time-series data into sliding window sequences required for LSTM training.
- **Deep Learning Architecture**: Custom LSTM model designed to capture both short-term volatility and long-term trends.
- **Visualization**: Generates clear, comparative graphs showing actual historical prices vs. the model's predictions.

---

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Fawaz-asif/stock-predictor-using-LSTM-model.git
cd stock-predictor-using-LSTM-model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

---

## Continuous Integration

This repository implements a CI/CD pipeline using **GitHub Actions**. Upon every commit:
- A cloud runner spins up an Ubuntu environment.
- Installs all data science dependencies.
- Runs structural linting (`flake8`) to guarantee code quality and catch syntax errors before they reach production.

---

## Disclaimer

This project is for educational and portfolio purposes only. Financial markets are highly volatile and influenced by unpredictable real-world events. Do not use this model for actual financial trading or investment decisions.

## Author
**Fawaz Asif**
