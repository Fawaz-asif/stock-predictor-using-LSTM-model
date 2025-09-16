# Stock Price Predictor using LSTM

This project is a simple demonstration of predicting stock prices using a Long Short-Term Memory (LSTM) model in Python.

The program fetches historical stock data, trains an LSTM model, and predicts future prices. Results are displayed in both numeric and graphical form.

 Features

Fetches stock data in real-time using Yahoo Finance API

Preprocesses and scales the data for training

Trains an LSTM neural network on past prices

Predicts and displays future stock prices (up to 30 days)

Interactive graph of actual vs predicted prices

 NOTES:

Default epochs are set to 3 for faster demonstration.

For increased accuracy, you can raise the number of epochs (e.g., 20–50), but this will also increase training time.

Accuracy depends on training length, stock volatility, and amount of historical data.

Future Improvements

Deploy as a full web app with real-time updates

Add support for multiple tickers in one run

Experiment with different models (GRU, ARIMA)
