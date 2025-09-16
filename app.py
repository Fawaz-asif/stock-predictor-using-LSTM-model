import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Streamlit UI

st.title("📈 Real-Time Stock Price Prediction")
st.write("Enter a stock ticker (e.g., AAPL, MSFT, GOOGL) to see predictions for the next 30 days.")

# Popular tickers + custom input
popular_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "NVDA"]
selected_ticker = st.selectbox("Choose a Popular Stock Ticker:", popular_tickers)
custom_ticker = st.text_input("Or Enter a Custom Ticker:", "")

# Final ticker selection
ticker = custom_ticker if custom_ticker else selected_ticker

if st.button("Predict"):
    with st.spinner("Fetching data and training model... Please wait ⏳"):
        try:
            
            # Fetch Stock Data
           
            data = yf.download(ticker, period="1y")

            if data.empty:
                st.error("⚠️ No data found for this ticker. Please try another.")
                st.stop()

            st.subheader("Raw Stock Price Data (Last 1 Year)")
            st.dataframe(data.tail())

            
            # Preprocessing
           
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

            seq_length = 60  # Use past 60 days
            x_train, y_train = [], []

            for i in range(seq_length, len(scaled_data)):
                x_train.append(scaled_data[i - seq_length:i])
                y_train.append(scaled_data[i])

            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # Reshape for LSTM

            
            # Build Model
            
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(LSTM(units=50))
            model.add(Dense(1))

            model.compile(optimizer="adam", loss="mean_squared_error")
            model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=0)

            
            # Evaluate Model
            
            train_predictions = model.predict(x_train, verbose=0)
            train_predictions_rescaled = scaler.inverse_transform(train_predictions)
            y_train_rescaled = scaler.inverse_transform(y_train.reshape(-1, 1))

            rmse = np.sqrt(mean_squared_error(y_train_rescaled, train_predictions_rescaled))
            st.success(f"✅ Model trained. RMSE on training data: {rmse:.2f}")

            
            # Prediction Loop (30 days ahead)
           
            future_predictions = []
            current_seq = scaled_data[-seq_length:]
            current_seq = np.expand_dims(current_seq, axis=0)

            for _ in range(30):
                pred = model.predict(current_seq, verbose=0)[0][0]
                future_predictions.append(pred)
                pred_reshaped = np.array([[[pred]]])
                current_seq = np.append(current_seq[:, 1:, :], pred_reshaped, axis=1)

            future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

           
            # Combine Actual + Predictions
            
            future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30)
            predicted_df = pd.DataFrame(future_predictions, index=future_dates, columns=["Predicted Price"])
            combined_df = pd.concat([data["Close"], predicted_df["Predicted Price"]])

            
            # Display Results
            
            st.subheader("Predicted Prices for Next 30 Days")
            st.dataframe(predicted_df)

            st.subheader("Stock Price Prediction Chart")
            plt.figure(figsize=(12, 6))
            plt.plot(combined_df.index, combined_df.values, label="Price (Actual + Predicted)")
            plt.axvline(x=data.index[-1], color="red", linestyle="--", label="Prediction Start")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            st.pyplot(plt)

           
            # Disclaimer
          
            st.markdown("---")
            st.caption("📌 *Disclaimer: This app is for educational purposes only and does not constitute financial advice.*")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
