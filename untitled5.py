import yfinance as yf
print("yfinance is successfully installed")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st


start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Price Prediction')


user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input,start,end)

#Describing Data
st.subheader('Data from 2010 -2019')
st.write(df.describe())

df.tail()

df = df.reset_index()
df.head()

df = df.drop(['Date','Adj Close'],axis=1)
df.head()

plt.plot(df['Close'])

df

ma100 = df.Close.rolling(100).mean()
ma100

plt.figure(figsize=(12,6))
plt.plot(df['Close'])
plt.plot(ma100,'r')

ma200 = df.Close.rolling(200).mean()
ma200

plt.figure(figsize = (12,6))
plt.plot(df['Close'])
plt.plot(ma200,'g')
plt.plot(ma100,'r')

df.shape

#splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
print(data_training.shape)
print(data_testing.shape)

data_training.head()

x_train.shape

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = data_training.values  # Convert data_training to NumPy array

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

#ML model

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

from re import M
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))


model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))


model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))


model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))


model.add(Dense(units=1))

model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(x_train, y_train, epochs=50)

model.save('keras_model.h5')

data_testing.head()

data_training.tail()

past_100days_data = data_training.tail(100)
final_df = pd.concat([past_100days_data, data_testing], ignore_index=True)
final_df.head()

input_data = scaler.fit_transform(final_df)
input_data

input_data.shape

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_train ,y_test = np.array(x_test),np.array(y_test)
print(x_train.shape)
print(y_test.shape)

#Making predictions
y_predicted = model.predict(x_test)

y_predicted.shape

y_test

y_test

y_predicted

scaler.scale_

scale_factor = 1/0.02123255
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label = 'Original Price')
plt.plot(y_predicted,'r',label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()