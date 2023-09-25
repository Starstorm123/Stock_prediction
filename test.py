import math
import pandas_datareader as web
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
import streamlit as st
plt.style.use('fivethirtyeight')

stocks = ("amzn","aapl","^NSEI","^BANKSEI")
ep_list= [1,2,3,4,5,6,7,8,9,10]
selected_stock=st.selectbox("Select Dataset for Predictions",stocks)
ep=st.selectbox("Select Number of Epochs for Predictions",ep_list)

df = yf.download(selected_stock, start='2015-01-01', end='2023-09-14');
df

plt.figure(figsize=(14,3));
plt.title('closing price');
plt.plot(df['Close'])
plt.show()

st.line_chart(df['Close'])

data=df.filter(['Close'])
dataset=data.values
training_data_len=math.ceil(len(dataset)*.8)
#training_data_len

scaler=MinMaxScaler(feature_range=(0,1));
scaled_data=scaler.fit_transform(dataset)
#scaled_data

train_data=scaled_data[0:training_data_len,:];
x_train=[]
y_train=[]
for i in range(60,len(train_data)):
  x_train.append(train_data[i-60:i,0])
  y_train.append(train_data[i,0])
  if i<=60:
    print(x_train)
    print(y_train)
    print()

x_train,y_train=np.array(x_train),np.array(y_train)

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1));
#x_train.shape

model=Sequential();
model.add(LSTM(50,return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False));
model.add(Dense(25,activation='relu'))
model.add(Dense(1,activation='relu'))

model.compile(optimizer='Adam',loss='mean_squared_error')

model.fit(x_train,y_train,batch_size=1,epochs=ep)

test_data=scaled_data[training_data_len-60:,:]
x_test=[]
y_test=dataset[training_data_len:,:]
for i in range(60,len(test_data)):
  x_test.append(test_data[i-60:i,0])

x_test=np.array(x_test)
x_test.shape
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)

rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
#rmse

train=data[:training_data_len]
valid=data[training_data_len:]
valid['predictions']=predictions
plt.figure(figsize=(14,6))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.plot(train['Close'])
plt.plot(valid[['Close','predictions']])
plt.legend(['Train','Val','predictions'],loc='lower right')
plt.show()

quote=yf.download(selected_stock, start='2015-01-01', end='2023-09-15')
newdf=quote.filter(['Close'])
last_60_days=newdf[-60:].values
last_60_days_scaled=scaler.transform(last_60_days)
X_test=[]
X_test.append(last_60_days_scaled)
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1));
pred_price=model.predict(X_test)
pred_price=scaler.inverse_transform(pred_price)
print(pred_price)
st.text_input('Predicted Value',pred_price)

quote2=yf.download(selected_stock, start='2015-01-01', end='2023-09-14')
print(quote2['Close'])

err=quote2['Close']-pred_price[0][0]
#err


def calculate_mape(y_true, y_pred):

    y_pred, y_true = np.array(y_pred), np.array(y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

accuracy=calculate_mape(quote2['Close'],pred_price[0][0])
#accuracy

