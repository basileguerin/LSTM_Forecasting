import streamlit as st
import pandas as pd
import numpy as np
import pickle
import mlflow
import os

aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

mlflow.set_tracking_uri("https://isen-mlflow-fae8e0578f2f.herokuapp.com/")

logged_model = 'runs:/90283baf70024e14afbc33cb57b7d4a9/LSTM_model'
model = mlflow.tensorflow.load_model(logged_model)

df = pd.read_csv('./data.csv', index_col=0)
scaler = pickle.load(open('./scaler.pkl', 'rb'))

look_back = 14
values = df.tail(look_back).values
x_input = scaler.transform(values).reshape((1, 14, 1))

st.set_page_config(page_title='LSTM Forecasting', page_icon="chart_with_upwards_trend", layout='wide')
st.title('Forecasting with LSTM Model')

period_to_visualize = st.selectbox('Select the period to visualize:', ['Entire Time Series', 'Last Month', 'Last Year'])

if period_to_visualize == 'Last Month':
    df_display = df.tail(30)
elif period_to_visualize == 'Last Year':
    df_display = df.tail(365)
else:
    df_display = df 

st.subheader('Time Series:')
st.line_chart(df_display)

if st.button('Predict the next 7 days'):
    predictions = []
    for _ in range(7):
        y_pred = model.predict(x_input)
        y_pred_rescaled = scaler.inverse_transform(y_pred)
        predictions.append(y_pred_rescaled[0][0])
        x_input = np.append(x_input[0], y_pred)
        x_input = x_input[-look_back:]
        x_input = x_input.reshape((1, look_back, 1))

    result_dict = {(pd.to_datetime(df.index[-1]) + pd.DateOffset(days=i+1)).strftime('%Y-%m-%d'): predictions[i] for i in range(7)}
    st.subheader('Model predictions :')
    st.json(result_dict)