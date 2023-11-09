import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import pickle
import mlflow
from mlflow.models import infer_signature
import os

os.environ['AWS_ACCESS_KEY_ID'] = "AKIA3R62MVALHESATEYJ"
os.environ['AWS_SECRET_ACCESS_KEY'] = "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"
os.environ['ARTIFACT_STORE_URI'] = "s3://isen-mlflow/models/"

df = pd.read_csv('./data.csv', index_col=0)

scaler = MinMaxScaler()
df['value'] = scaler.fit_transform(df['value'].values.reshape(-1, 1))

def create_dataset(df, look_back=1, days_ahead=1):
    """
    Function pour reshape les données en X=t et Y=t+days_ahead
    """
    dataX, dataY = [], []
    for i in range(len(df)-look_back-days_ahead):
        a = df[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(df[i + look_back + days_ahead - 1, 0])
    return np.array(dataX), np.array(dataY)

look_back = 14

X, y = create_dataset(df.values, look_back, 1)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1), activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X, y, epochs=100, batch_size=32, verbose=1, shuffle=0)

pickle.dump(scaler, open('./scaler.pkl', 'wb'))

mlflow.set_tracking_uri("https://isen-mlflow-fae8e0578f2f.herokuapp.com/")
mlflow.set_experiment("Basile")
experiment = mlflow.get_experiment_by_name("Basile")
signature = infer_signature(X, y)

with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='Training LSTM'):
    mlflow.tensorflow.log_model(model,
                           "LSTM_model",
                           signature=signature,
                           input_example = X[0],
                           registered_model_name='LSTM_model')