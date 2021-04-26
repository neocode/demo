import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import urllib, base64

from flask import Flask, render_template, request
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from pickle import load
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering or Chrome Frame,
    and also to cache the rendered page for 10 minutes
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers["Cache-Control"] = "public, max-age=0"
    return r


def get_model(look_back):
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(1, look_back)))
    model.add(LSTM(256))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


look_back = 7
model = get_model(look_back)
model.summary()

model.load_weights('bitcoin_model.h5')
scaler = load(open('scaler.pkl', 'rb'))

dataframe = pd.read_csv('data.csv')
new_index = pd.DatetimeIndex(dataframe['Date'])
dataframe.set_index(new_index, inplace=True)
dataframe.drop('Date', axis=1, inplace=True)


def predict_global(dataset, X, start_date, end_date):

    if start_date == '':
        start_date = '2020-03-01'

    if end_date == '':
        end_date = '2021-01-01'

    if datetime.strptime(end_date, "%Y-%m-%d") <= datetime.strptime(start_date, "%Y-%m-%d"):
        start_date = '2020-03-01'
        end_date = '2021-01-01'

    predicts = model.predict(X)
    predicts = scaler.inverse_transform(predicts)
    predictions = np.empty_like(dataset)
    predictions[:, :] = np.nan

    predictions[look_back:len(predicts) + look_back, :] = predicts
    predictionsDF = pd.DataFrame(predictions, columns=["predicted"], index=dataframe.index)
    ans = pd.concat([dataframe, predictionsDF], axis=1)

    sns.set(rc={'figure.figsize': (20, 9), 'axes.titlesize': 'x-large'})
    ans[start_date:end_date].plot()
    plt.title('Bitcoin exchange rate (closing trades) and prediction from %s to %s' % (start_date, end_date))

    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', pad_inches=0.1, bbox_inches='tight')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    return uri


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        #takes
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i+look_back, 0])
    return np.array(dataX), np.array(dataY)


def make_original_image(ans):
    sns.set(rc={'figure.figsize': (20, 9), 'axes.titlesize': 'x-large'})
    ans['2020-03-01':'2021-01-01'].plot()
    plt.title('Bitcoin exchange rate (closing trades) from 2020-03-01 to 2021-01-01')
    # plt.savefig('static/original.png', format='png', pad_inches=0.1, bbox_inches='tight')
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', pad_inches=0.1, bbox_inches='tight')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    return uri


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    print(start_date, end_date)
    np.random.seed(0)
    dataset = dataframe.values
    dataset = dataset.astype('float64').reshape(-1, 1)
    dataset = scaler.transform(dataset)
    X, _ = create_dataset(dataset, look_back)
    # Reshape input to be [samples, time steps, features].
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    img_png = predict_global(dataset, X, start_date, end_date)
    return render_template("index.html",
                           main_image="prediction.png",
                           img_png=img_png,
                           visib1="d-none",
                           visib2="")


@app.route("/")
def main():
    img_png = make_original_image(dataframe)
    return render_template("index.html",
                           main_image="original.png",
                           img_png=img_png,
                           visib1="",
                           visib2="d-none")


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


if __name__ == '__main__':
    app.run(debug=True)