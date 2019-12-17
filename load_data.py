import time
import datetime as dt
import pandas as pd
import pandas_datareader as web
import numpy as np


def get_technical_indicators(dataset):
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['adjusted close'].rolling(window=7).mean()
    dataset['ma21'] = dataset['adjusted close'].rolling(window=21).mean()

    # Create MACD
    dataset['26ema'] = dataset['adjusted close'].ewm(span=26).mean()
    dataset['12ema'] = dataset['adjusted close'].ewm(span=12).mean()
    dataset['MACD'] = (dataset['12ema'] - dataset['26ema'])

    # Create Bollinger Bands
    dataset['20sd'] = dataset['adjusted close'].rolling(20).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd'] * 2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd'] * 2)

    # Create Exponential moving average
    dataset['ema'] = dataset['adjusted close'].ewm(com=0.5).mean()

    # Create Momentum
    dataset['momentum'] = dataset['adjusted close'] - 1

    return dataset


def get_av_data(symbol, type='av-daily-adjusted'):
    start = dt.datetime(2010, 1, 1)
    end = dt.datetime.now()
    key = 'L43U3XRM44TNHHE7'

    data_received = False
    while not data_received:
        try:
            data = web.DataReader(symbol, type, start, end, api_key=key)
            time.sleep(2)
        except web._utils.RemoteDataError:
            time.sleep(5)
        else:
            data_received = True
    return data


def load(company1, company2, competitors):
    c1 = get_av_data(company1)
    c1 = get_technical_indicators(c1)

    c2 = get_av_data(company2)
    c2 = get_technical_indicators(c2)

    for i, col in c2.items():
        new_name = col.name + " B"
        c1[new_name] = col

    for name in competitors:
        data = get_av_data(name)
        c1[name] = data['adjusted close']
	
    for i, col in c1.items():
        if np.sum(np.isnan(col)) > 0:
            for j, val in zip(range(len(col)), col):
                if np.isnan(val):
                    c1[i][c1.index.values[j]] = np.mean(col)
    np.savetxt('data.txt', c1.values, fmt='%f', delimiter=',')


if __name__ == '__main__':
    competitors = ['JEF', 'PGR', 'AIG', 'STFGX', 'BLK']
    load('BRK.A', 'BRK.B', competitors)
    print(pd.read_csv('data.txt', delimiter=',', header=None))
    exit(0)