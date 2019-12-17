import time
import datetime as dt
import pandas as pd
import pandas_datareader as web
import numpy as np


def fourier_transforms(dataset):
    data_FT = dataset['adjusted close']
    close_fft = np.fft.fft(np.asarray(data_FT.tolist()))
    fft_df = pd.DataFrame({'fft': close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    fft_list = np.asarray(fft_df['fft'].tolist())
    for num_ in [3, 6, 9]:
        fft_list_m10 = np.copy(fft_list)
        fft_list_m10[num_:-num_] = 0
        dataset["fourier " + str(num_)] = np.fft.ifft(fft_list_m10)

    return dataset


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
    c1 = fourier_transforms(c1)

    c2 = get_av_data(company2)
    c2 = get_technical_indicators(c2)
    c2 = fourier_transforms(c2)

    for i, col in c2.items():
        new_name = col.name + " B"
        c1[new_name] = col

    for name in competitors:
        data = get_av_data(name)
        c1[name] = data['adjusted close']

    for i, col in data.items():
        if np.sum(np.isna(col)) > 0:
            for j, val in zip(range(len(col)), col):
                if val.isna():
                    col[j] = np.mean(col)
            data[[i]] = col
            
    return c1


if __name__ == '__main__':
    competitors = ['AVVIY', 'JEF', 'PGR', 'AIG', 'STFGX', 'BLK']
    data = load('BRK.A', 'BRK.B', competitors)
    exit(0)
