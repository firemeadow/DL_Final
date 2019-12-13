import time
import datetime as dt
import pandas as pd
import pandas_datareader as web
import numpy as np

if __name__ == '__main__':
    ticker_names = np.asarray(pd.read_csv('tickers.txt', delimiter=',', encoding="utf-8-sig").values.tolist()).flatten()
    start = dt.datetime(2015, 1, 1)
    end = dt.datetime.now()
    data = []
    #data = pd.read_csv('data.txt', delimiter=',', encoding='utf-8-sig', header=None)
    #print(data)
    i = 0
    for name in ticker_names:
        print(i)
        try:
            df = web.DataReader(name, 'av-daily', start, end, api_key='XBSV0WW2LIIXV95A')
            df = df[['open', 'close']].values.flatten()
            data.append(df)
            time.sleep(20)
        except web._utils.RemoteDataError:
            print(name)
            time.sleep(20)
            try:
                df = web.DataReader(name, 'av-daily', start, end, api_key='XBSV0WW2LIIXV95A')
                df = df[['open', 'close']].values.flatten()
            except web._utils.RemoteDataError:
                print(name)
                time.sleep(20)
                df = web.DataReader(name, 'av-daily', start, end, api_key='XBSV0WW2LIIXV95A')
                df = df[['open', 'close']].values.flatten()
        i += 1

    data = pd.DataFrame(data)
    print(data)
    np.savetxt('data.txt', data.values, fmt='%f', delimiter=',')