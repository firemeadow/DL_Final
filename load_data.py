import time
import datetime as dt
import pandas as pd
import pandas_datareader as web
import numpy as np

if __name__ == '__main__':
    ticker_names = np.asarray(pd.read_csv('tickers.txt', delimiter=',', encoding="utf-8-sig", header=None).values.tolist()).flatten()
    start = dt.datetime(2015, 1, 1)
    end = dt.datetime.now()
    data = []
    #data = pd.read_csv('data.txt', delimiter=',', encoding='utf-8-sig', header=None)
    i = 0
    for name in ticker_names:
        recieved_data = False
        print(i)
        temp_ticker_names = ticker_names[i + 1:]
        while not recieved_data:
            try:
                df = web.DataReader(name, 'av-daily', start, end, api_key='L43U3XRM44TNHHE7')
                df = df[['open', 'close']].values.flatten()
                data.append(df)
                np.savetxt('data.txt', data, fmt='%s', delimiter=',')
                np.savetxt('tickers.txt', temp_ticker_names, fmt='%s', delimiter=',')
                time.sleep(1)
            except web._utils.RemoteDataError:
                time.sleep(10)
            else:
                recieved_data = True
        i += 1