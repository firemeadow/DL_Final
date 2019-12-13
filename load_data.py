import time
import datetime as dt
import pandas as pd
import pandas_datareader as web
import numpy as np

def import_stock_tickers():
    ticker_names = pd.read_csv('nasdaqlisted.txt', '|')
    ticker_names = ticker_names['Symbol']
    ticker_names = ticker_names.drop(len(ticker_names) - 1)
    np.savetxt('tickers.txt', ticker_names.values, fmt='%s', delimiter=',')
    return ticker_names

if __name__ == '__main__':
    ticker_names = import_stock_tickers()
    start = dt.datetime(2015, 1, 1)
    end = dt.datetime.now()
    data = []
    for name in ticker_names:
        try:
            df = web.DataReader(name, 'av-daily', start, end, api_key='ARWVOEXF8X8OW4GF')
            df = df[['open', 'close']].values.flatten()
            data.append(df)
            time.sleep(12)
        except web._utils.RemoteDataError:
            print(name)

    data = pd.DataFrame(data)
    print(data)
    np.savetxt('data.txt', data.values, fmt='%f')