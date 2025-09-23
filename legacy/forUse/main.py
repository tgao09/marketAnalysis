import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
from termcolor import colored
from curl_cffi import requests

import time_difference as tdf

print(tdf.minutes_since_930am_est())
print(tdf.minutes_since_xx30_est())

warnings.filterwarnings("ignore", category=UserWarning)
session = requests.Session(impersonate = "chrome")

def get_projected(tickerName, period, interval, current):
    ticker = yf.Ticker(tickerName, session=session)
    df = ticker.history(period = period, interval = interval)
    df = df.drop(['Dividends', 'Stock Splits'], axis = 1)
    df.to_csv(f"./stock_data/{tickerName}.csv")
    df = pd.read_csv(f"./stock_data/{tickerName}.csv")

    df = df.assign(**{'Open(t-1)': df['Open'].shift(-1)},
                    **{'Close(t-1)': df['Close'].shift(-1)},
                    **{'High(t-1)': df['High'].shift(-1)},
                    **{'Low(t-1)': df['Low'].shift(-1)},
                    **{'Volume(t-1)': df['Volume'].shift(-1)},
                    **{'Open(t-2)': df['Open'].shift(-2)},
                    **{'Close(t-2)': df['Close'].shift(-2)},
                    **{'High(t-2)': df['High'].shift(-2)},
                    **{'Low(t-2)': df['Low'].shift(-2)},
                    **{'Volume(t-2)': df['Volume'].shift(-2)},
                    **{'Open(t-3)': df['Open'].shift(-3)},
                    **{'Close(t-3)': df['Close'].shift(-3)},
                    **{'High(t-3)': df['High'].shift(-3)},
                    **{'Low(t-3)': df['Low'].shift(-3)},
                    **{'Volume(t-3)': df['Volume'].shift(-3)})

    df.drop(['High', 'Low', 'Close', 'Volume'], axis = 1)
    df = df.dropna()
    df = df.copy()

    x = df[['Open',
            'Open(t-1)', 'Close(t-1)', 'High(t-1)', 'Low(t-1)', 'Volume(t-1)',
            'Open(t-2)', 'Close(t-2)', 'High(t-2)', 'Low(t-2)', 'Volume(t-2)',
            'Open(t-3)', 'Close(t-3)', 'High(t-3)', 'Low(t-3)', 'Volume(t-3)']].values
    y = df['Close'].values

    try:
        x_full_train = df.drop(['Close', 'Datetime', 'High', 'Low', 'Volume'], axis = 1)
    except:
        x_full_train = df.drop(['Close', 'Date', 'High', 'Low', 'Volume'], axis = 1)
        
    y_full_train = df['Close']

    regressor = LinearRegression()

    model = regressor.fit(x_full_train, y_full_train)

    data_for_next_close = [current, 
                            df['Open'].iloc[-1], df['Close'].iloc[-1], df['High'].iloc[-1], df['Low'].iloc[-1], df['Volume'].iloc[-1],
                            df['Open(t-1)'].iloc[-1], df['Close(t-1)'].iloc[-1], df['High(t-1)'].iloc[-1], df['Low(t-1)'].iloc[-1], df['Volume(t-1)'].iloc[-1],
                            df['Open(t-2)'].iloc[-1], df['Close(t-2)'].iloc[-1], df['High(t-2)'].iloc[-1], df['Low(t-2)'].iloc[-1], df['Volume(t-2)'].iloc[-1]]

    prediction = model.predict([data_for_next_close])
    return(prediction[0])

tickerList = input("Tickers:").split(",")
period = "max"
interval = "5d"

period2 = "2y"
interval2 = "1d"


for item in tickerList:
    ticker = yf.Ticker(item)
    ticker_info = ticker.get_fast_info()
    current_price = ticker_info.get("lastPrice")

    df_min = ticker.history(period = '5d', interval = '1m')
    df_day = ticker.history(period = '5d', interval = '1d')

    EOW = get_projected(item, "max", "5d", df_day['Open'].iloc[-4])
    EOD = get_projected(item, "2y", "1d", df_min['Open'].iloc[-tdf.minutes_since_930am_est()])

    print("Week Open: " + str(df_day.index[-4]))
    print("Day Open: " + str(df_min.index[-tdf.minutes_since_930am_est()]))
    
    print(colored(f"{item}: {current_price}", 'light_cyan'))
    print(f"{item} --> " + "EOW: " + (colored('Buy ','green') if current_price < EOW else colored('Sell', 'red')) + f" --> {EOW}")
    print(f"{item} --> " + "EOD: " + (colored('Buy ','green') if current_price < EOD else colored('Sell', 'red')) + f" --> {EOD}")