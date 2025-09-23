import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import statsmodels.api as sm
import warnings
from termcolor import colored

warnings.filterwarnings("ignore", category=UserWarning)

def get_projected(tickerName, period, interval, current):
    ticker = yf.Ticker(tickerName)
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

    prediction = regressor.predict([data_for_next_close])
    return(prediction[0])

tickerList = input("Tickers:").split(",")
period = "max"
interval = "1h"

period2 = "2y"
interval2 = "1d"

for item in tickerList:
    ticker = yf.Ticker(item)
    ticker_info = ticker.get_fast_info()
    current_price = ticker_info.get("lastPrice")

    EOH = get_projected(item, "max", "1h", float(input(f"{item} hour open:")))
    EOD = get_projected(item, "2y", "1d", float(input(f"{item} day open:")))
    EODA = get_projected(item, "2y", "1d", current_price)


    print(colored(f"{item}: {current_price}", 'light_cyan'))
    print(f"{item} --> " + "EOH: " + (colored('Buy ','green') if current_price < EOH else colored('Sell', 'red')) + f" --> {EOH}")
    print(f"{item} --> " + "EOD: " + (colored('Buy ','green') if current_price < EOD else colored('Sell', 'red')) + f" --> {EOD}")
    print(f"{item} --> " + "EODA: " + (colored('Buy ','green') if current_price < EODA else colored('Sell', 'red')) + f" --> {EODA}")
    print()