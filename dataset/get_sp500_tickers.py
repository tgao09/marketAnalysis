import requests
import pandas as pd

def get_sp500_tickers():
    
    try:
        # read s&p 500 companies from wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        
        # the first table contains the current s&p 500 companies
        sp500_table = tables[0]
        
        # extract ticker symbols
        tickers = sp500_table['Symbol'].tolist()
        
        # clean up tickers (remove any whitespace)
        tickers = [ticker.strip() for ticker in tickers if ticker.strip()]
        
        print(f"Found {len(tickers)} S&P 500 tickers")
        
        return tickers
        
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        return []

def save_tickers_to_file(tickers, filename='training_stocks.txt'):
    
    if tickers:
        ticker_string = ','.join(tickers)
        with open(filename, 'w') as f:
            f.write(ticker_string)
        print(f"Saved {len(tickers)} tickers to {filename}")
    else:
        print("No tickers to save")

if __name__ == "__main__":
    # get s&p 500 tickers
    tickers = get_sp500_tickers()
    
    # save to file
    save_tickers_to_file(tickers)
    
    # print first 10 tickers as sample
    if tickers:
        print(f"\nFirst 10 tickers: {', '.join(tickers[:10])}")
        print(f"Last 10 tickers: {', '.join(tickers[-10:])}")