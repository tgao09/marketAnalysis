from typing import List

import requests
from bs4 import BeautifulSoup

def get_sp500_tickers() -> List[str]:

    try:
        # read s&p 500 companies from wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
            )
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', id='constituents')
        if table is None:
            tables = soup.find_all('table')
            if not tables:
                raise ValueError('Could not find any tables on the Wikipedia page.')
            table = tables[0]

        tickers = []
        for row in table.select('tbody tr'):
            cells = row.find_all('td')
            if not cells:
                continue
            ticker = cells[0].get_text(strip=True)
            if ticker:
                tickers.append(ticker)

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