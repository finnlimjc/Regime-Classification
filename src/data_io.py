import numpy as np
import pandas as pd
import yfinance as yf

class YahooFinance:
    def __init__(self, symbol:str, start_date:str, end_date:str, interval:str='1d'):
        self.params = {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'interval': interval
        }
        self.col_names = ['date', 'adj_close', 'log_return']
    
    def _download_data(self, symbol:str, start_date:str, end_date:str, interval:str='1d') -> pd.DataFrame:
        prices = yf.download(symbol, start=start_date, end=end_date, interval=interval, auto_adjust=False)
        prices.index = prices.index.date
        return prices
    
    def _filter_cols(self, prices:pd.DataFrame) -> pd.DataFrame:
        filtered_prices = prices['Adj Close'].copy()
        filtered_prices.columns = ['Adj Close']
        return filtered_prices
    
    def _pct_change(self, filtered_prices:pd.DataFrame) -> pd.Series:
        df = filtered_prices.copy()
        log_return = np.log(df['Adj Close']/ df['Adj Close'].shift(1))
        return log_return
    
    def _reset_index(self, filtered_prices:pd.DataFrame) -> pd.DataFrame:
        df = filtered_prices.reset_index(names='date', drop=False).copy()
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%d/%m/%Y')
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y') #Convert from string to datetime
        return df
    
    def pipeline(self) -> pd.DataFrame:
        prices = self._download_data(**self.params)
        df = self._filter_cols(prices)
        df['log_return'] = self._pct_change(df)
        df = self._reset_index(df)
        df.columns = self.col_names
        return df