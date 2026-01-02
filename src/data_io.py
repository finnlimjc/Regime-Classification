import numpy as np
import pandas as pd
import yfinance as yf

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

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

class VolatilityYF(YahooFinance):
    def __init__(self, symbol, start_date, end_date, interval = '1d'):
        super().__init__(symbol, start_date, end_date, interval)
    
    def _remove_multi_index(self, target_df:pd.DataFrame) -> pd.DataFrame:
        df = target_df.copy()
        df.columns = df.columns.get_level_values(0)
        return df
    
    def pipeline(self) -> pd.DataFrame:
        prices = self._download_data(**self.params)
        df = self._remove_multi_index(prices)
        df['log_return'] = self._pct_change(df)
        df = self._reset_index(df)
        return df

class AlpacaStockData:
    def __init__(self, api_key:str, secret_key:str):
        self.client = StockHistoricalDataClient(api_key, secret_key)
    
    def _convert_timeframe(self, frequency:int, timeframe:str) -> TimeFrame:
        allowed_timeframe = ('minutes', 'hours', 'daily')
        if timeframe not in allowed_timeframe:
            raise ValueError(f"Timeframe must be one of the allowed values: {allowed_timeframe}")
        
        if timeframe == allowed_timeframe[0]: timeframe = TimeFrame(int(frequency), TimeFrameUnit.Minute)
        if timeframe == allowed_timeframe[1]: timeframe = TimeFrame(int(frequency), TimeFrameUnit.Hour)
        if timeframe == allowed_timeframe[2]: timeframe = TimeFrame(int(frequency), TimeFrameUnit.Day)
        return timeframe
    
    def get_stock_bar(self, symbol:str, start:str='2024-01-01', end:str='2024-12-31', frequency:int=5, timeframe:str='minutes') -> pd.DataFrame:
        timeframe, frequency = str(timeframe).lower(), int(frequency)
        timeframe = self._convert_timeframe(frequency=frequency, timeframe=timeframe)
        
        request = StockBarsRequest(
            symbol_or_symbols=str(symbol),
            timeframe=timeframe,
            start=str(start),
            end=str(end)
        )
        
        return self.client.get_stock_bars(request).df