import pandas as pd
import numpy as np

class LowFrequencyVolatility:
    def __init__(self, df:pd.DataFrame):
        self.df = df.copy()
    
    def rogers_satchell_volatility(self, close:float, high:float, low:float, open:float) -> float:
        first_term = np.log(high/open)* np.log(high/close)
        second_term = np.log(low/open)* np.log(low/close)
        return np.sqrt(first_term + second_term)
    
    def garman_klass_volatility(self, close:float, high:float, low:float, open:float) -> float:
        first_term = 0.5* np.log(high/low)**2
        second_term = (2*np.log(2) - 1)* np.log(close/open)**2
        return np.sqrt(first_term - second_term)
    
    def parkinson_volatility(self, high:float, low:float) -> float:
        squared_log_hl = np.log(high/low)**2
        denom = 4*np.log(2)
        return np.sqrt(squared_log_hl/denom)
    
    def average_volatility(self, close:float, high:float, low:float, open:float) -> float:
        numerator = self.parkinson_volatility(high, low) + self.garman_klass_volatility(close, high, low, open) + self.rogers_satchell_volatility(close, high, low, open)
        return numerator/3
    
    def pipeline(self) -> pd.DataFrame:
        self.df['rogers_satchell'] = self.rogers_satchell_volatility(self.df['Close'], self.df['High'], self.df['Low'], self.df['Open'])
        self.df['garman_klass'] = self.garman_klass_volatility(self.df['Close'], self.df['High'], self.df['Low'], self.df['Open'])
        self.df['parkinson_volatility'] = self.parkinson_volatility(self.df['High'], self.df['Low'])
        self.df['average_volatility'] = self.average_volatility(self.df['Close'], self.df['High'], self.df['Low'], self.df['Open'])
        return self.df