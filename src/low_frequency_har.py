import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class RecursiveHAR:
    """
    RecursiveHAR implements a low-frequency Heterogeneous Autoregressive (HAR) model 
    to forecast realized volatility using daily, weekly, and monthly lagged components.
    For the forecasting method, it uses the rolling window to update its parameters.
    Subsequently, it uses the forecasted value to re-fit and forecast the next timestep.

    Input:
        volatility_measure : A time series of realized volatility estimates (e.g. Rogers-Satchell, Garman-Klass, Parkinson).
        weekly : Number of observations to use for the weekly rolling average (typically 5 for trading days).
        monthly : Number of observations to use for the monthly rolling average (typically 22 for trading days).
    """
    def __init__(self, volatility_measure:pd.Series, weekly:int=5, monthly:int=22):
        self.data = volatility_measure.copy()
        self.days_by_period = {
            'weekly': weekly,
            'monthly': monthly
        }
        
        self.df = self._prepare_features()
    
    def _prepare_features(self) -> pd.DataFrame:
        prev_day = self.data.shift(1)
        weekly = self.data.rolling(self.days_by_period['weekly']).mean().shift(1) #current volatility cannot rely on a rolling average using its own value 
        monthly = self.data.rolling(self.days_by_period['monthly']).mean().shift(1)
        col_names = ['volatility', 'prev_day', 'weekly_rolling_avg', 'monthly_rolling_avg']
        df = pd.concat([self.data, prev_day, weekly, monthly], axis=1)
        df.columns = col_names
        return df.dropna()
    
    def _rolling_ols(self, X:np.ndarray, y:np.ndarray, window:int=1000) -> dict:
        lr = LinearRegression(fit_intercept=True)
        preds, betas, intercepts = [], [], []
        
        for k in range(window, len(y)):
            X_train = X[k-window:k] #right side excludes k-th data point
            y_train = y[k-window:k]
            model = lr.fit(X_train, y_train)
            
            X_test = X[[k]]
            y_pred = model.predict(X_test)
            preds.append(y_pred[0])
            betas.append(model.coef_)
            intercepts.append(model.intercept_)
        
        results = {
            'preds': preds,
            'betas': betas,
            'intercepts': intercepts
        }
        
        return results
    
    def backtest(self, window:int=1000) -> pd.DataFrame:
        X = self.df.iloc[:, 1:].values
        y = self.df.iloc[:, 0].values
        results = self._rolling_ols(X, y, window)
        
        #Create Dataframe
        preds = pd.Series(results['preds'], index=self.df.index[window:], name='y_hat')
        betas = pd.DataFrame(results['betas'], index=self.df.index[window:], columns=self.df.columns[1:])
        betas = betas.add_prefix("beta_")
        intercepts = pd.Series(results['intercepts'], index=self.df.index[window:], name='intercept')
        self.eval = pd.concat([betas, intercepts, preds, self.df.iloc[window:, 0]], axis=1)
        return self.eval
    
    def fit(self, window:int=1000):
        self.window = window
        self.fitted_values = self.df.iloc[-window:, 1:]
        y = self.df.iloc[-window:, 0]
        self.model = LinearRegression(fit_intercept=True).fit(self.fitted_values.values, y.values)
    
    def _process_next_day_features(self, latest_data:np.ndarray) -> np.ndarray:
        prev_day = latest_data[-1]
        weekly = latest_data[-self.days_by_period['weekly']:].mean()
        monthly = latest_data[-self.days_by_period['monthly']:].mean()
        return np.array([[prev_day, weekly, monthly]]) #(1, 3)
    
    def forecast(self, steps:int=1) -> np.ndarray:
        if steps < 1:
            raise ValueError("Number of steps must be more than or equal to 1.")
        
        latest_data = self.data.iloc[-self.days_by_period['monthly']:].values.flatten()
        preds = []
        for _ in range(steps):
            X = self._process_next_day_features(latest_data)
            pred = self.model.predict(X)[0]
            preds.append(pred)
            
            #Update latest data
            latest_data[:-1] = latest_data[1:]
            latest_data[-1] = pred
        
        return np.array(preds)
    
    @property
    def mse(self) -> float:
        """The average squared error between the HAR model and the target volatility measure, not the intraday Realized Volatility."""
        if not hasattr(self, "eval"):
            raise ValueError("Evaluate the model using .backtest() before calling this method.")
        
        squared_error = (self.eval.iloc[:, -1] - self.eval.iloc[:, -2])**2
        mse = squared_error.mean()
        return mse
    
class DirectHAR(RecursiveHAR):
    """
    DirectHAR implements a low-frequency Heterogeneous Autoregressive (HAR) model 
    to forecast realized volatility using daily, weekly, and monthly lagged components.
    For the forecasting method, it uses the rolling window to update its parameters.

    Key differences vs RecursiveHAR:
    - Uses DIRECT multi-horizon forecasting
    - Never feeds forecasted values back into regressors
    - Redefines the dependent variable as a future rolling average
    """
    def __init__(self, volatility_measure:pd.Series, weekly:int=5, monthly:int=22):
        super().__init__(volatility_measure, weekly, monthly)
    
    def _build_target(self, horizon:int) -> pd.Series:
        """Redefine the dependent variable as the future h-average volatility"""
        if horizon < 1:
            raise ValueError("Horizon must be more than or equal to 1.")
        
        new_target = self.data.rolling(horizon).mean()
        new_target = new_target.shift(-horizon+1) #Use the data today, to predict the average h-day future volatility
        return new_target.dropna()
    
    def backtest(self, horizon:int, window:int=1000) -> pd.DataFrame:
        target = self._build_target(horizon=horizon)
        df = self.df.iloc[:, 1:].join(target, how="inner") #Align Index
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        y.name = 'target'
        results = self._rolling_ols(X.values, y.values, window)
        
        #Create Dataframe
        betas = pd.DataFrame(results['betas'], index=X.index[window:], columns=X.columns).add_prefix("beta_")
        intercepts = pd.Series(results['intercepts'], index=X.index[window:], name='intercept')
        preds = pd.Series(results['preds'], index=X.index[window:], name='y_hat')
        self.eval = pd.concat([betas, intercepts, preds, y.iloc[window:]], axis=1)
        return self.eval
    
    def fit(self, horizon:int, window:int=1000):
        self.window = window
        
        #Filter only for what we need
        start_pt = -window+1-horizon
        X = self.df.iloc[start_pt:-horizon+1, 1:] if horizon > 1 else self.df.iloc[start_pt:, 1:]
        y = self._build_target(horizon).iloc[-window:]
        y.columns = ['target']
        self.fitted_values = pd.concat([X, y], axis=1)
        
        self.model = LinearRegression(fit_intercept=True).fit(X.values, y.values)
    
    def forecast(self) -> np.ndarray[float]:
        """Uses the latest data and the fitted regressors to predict the next h-day average volatility."""
        X = self.df.iloc[[-1], 1:].values
        pred = self.model.predict(X)[0]
        return pred