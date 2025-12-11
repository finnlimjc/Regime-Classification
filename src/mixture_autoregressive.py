import pandas as pd
import numpy as np
from scipy.stats import norm

class GaussianMAR:
    def __init__(self, data:pd.Series, n_components:int, phi_orders:list[int], tol:float=1e-6, max_iter:int=200, 
                 alpha:np.ndarray[float]=None, sigma:np.ndarray[float]=None, phi:list[np.ndarray[float]]=None, seed:int=None):
        
        if n_components != len(phi_orders):
            raise ValueError(f"Number of components must match the number of phi_orders.")
        
        self.y = data.to_numpy()
        self.n = len(self.y)
        self.n_components = n_components
        self.phi_orders = phi_orders
        
        self.tol = tol
        self.max_iter = max_iter
        
        self.rng = np.random.default_rng(seed)
        self.params = {
            'alpha': alpha,
            'sigma': sigma,
            'phi': phi
        }
        self.initial_guess()
        
        self.min_timestep = max(phi_orders)
        self.train_n = len(self.y) #for AIC/BIC calculation
    
    def initial_guess(self):
        if self.params['alpha'] is None:
            self.params['alpha'] = np.ones(self.n_components)/ self.n_components #must sum to 1
        
        if self.params['sigma'] is None:
            self.params['sigma'] = np.ones(self.n_components)
        
        if self.params['phi'] is None:
            self.params['phi'] = [self.rng.standard_normal(order+1) for order in self.phi_orders] #Account for order 0 (the intercept)
    
    def _reverse_index(self, x:np.ndarray, from_pt:int, to_pt:int) -> np.ndarray:
        if to_pt == 0:
            return x[from_pt::-1]
        
        return x[from_pt:to_pt-1:-1]

    def e_step(self, y:np.ndarray, alpha:np.ndarray, sigma:np.ndarray, phi:list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:        
        tau = np.zeros(shape=(self.n, self.n_components))
        epsilon = tau.copy()
        for t in range(self.min_timestep, self.n):
            for k in range(self.n_components):
                p = phi[k]
                start_pt = t - self.phi_orders[k]
                y_lags = np.array([1] + self._reverse_index(y, t-1, start_pt).tolist()) #1 for phi_0, and remaining is in reversed order to get t-1, t-2 ..., t-n
                eps = y[t] - p.T@y_lags
                
                numerator = (alpha[k]/sigma[k])* norm.pdf(eps/ sigma[k])
                
                epsilon[t, k] = eps
                tau[t, k] = numerator
            
            tau[t, :] /= np.sum(tau[t, :]) #normalize to sum to 1
        
        return epsilon, tau #(timesteps, n_components)
    
    def log_likelihood(self, alpha:np.ndarray[float], sigma:np.ndarray[float], epsilon:np.ndarray) -> float:
        filtered_epsilon = epsilon[self.min_timestep:]
        component_densities = (alpha/sigma)* np.exp(-(filtered_epsilon**2)/ (2*sigma**2))
        mixture_density = np.sum(component_densities, axis=1) #sum columns
        return np.sum(np.log(mixture_density))
    
    def _maximize_alpha(self, tau_k:np.ndarray, phi_order_k:int) -> float:
        filtered_tau = tau_k[phi_order_k+1:]
        numerator = np.sum(filtered_tau)
        denominator = self.n-phi_order_k
        return numerator/denominator
    
    def _maximize_sigma(self, tau_k:np.ndarray, epsilon_k:np.ndarray, phi_order_k:int) -> float:
        filtered_tau = tau_k[phi_order_k+1:]
        filtered_epsilon = epsilon_k[phi_order_k+1:]
        numerator = np.sum(filtered_tau* filtered_epsilon**2)
        denominator = np.sum(tau_k)
        return np.sqrt(numerator/denominator)
    
    def _maximize_phi(self, y:np.ndarray, tau_k:np.ndarray, phi_order_k:int) -> np.ndarray:
        W = np.diag(tau_k[phi_order_k:])
        Y = y[phi_order_k:]
        
        y_lags = [y[phi_order_k-p-1: self.n-p-1] for p in range(phi_order_k)] #(p, n-p)
        X = np.column_stack([np.ones(self.n-phi_order_k)] + y_lags) #(n-p, p+1)
        
        XT_W = X.T @ W
        A = XT_W @  X
        b = XT_W @ Y
    
        return np.linalg.solve(A, b) #More stable than finding the inverse
    
    def m_step(self, tau:np.ndarray, epsilon:np.ndarray) -> dict[np.ndarray[float], np.ndarray[float], list[np.ndarray[float]]]:
        alpha, sigma, phi = [], [], []
        for k in range(self.n_components):
            alpha_hat = self._maximize_alpha(tau_k=tau[:, k], phi_order_k=self.phi_orders[k])
            sigma_hat = self._maximize_sigma(tau_k=tau[:, k], epsilon_k=epsilon[:, k], phi_order_k=self.phi_orders[k])
            phi_hat = self._maximize_phi(y=self.y, tau_k=tau[:, k], phi_order_k=self.phi_orders[k])
            
            alpha.append(alpha_hat)
            sigma.append(sigma_hat)
            phi.append(phi_hat)
        
        params = {
            'alpha': np.array(alpha),
            'sigma': np.array(sigma),
            'phi': phi
        }
        return params
    
    def fit(self) -> dict[np.ndarray[float], np.ndarray[float], list[np.ndarray[float]]]:
        self.log_likelihood_vals = []
        curr_log_likelihood = 0
        for i in range(self.max_iter):
            epsilon, self.tau = self.e_step(self.y, **self.params)
            
            old_log_likelihood = curr_log_likelihood
            curr_log_likelihood = self.log_likelihood(self.params['alpha'], self.params['sigma'], epsilon)
            self.log_likelihood_vals.append(curr_log_likelihood)
            if np.abs(curr_log_likelihood - old_log_likelihood) <= self.tol:
                print("Parameters have converged.")
                return self.params
            
            self.params = self.m_step(tau=self.tau, epsilon=epsilon)
        
        print("Max iteration has been reached.")
        return self.params
    
    @property
    def fitted_values(self):
        col_names = [str(f"cluster_{k}") for k in range(self.n_components)]
        df = pd.DataFrame(self.tau, columns=col_names)
        return df
    
    def greedy_assignment(self, tau:np.ndarray) -> pd.Series:
        col_names = [str(f"cluster_{k}") for k in range(self.n_components)]
        df = pd.DataFrame(tau, columns=col_names)
        labels = df.idxmax(axis=1).str[-1].astype(int)
        labels = labels.where(df.nunique(axis=1) > 1, -1)
        return labels
    
    def predict(self, y:float|list|np.ndarray|pd.Series, update_history:bool=False) -> np.ndarray:
        new_y = np.asarray(y).reshape(-1) #1D array
        n = len(new_y)
        tau = np.zeros((n, self.n_components))
        
        for t in range(n):
            for k in range(self.n_components):
                p = self.params['phi'][k]
                phi_order = self.phi_orders[k]
                
                if phi_order == 0:
                    y_lags = np.array([1])
                elif phi_order > 0:
                    y_lags = self.y[-phi_order:]
                    y_lags = np.array([1] + y_lags[::-1].tolist())
                eps = new_y[t] - p.T @ y_lags
                
                alpha = self.params['alpha'][k]
                sigma = self.params['sigma'][k]
                numerator = (alpha/sigma)* norm.pdf(eps/sigma)
                tau[t, k] = numerator
            
            tau[t, :] /= np.sum(tau[t, :])
        
        if update_history:
            self.update_history(y)
        
        return tau
    
    def update_history(self, y:float|list|np.ndarray|pd.Series):
        new_y = np.asarray(y).reshape(-1)
        self.y = np.append(self.y, new_y)
        self.n = len(self.y)
    
    @property
    def aic(self):
        val = -2*self.log_likelihood_vals[-1] +\
            2*(3*self.n_components - 1 + np.sum(self.phi_orders))
        return val
    
    @property
    def bic(self):
        val = -2*self.log_likelihood_vals[-1] +\
            np.log(self.train_n - self.min_timestep) *\
            (3*self.n_components - 1 + np.sum(self.phi_orders))
        return val