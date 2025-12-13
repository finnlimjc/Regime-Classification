# Instructions
1. Create a virtual environment:
```sh
# Open a terminal and navigate to your project folder
cd myproject

# Create the .venv folder
python -m venv .venv
```

2. Activate the virtual environment:
```sh
# Windows command prompt
.venv\Scripts\activate.bat

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS and LinuxS
source .venv/bin/activate
```

3. Install packages in the environment:
```sh
python -m pip install -r requirements.txt
```

4. Run the dashboard:
```sh
streamlit run app.py
```

5. Alternatively, if you have already completed the virtual environment and packages installation, run the appropriate commands as follows:
```sh
cd myproject
.venv\Scripts\activate
python -m streamlit run app.py
```

# Dashboard
In the dashboard, a few assumptions were made to simplify the UI/UX:
- I(1) or Log Change is sufficient to make the data weakly stationary.
- The parameters found by the grid search, using BIC as the selection criterion, are optimal.
- Parameters will converge within 1000 iterations.

The following images show the MAR model fitted on the SPY from 1992 to 2025:
<img width="1860" height="450" alt="image" src="https://github.com/user-attachments/assets/c47101ee-f379-4d4a-9bd8-84d91cf2dc8d" />
<img width="1860" height="971" alt="image" src="https://github.com/user-attachments/assets/3e469d31-358b-48a9-a1f5-ce98676a0750" />
<img width="1860" height="971" alt="image" src="https://github.com/user-attachments/assets/5ca28c61-da41-454b-8ff0-7f1442345ce6" />
<img width="1860" height="971" alt="image" src="https://github.com/user-attachments/assets/60ca41db-57fa-4c70-b67a-b618a18b60d6" />

# MAR Model
Under the assumption that a finite mixture of autoregressive processes generates the observed time series, and that each regime (component) has its own autoregressive coefficients and variance, we can utilize the Gaussian Mixture Autoregressive (MAR) model to soft-cluster the regimes. The latent regime memberships are inferred using the Expectation-Maximization (EM) algorithm.

$$F(y_t | \mathcal{F}_{t-1}) = \sum^K_{k=1}\alpha_k\Phi\left(\frac{y_t - \phi_{k,0} - \phi_{k,1}y_{t-1} - ... -  \phi_{k,p_k}y_{t-p_k}}{\sigma_k}\right)$$

where $\alpha_k$ is the mixture weights, $\phi_{k,0}$ is the intercept, $\phi_{k, p_k}$ is the autoregressive coefficients, $\sigma_k$ is the standard deviation of the Gaussian noise in component k.

E-Step: 

$$\tau_{k,t} = \frac{(\alpha_k/\sigma_k)\phi(\epsilon_{k,t}/\sigma_k)}{\sum^K_{k=1}(\alpha_k/\sigma_k)\phi(\epsilon_{k,t}/\sigma_k)}$$

M-Step:

$$\hat{\alpha}_k = \frac{\sum^n_{t=p+1}\tau_{t,k}}{n-p}$$
$$\hat{\sigma}_k = \left(\frac{\sum^n_{t=p+1}\tau_{t,k}\epsilon^2_{t,k}}{\sum^n_{t=p+1}\tau_{t,k}}\right)^{1/2}$$
$$\hat{\phi}_k = (X^T_kW_kX_k)^{-1}X^T_kW_kY$$

For more information on the M-step formula, particularly $\hat{\phi}_k$, refer to "./notebooks/mixture_autoregressive_model.ipynb".

# References
Wong, C. S., & Li, W. K. (2000). On a Mixture Autoregressive Model. Journal of the Royal Statistical Society. Series B (Statistical Methodology), 62(1), 95–115. http://www.jstor.org/stable/2680680
