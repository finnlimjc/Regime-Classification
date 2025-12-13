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

5. Alternatively, if you have already completed the virtual environment and packages installation, run the appropriate commands as the following example:
```sh
cd myproject
.venv\Scripts\activate
python -m streamlit run app.py
```

# References
Wong, C. S., & Li, W. K. (2000). On a Mixture Autoregressive Model. Journal of the Royal Statistical Society. Series B (Statistical Methodology), 62(1), 95–115. http://www.jstor.org/stable/2680680