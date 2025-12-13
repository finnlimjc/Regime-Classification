import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import date

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.family': 'sans-serif'
})

class ParamsSelector:
    def select_stock_info(self) -> str:
        st.subheader("💰 Ticker Information")
        symbol = st.text_input("Yahoo Finance Ticker Symbol", value='SPY')
        return str(symbol)
    
    def select_date(self) -> tuple[str, str]:
        st.subheader("📅 Select Date Range")
        default_start = date(1960, 1, 1)
        default_end = date(2025, 12, 1)
        start_date = st.date_input("Start Date", default_start, min_value=default_start, max_value=default_end)
        end_date = st.date_input("End Date", default_end, min_value=default_start, max_value=default_end)
        
        if start_date > end_date:
            st.error("Start date must be before end date.")
            return None
        
        # For YahooFinance
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")
        
        return (start_date, end_date)
    
    def select_mar_params(self) -> tuple[int, int]:
        st.subheader("📋 MAR Parameters")
        max_components = st.number_input("Maximum Number of Components", value=3)
        max_ar = st.number_input("Maximum AR Order", value=3)
        return (max_components, max_ar)
    
    def render(self) -> tuple[dict, dict]:
        with st.sidebar:
            st.header("⚙️ Parameter Selector")
            symbol = self.select_stock_info()
            start_date, end_date = self.select_date()
            
            st.divider()
            max_components, max_ar = self.select_mar_params()
            
            yf_params = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date
            }
            
            mar_params = {
                'max_components': max_components,
                'max_ar': max_ar
            }
            
            return (yf_params, mar_params)

class Charts:
    def __init__(self, data:pd.DataFrame, n_components:int, label_col_name:str='cluster'):
        self.data = data.dropna()
        self.n_components = n_components
        self.label_col_name = label_col_name
        self.palette = sns.color_palette("husl", n_components)
        self._create_df_groups()
    
    def _create_df_groups(self):
        self.groups = []
        for i in range(self.n_components):
            grp = self.data.copy()
            grp.loc[grp[self.label_col_name] != i] = np.nan 
            self.groups.append(grp)
    
    def plot_line_charts(self, target_component:str, title:str, target_col_name:str) -> plt.Figure:
        fig, axes = plt.subplots(figsize=(8,2))
        
        axes.set_title(title)
        axes.plot(self.data[target_col_name], color='black', label='Underlying', linestyle='--', alpha=0.2)
        if target_component == 'All':
            for i in range(self.n_components):
                df = self.groups[i]
                axes.scatter(x=df.index, y=df[target_col_name], color=self.palette[i], label=f'Cluster {i}', s=0.3) 
        
        if target_component != 'All':
            target_data = self.groups[int(target_component)]
            axes.scatter(x=target_data[target_col_name].index, y=target_data[target_col_name], color=self.palette[target_component], label=f'Cluster {target_component}', s=0.3) 
        
        axes.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        return fig
    
    def plot_hist(self, title:str, target_col_name:str) -> plt.Figure:
        fig, axes = plt.subplots(figsize=(8,2))
        
        axes.set_title(title)
        axes.hist(self.data[target_col_name], alpha=0.2, label='Underlying', density=True, color='black')
        for i in range(self.n_components):
            df = self.groups[i]
            axes.hist(df[target_col_name], alpha=0.5, label=f'Cluster {i}', density=True, color=self.palette[i])
        
        axes.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        return fig