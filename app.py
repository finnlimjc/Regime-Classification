from src.dashboard_design import *
from src.mixture_autoregressive import *
from src.yahoo_finance import *

@st.cache_data
def get_data(symbol:str, start_date:str, end_date:str) -> pd.DataFrame:
    yf = YahooFinance(symbol, start_date, end_date)
    df = yf.pipeline()
    df['first_diff'] = df['adj_close'].diff(1)
    df = df.dropna().reset_index(drop=True)
    return df

def select_params(data:pd.Series, max_components:int, max_ar:int, tol:float=1e6, max_iter:int=1000, seed:int=123, quiet:bool=True) -> dict[int, tuple[int]]:
    grid_result = grid_search(data=data, max_components=max_components, max_ar=max_ar, tol=tol,
                              max_iter=max_iter, seed=seed, quiet=quiet)
    return get_best_params(grid_result=grid_result, criteria='bic') 

@st.cache_resource
def fit_mar_model(data:pd.Series, mar_params:dict) -> tuple[OptimizedMAR, dict[int, tuple[int]]]:
    best_params = select_params(data, **mar_params)
    mar = OptimizedMAR(data, **best_params, max_iter=1000)
    mar.fit()
    return mar, best_params

if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.header("📊 Regime Classification Model")
    
    selector = ParamsSelector()
    yf_params, mar_params = selector.render()
    df = get_data(**yf_params)
    
    #Use log returns instead of first difference of raw price
    mar, best_params = fit_mar_model(df['log_return'], mar_params)
    df['cluster'] = mar.greedy_assignment(mar.tau)
    df.set_index('date', drop=True, inplace=True)
    
    c = Charts(df, best_params['n_components'], 'cluster')
    dist_return_fig = c.plot_hist('Distribution of Log Return', 'log_return')
    st.pyplot(dist_return_fig) #Filter not compatible with densities 
    plt.close(dist_return_fig)
    
    selected_cluster = st.radio(
        "Highlight cluster",
        options=["All"] + list(range(best_params['n_components'])),
        horizontal=True,
    )
    
    adj_close_fig = c.plot_line_charts(selected_cluster, "Adjusted Close Time Series", 'adj_close')
    st.pyplot(adj_close_fig)
    plt.close(adj_close_fig)
    
    log_return_fig = c.plot_line_charts(selected_cluster, "Log Return", 'log_return')
    st.pyplot(log_return_fig)
    plt.close(log_return_fig)