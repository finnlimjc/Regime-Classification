import pandas as pd
import numpy as np

def wedge_volume(df:pd.DataFrame, window_size:int=60):
    """
    Calculate the wedge volume using the Gram matrix approach. Assumes window_data is a 2D array where rows are time points and columns are assets.
    Note that we take the absolute value of the determinant to ensure we get a non-negative volume, 
    as numerical issues can sometimes lead to a slightly negative determinant for a positive semidefinite matrix.
    
    Volume = 0 means linear dependence (Crisis mode) while volume = 1 means orthogonality (Perfect Diversification).
    """
    wedge_volumes = []
    data = df.values
    total_len = len(data)
    for i in range(window_size, total_len):
        window_data = data[i-window_size:i]
        
        # Normalize vectors to isolate direction from magnitude
        norms = np.linalg.norm(window_data, axis=0)
        norms[norms == 0] = 1e-8 # Prevent divide-by-zero
        normalized_window = window_data / norms
        
        # Calculate the Gram matrix and the wedge volume
        gram_mat = normalized_window.T @ normalized_window
        det = np.linalg.det(gram_mat)
        wedge_volume = np.sqrt(np.abs(det)) # Absolute to ensure non-negativity due to numerical issues
        wedge_volumes.append(wedge_volume)
    
    result = np.full(total_len, np.nan)
    result[window_size:] = wedge_volumes
    result = pd.Series(result, index=df.index)
    return result