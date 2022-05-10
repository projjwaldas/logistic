"""
- Check correlation of numeric variables in a dataframe
- Check VIF of variables
- Check CSI of variables
"""

import pandas as pd
import numpy as np
import scipy as sp
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .measure import csi

def corr_check(df, resp_var, corr_cutoff):
    """
    Parameters
    ----------
        df: pandas dataframe
        resp_var: str, response variable
        corr_cutoff: float, correlation cutoff in decimal
    Returns
    -------
        pandas dataframe with predictor variables having higher correlation with other predictor variables, and based on their explanatory powers    
    """
    # Compute Correlations of Dependent variable with Predictor Variables
    rank_df = df.select_dtypes(include=np.number).corr().reset_index()[['index', resp_var]]
    rank_df['inform'] = rank_df[resp_var]**2
    rank_df.sort_values(['inform', 'index'], ascending=[False, True], inplace=True)
    rank_df.reset_index(drop=True, inplace=True)
    rank_df['rank'] = rank_df.index + 1
    rank_df.rename(columns={'index': 'var1', resp_var: 'corr_dep'}, inplace=True)
    rank_df.drop(['corr_dep', 'inform'], axis=1, inplace=True)
    
    # Compute Correlation Matrix
    corr_matrix = df.select_dtypes(include=np.number).drop(resp_var, axis=1).corr().reset_index()
    corr_matrix_l = corr_matrix.melt(id_vars='index')
    corr_matrix = corr_matrix_l[(corr_matrix_l['index'] > corr_matrix_l['variable']) &
                                (corr_matrix_l['value'].abs() > float(corr_cutoff))]
    corr_matrix.rename(columns={'index': 'var1', 'variable': 'var2', 'value': 'corr_val'}, inplace=True)
    
    # Get Ranks of Individual Variables
    del_var_df = corr_matrix.merge(rank_df, on='var1', how='left')
    del_var_df = del_var_df.merge(rank_df, left_on='var2', right_on='var1', how='left')
    del_var_df.rename(columns={'var1_x': 'var1', 'rank_x': 'var1_rank', 'rank_y': 'var2_rank'}, inplace=True)
    del_var_df.drop('var1_y', axis=1, inplace=True)
    
    # Final Data Createion
    del_var_df['del_var'] = np.where(del_var_df['var1_rank'] > del_var_df['var2_rank'], del_var_df['var1'], del_var_df['var2'])
    return(del_var_df)


def compute_csi(dev_df, val_df, resp_var, id_varlist):
    """
    Parameters
    ----------
        dev_df: pandas dataframe, development data
        val_df: pandas dataframe, validation data
        resp_var: str, response variable name
        id_varlist: list of ID variables
    Returns
    -------
        pandas dataframe with CSI of variables
    """
    val_df = val_df[dev_df.columns.tolist()]
    csi_var_list = [x for x in dev_df.columns.tolist() if x not in id_varlist+[resp_var]]
    
    # Compute CSI
    csi_df = csi(dev_df, val_df, csi_var_list, resp_var)
    csi_df = csi_df[['var_name', 'csi_var']].drop_duplicates()
    csi_df.columns = ['feature', 'CSI']
    csi_df.sort_values('CSI', ascending=False, inplace=True)
    csi_df['CSI'] = csi_df['CSI'].apply(lambda x: np.round(x, 2))
    return(csi_df)


def check_vif(df):
    """
    Parameters
    ----------
        df: pandas dataframe
    Returns
    -------
        returns a pandas dataframe VIF values of variables present in the input dataframe
    """
    # Initialise VIF Output Dataframe
    vif_df = pd.DataFrame()
    vif_df['feature'] = df.columns
    
    # Calculate VIF for each feature
    vif_df['VIF'] = [np.round(variance_inflation_factor(df.values, i), 2) for i in range(len(df.columns))]
    vif_df.sort_values('VIF', ascending=False, inplace=True)
    return(vif_df)