"""
- Create Coarse Classing Results
- Create Fine Classing Results
"""

import pandas as pd
import numpy as np
import scipy as sp
import math
from tqdm import tqdm

from ._woeBinningPandas import woe_binning, woe_binning3

def coarse_classing(data, woe_data, resp_var, var_list, special_val_list):
    """
    Parameters
    ----------
        data: pandas dataframe on which coarse classing will be computed
        woe_data: pandas dataframe
        resp_var: str, name of response variable
        var_list: list of variables for which coarse classing is required
    """
    # Initialise Dataframes
    num_classing_df = pd.DataFrame()
    char_classing_df = pd.DataFrame()
    
    # Convert Boolean Variables to Character
    bool_varlist = [x for x in data.columns if data[x].dtypes.kind == 'b']
    for var in bool_varlist:
        data[var] = data[var].astype(str)
        
    # Loop over all Variables and compute coarse classing
    for i in tqdm(range(len(var_list))):
        var_name = var_list[i]
        df1 = data[[resp_var, var_name]]
        
        # Binning for Numeric Variables
        if (np.issubdtype(data[var_name].dtypes, np.number)) or (str(data[var_name].dtypes)=='bool'):
            nunique_cnt = df1[var_name].nunique()
            try:
                if nunique_cnt>20:
                    binning_df = woe_binning(df1, resp_var, var_name, 0.05, 0.00001, 0, 50, 'bad')
                else:
                    binning_df = woe_binning(df1, resp_var, var_name, 0.001, 0.00001, 0, 50, 'bad')
                    
                var_cuts_except = binning_df['cutpoints_final'].tolist()
                df1['predictor_var_binned'] = pd.cut(df1[var_name], var_cuts_except, right=True, labels=None, retbins=False, precision=10, include_lowest=False)
                
            except:
                var_cuts_except = woe_binning3(df1, resp_var, var_name, 0.05, 0.00001, 0, 50, 'bad', 'good')
                df1['predictor_var_binned'] = pd.cut(df1[var_name], var_cuts_except, right=True, labels=None, retbins=False, precision=10, include_lowest=False)
                    
            # Get Lower & Upper Boundaries of Each Bin
            df1['bin_left'] = df1['predictor_var_binned'].apply(lambda x: x.left)
            df1['bin_right'] = df1['predictor_var_binned'].apply(lambda x: x.right)
            
            ############## Deal Special Values ##############
            special_val_list.sort()
            for val in special_val_list:
                if sum(df1[var_name] == val) > 0:
                    bin0_lower_val = df1[df1[var_name]<val][var_name].max()
                    bin0_lower_val = np.NINF if math.isnan(bin0_lower_val) else bin0_lower_val
                    df1['bin_left'] = np.where(df1[var_name] == val, bin0_lower_val, df1['bin_left'])
                    df1['bin_right'] = np.where(df1[var_name] == val, val, df1['bin_right'])
                    df1['bin_left'] = np.where((df1[var_name]>val) & (df1['bin_left']<val) & (df1['bin_right']>val), val, df1['bin_left'])
                    
                    if sum(df1[var_name] < val) > 0:
                        bin0_upper_val = df1[(df1[var_name] < val)][var_name].max()
                        df1['bin_right'] = np.where((df1[var_name] < val) & (df1['bin_left'] < val) & (df1['bin_right'] >= val), bin0_upper_val, df1['bin_right'])
                        
            # Open the right most boundary, if close
            if df1['bin_right'].max() != np.inf:
                df1['bin_right'] = np.where(df1['bin_right'] == df1['bin_right'].max(), np.inf, df1['bin_right'])
                
            # Construct New Bins
            df1['VAR_BINS'] = df1.apply(lambda row: pd.Interval(left=row['bin_left'], right=row['bin_right'], closed='right'), axis=1)
            
            ############## Special Values Dealing Ends Here ##############
            
            # Update WOE Dataset
            woe_data[var_name] = df1['VAR_BINS']
            
            # Create Summary Data
            summary_df = df1.groupby('VAR_BINS').agg({resp_var: ['count', 'sum']}).reset_index()
            summary_df.columns = ['VAR_BINS', 'TOT_ACTS', 'COUNT_RESP']
            summary_df['COUNT_NON_RESP'] = summary_df['TOT_ACTS'] - summary_df['COUNT_RESP']
            summary_df = summary_df[summary_df['TOT_ACTS']!=0]
            summary_df['VAR_NAME'] = var_name
            summary_df['lower_limit'] = summary_df['VAR_BINS'].apply(lambda x: df1[var_name].min() if x.left == np.NINF else x.left)
            summary_df['upper_limit'] = summary_df['VAR_BINS'].apply(lambda x: df1[var_name].max() if x.right == np.inf else x.right)
            
            num_classing_df = num_classing_df.append(summary_df)
            
            
        # Binning for Character Variables
        else:
            summary_df = df1.groupby(var_name).agg({resp_var: ['count', 'sum']}).reset_index()
            summary_df.columns = ['VAR_BINS', 'TOT_ACTS', 'COUNT_RESP']
            summary_df['COUNT_NON_RESP'] = summary_df['TOT_ACTS'] - summary_df['COUNT_RESP']
            summary_df['VAR_NAME'] = var_name
            
            char_classing_df = char_classing_df.append(summary_df)
            
    
    # WOE & IV Calculation
    coarse_classing_df = pd.concat([num_classing_df, char_classing_df], axis=0, ignore_index=True)
    
    coarse_classing_df_g = coarse_classing_df.groupby('VAR_NAME').agg({'TOT_ACTS': 'sum', 'COUNT_NON_RESP': 'sum', 'COUNT_RESP': 'sum'}).reset_index()
    coarse_classing_df_g.columns = ['VAR_NAME', 'Total_sum', 'Non_responders_sum', 'Responders_sum']
    coarse_classing_df = coarse_classing_df.merge(coarse_classing_df_g, on='VAR_NAME', how='inner')
    
    coarse_classing_df['RESP_RATE'] = coarse_classing_df['COUNT_RESP']/coarse_classing_df['TOT_ACTS']
    coarse_classing_df['ROWP_TOT'] = coarse_classing_df['TOT_ACTS']/coarse_classing_df['Total_sum']
    coarse_classing_df['PER_NON_RESP'] = coarse_classing_df['COUNT_NON_RESP']/coarse_classing_df['Non_responders_sum']
    coarse_classing_df['PER_RESP'] = coarse_classing_df['COUNT_RESP']/coarse_classing_df['Responders_sum']
    coarse_classing_df['RAW_ODDS'] = coarse_classing_df['PER_NON_RESP']/coarse_classing_df['PER_RESP']
    coarse_classing_df['RAW_ODDS'] = coarse_classing_df['RAW_ODDS'].apply(lambda x: 0 if x == np.inf or x == np.NINF else x)
    coarse_classing_df['LN_ODDS'] = np.log(coarse_classing_df['RAW_ODDS'])
    coarse_classing_df['LN_ODDS'] = coarse_classing_df['LN_ODDS'].apply(lambda x: 0 if x == np.inf or x == np.NINF else x)
    coarse_classing_df['INFO_VAL'] = (coarse_classing_df['PER_NON_RESP']-coarse_classing_df['PER_RESP'])*coarse_classing_df['LN_ODDS']
    
    coarse_classing_df['GP'] = coarse_classing_df['Non_responders_sum']/coarse_classing_df['Total_sum']
    coarse_classing_df['BP'] = coarse_classing_df['Responders_sum']/coarse_classing_df['Total_sum']
    coarse_classing_df['exp_bad'] = coarse_classing_df['TOT_ACTS']*coarse_classing_df['BP']
    coarse_classing_df['exp_good'] = coarse_classing_df['TOT_ACTS']*coarse_classing_df['GP']
    coarse_classing_df['CH_SQ'] = (((coarse_classing_df['COUNT_RESP']-coarse_classing_df['exp_bad'])**2)/coarse_classing_df['exp_bad'])+(((coarse_classing_df['COUNT_NON_RESP']-coarse_classing_df['exp_good'])**2)/coarse_classing_df['exp_good'])
    
    # Coarse Classing Bin Number
    coarse_classing_df = coarse_classing_df[coarse_classing_df['VAR_NAME'].isin(data.columns.tolist())]
    coarse_classing_df['FINE_BIN_NUM'] = coarse_classing_df.groupby('VAR_NAME').cumcount() + 1
    
    # Final Data Preparation
    keep_col_list = ['VAR_NAME', 'VAR_BINS', 'TOT_ACTS', 'ROWP_TOT', 'COUNT_RESP', 'PER_RESP', 'COUNT_NON_RESP', 'PER_NON_RESP', 'RAW_ODDS', 'LN_ODDS', 'INFO_VAL', 'CH_SQ', 'RESP_RATE', 'FINE_BIN_NUM']
    coarse_classing_df = coarse_classing_df[keep_col_list]
    
    return(coarse_classing_df)


def fine_classing(dev_df, val_df, c_class_df):
    """
    Parameters
    ----------
        dev_df: pandas dataframe, development data
        val_df: pandas dataframe, validation data
        c_class_df: pandas dataframe, Coarse Classing Data
    Returns
    -------
        returns pandas dataframe of fine classing data with WOE and IV values
    """
    # Initialise Dataframes
    num_classing_df = pd.DataFrame()
    char_classing_df = pd.DataFrame()
    
    # Convert Boolean Variables to Character
    bool_varlist = [x for x in dev_df.columns if dev_df[x].dtypes.kind == 'b']
    for var in bool_varlist:
        dev_df[var] = dev_df[var].astype(str)
        
    # Loop Over All Variables
    var_list = c_class_df['VAR_NAME'].drop_duplicates().tolist()
    for i in tqdm(range(len(var_list))):
        var_name = var_list[i]
        _c_class_df = c_class_df[c_class_df['VAR_NAME'] == var_name]
        
        # Binning for Numeric Variables
        if (np.issubdtype(dev_df[var_name].dtypes, np.number)) or (str(dev_df[var_name].dtypes)=='bool'):
            c_bin_list = _c_class_df['FINE_BIN_NUM'].drop_duplicates().tolist()
            c_bin_list.sort()
            bin_left_replace_val = np.NINF
            for bin in c_bin_list:
                _f_bins_df = _c_class_df[_c_class_df['FINE_BIN_NUM'] == bin].reset_index(drop=True)
                
                # Get Lower and Upper Boundaries for each Bin
                _f_bins_df['bin_left'] = _f_bins_df['VAR_BINS'].apply(lambda x: x.split(',')[0].replace('(', '')).astype(float)
                _f_bins_df['bin_right'] = _f_bins_df['VAR_BINS'].apply(lambda x: x.split(',')[1].replace(']', '')).astype(float)
                
                bin_left = _f_bins_df['bin_left'].min()
                bin_right = _f_bins_df['bin_right'].max()
                if (bin_left == bin_right):
                    bin_left = bin_left_replace_val
                bin_left_replace_val = bin_right
                _f_bins_df_g = _f_bins_df.groupby('VAR_NAME').sum()[['TOT_ACTS', 'COUNT_RESP', 'COUNT_NON_RESP']].reset_index()
                _f_bins_df_g['VAR_BINS'] = pd.Interval(left=bin_left, right=bin_right, closed='right')
                
                # Append to Master Data
                num_classing_df = num_classing_df.append(_f_bins_df_g)
                
            # Classify Original Data with Updated Bins - this will be required to replace original data values with WOE values
            num_classing_df['bin_left'] = num_classing_df['VAR_BINS'].apply(lambda x: x.left)
            num_classing_df['bin_right'] = num_classing_df['VAR_BINS'].apply(lambda x: x.right)
            var_df = num_classing_df[num_classing_df['VAR_NAME'] == var_name]
            
            # Create Cutpoints List
            cutpoints = var_df['bin_left'].tolist() + var_df['bin_right'].tolist()
            cutpoints = list(set(cutpoints))
            cutpoints.sort()
            
            # Update Original Values with Bins - Development
            dev_df[var_name] = dev_df[var_name].astype(float)
            dev_df['var_bin_lat'] = pd.cut(dev_df[var_name], cutpoints, labels=False, retbins=False, precision=10, include_lowest=False)+1
            dev_df.drop(var_name, axis=1, inplace=True)
            dev_df.rename(columns={'var_bin_lat': var_name}, inplace=True)
            
            # Update Original Values with Bins - Validation
            val_df[var_name] = val_df[var_name].astype(float)
            val_df['var_bin_lat'] = pd.cut(val_df[var_name], cutpoints, labels=False, retbins=False, precision=10, include_lowest=False)+1
            val_df.drop(var_name, axis=1, inplace=True)
            val_df.rename(columns={'var_bin_lat': var_name}, inplace=True)
            
        # Fine Classing for Character Variables
        else:
            
            # Get Bin List
            c_bin_list = _c_class_df['FINE_BIN_NUM'].drop_duplicates().tolist()
            c_bin_list.sort()
            
            for bin in c_bin_list:
                _f_bins_df = _c_class_df[_c_class_df['FINE_BIN_NUM'] == bin].reset_index(drop=True)
                _f_bins_df_g = _f_bins_df.groupby('VAR_NAME').sum()[['TOT_ACTS', 'COUNT_RESP', 'COUNT_NON_RESP']].reset_index()
                _f_bins_df_g['VAR_BINS'] = bin
                
                char_classing_df = char_classing_df.append(_f_bins_df_g)
                
            # Update Original Values with Bins - Development
            _c_class_df['VAR_BINS'] = _c_class_df['VAR_BINS'].astype(str)
            dev_df = dev_df.merge(_c_class_df[['VAR_BINS', 'FINE_BIN_NUM']], left_on=var_name, right_on='VAR_BINS', how='left')
            dev_df.drop([var_name, 'VAR_BINS'], axis=1, inplace=True)
            dev_df.rename(columns={'FINE_BIN_NUM': var_name}, inplace=True)
            
            # Update Original Values with Bins - Validation
            val_df = val_df.merge(_c_class_df[['VAR_BINS', 'FINE_BIN_NUM']], left_on=var_name, right_on='VAR_BINS', how='left')
            val_df.drop([var_name, 'VAR_BINS'], axis=1, inplace=True)
            val_df.rename(columns={'FINE_BIN_NUM': var_name}, inplace=True)
            
            
    # Compute WOE & IV
    fine_classing_df = pd.concat([num_classing_df, char_classing_df], axis=0, ignore_index=True)
    fine_classing_df.reset_index(drop=True, inplace=True)
    fine_classing_df['BIN_NUM'] = fine_classing_df.groupby('VAR_NAME').cumcount()+1
    fine_classing_df['BIN_NUM'] = np.where(fine_classing_df['bin_left'].isnull(), fine_classing_df['VAR_BINS'], fine_classing_df['BIN_NUM'])
    
    f_class_df_g = fine_classing_df.groupby('VAR_NAME').sum()[['TOT_ACTS', 'COUNT_NON_RESP', 'COUNT_RESP']].reset_index()
    f_class_df_g.columns = ['VAR_NAME', 'Total_sum', 'Non_responders_sum', 'Responders_sum']
    fine_classing_df = fine_classing_df.merge(f_class_df_g, on='VAR_NAME', how='inner')
    
    fine_classing_df['RESP_RATE'] = fine_classing_df['COUNT_RESP']/fine_classing_df['TOT_ACTS']
    fine_classing_df['ROWP_TOT'] = fine_classing_df['TOT_ACTS']/fine_classing_df['Total_sum']
    fine_classing_df['PER_NON_RESP'] = fine_classing_df['COUNT_NON_RESP']/fine_classing_df['Non_responders_sum']
    fine_classing_df['PER_RESP'] = fine_classing_df['COUNT_RESP']/fine_classing_df['Responders_sum']
    fine_classing_df['RAW_ODDS'] = fine_classing_df['PER_NON_RESP']/fine_classing_df['PER_RESP']
    fine_classing_df['RAW_ODDS'] = fine_classing_df['RAW_ODDS'].apply(lambda x: 0 if x == np.inf or x == np.NINF else x)
    fine_classing_df['LN_ODDS'] = np.log(fine_classing_df['RAW_ODDS'])
    fine_classing_df['LN_ODDS'] = fine_classing_df['LN_ODDS'].apply(lambda x: 0 if x == np.inf or x == np.NINF else x)
    fine_classing_df['INFO_VAL'] = (fine_classing_df['PER_NON_RESP']-fine_classing_df['PER_RESP'])*fine_classing_df['LN_ODDS']
    
    fine_classing_df['GP'] = fine_classing_df['Non_responders_sum']/fine_classing_df['Total_sum']
    fine_classing_df['BP'] = fine_classing_df['Responders_sum']/fine_classing_df['Total_sum']
    fine_classing_df['exp_bad'] = fine_classing_df['TOT_ACTS']*fine_classing_df['BP']
    fine_classing_df['exp_good'] = fine_classing_df['TOT_ACTS']*fine_classing_df['GP']
    fine_classing_df['CH_SQ'] = (((fine_classing_df['COUNT_RESP']-fine_classing_df['exp_bad'])**2)/fine_classing_df['exp_bad'])+(((fine_classing_df['COUNT_NON_RESP']-fine_classing_df['exp_good'])**2)/fine_classing_df['exp_good'])
    
    # Final Data Preparation
    keep_col_list = ['VAR_NAME', 'BIN_NUM', 'VAR_BINS', 'bin_left', 'bin_right', 'TOT_ACTS', 'ROWP_TOT', 'COUNT_RESP', 'PER_RESP', 'COUNT_NON_RESP', 'PER_NON_RESP', 'RAW_ODDS', 'LN_ODDS', 'INFO_VAL', 'CH_SQ', 'RESP_RATE']
    fine_classing_df = fine_classing_df[keep_col_list]
    
    return(fine_classing_df, dev_df, val_df)