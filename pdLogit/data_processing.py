"""
- Basic Data preparation, including dropping of variables, variable type change etc.
- Create a simple EDA
- Create groups for categorical variables with higher number of classes
- Copy development data types to validation/ scoring data
- Missing value imputation
- Replace original features with dummy features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
import statistics

from .feature_selection import corr_check

def check_data_consistency(dev_data, val_data, char_col_list):
    """
    Parameters
    ----------
        dev_data: pandas dataframe, development data
        val_data: pandas dataframe, validation data
    Returns
    -------
        checks if there are any extra columns in the development and validation data
        pandas dataframe, features having values in validation data that are not present in development data 
    """
    # Columns Present in Dev Data but not in Val Data
    probVarList1 = [x for x in dev_data.columns if x not in val_data.columns]
    if len(probVarList1) > 0:
        print(f'The following columns are not present in Validation Data: {probVarList1}')
        
    # Columns Present in Val Data but not in Dev Data
    probVarList2 = [x for x in val_data.columns if x not in dev_data.columns]
    if len(probVarList2) > 0:
        print(f'The following columns are not present in Development Data: {probVarList2}')
        
    if len(probVarList1)+len(probVarList2) == 0:
        print('Development & Validation data have the same columns')
        
    # Create a dataframe of character variables having values in Val data that do not appear in Dev data
    val_issue_df = pd.DataFrame(columns=['Variable', 'Values', 'Replace Value'])
    for col in char_col_list:
        _var_df = val_data.loc[~val_data[col].isin(dev_data[col].tolist())][[col]].rename(columns={col: 'Values'})
        _var_df['Variable'] = col
        val_issue_df = val_issue_df.append(_var_df)
    val_issue_df['Replace Value'] = np.nan
    return(val_issue_df[['Variable', 'Values', 'Replace Value']])


def pre_process_data(data, drop_varlist):
    """
    Parameters
    ----------
        data: pandas dataframe
        drop_varlist: list of variables to be dropped from the development data
    Returns
    -------
        drops unnecessary columns, removes leading and trailing blanks from all columns and returns pandas dataframe
    """
    # Drop Variables
    drop_varlist = [x for x in drop_varlist if x in data.columns]
    if len(drop_varlist) > 0:
        print(f'Dropping {len(drop_varlist)} vars as per config file.')
        data.drop(drop_varlist, axis=1, inplace=True)
        
    # Remove Leading and trailing Blanks
    data = data.replace(r"^ +| +$", r'', regex=True)
    data = data.replace({col: {'': np.nan} for col in data.columns})
    data = data.rename(columns=lambda x: x.strip())
    
    return(data)


def nmiss_nunique_check(dev_df, val_df, miss_cutoff=0.95):
    """
    Parameters
    ----------
        dev_df: pandas dataframe, development data
        val_df: pandas dataframe, validation data
        miss_cutoff: optional parameter; float, missing value cut-off
    Returns
    -------
        dev_df: pandas dataframe, development data with reduced features after missing value and degenerate value check
        val_df: pandas dataframe, validation data with reduced features after missing value and degenerate value check
        nmiss_df: pandas dataframe, summary of missing values for all variables, both dev & val
        drop_reason_df: pandas dataframe, summary of feature reduction
    """
    # Missing % - Dev
    nmiss_dev_df = pd.DataFrame(dev_df.isnull().sum().rename('dev_nmiss')).rename_axis('feature').reset_index()
    nmiss_dev_df['dev_nmiss_pct'] = nmiss_dev_df['dev_nmiss']/nmiss_dev_df.index.size
    
    # Missing % - Val
    nmiss_val_df = pd.DataFrame(val_df.isnull().sum().rename('val_nmiss')).rename_axis('feature').reset_index()
    nmiss_val_df['val_nmiss_pct'] = nmiss_val_df['val_nmiss']/nmiss_val_df.index.size
    
    # Fianl Missing % Summary Data
    nmiss_df = nmiss_dev_df.merge(nmiss_val_df, on='feature', how='outer')
    drop_miss_df = nmiss_df[(nmiss_df['dev_nmiss_pct'] > miss_cutoff) | (nmiss_df['val_nmiss_pct'] > miss_cutoff)]
    drop_miss_df['drop_reason'] = np.where(drop_miss_df['dev_nmiss_pct'] > miss_cutoff, 'high_missing_pct_dev', 'high_missing_pct_val')
    drop_miss_df = drop_miss_df[['feature', 'drop_reason']]
    print(f'Missing Value cut-off being used: {miss_cutoff}. Variables dropped: {drop_miss_df.index.size}')
    
    # Degenerate Variable Check
    degen_drop_var_dev_df = pd.DataFrame([col for col in dev_df.columns if dev_df[col].nunique()==1 and col not in drop_miss_df['feature'].tolist()], columns=['feature'])
    degen_drop_var_dev_df['drop_reason'] = 'degenerate_feature_dev'
    degen_drop_var_val_df = pd.DataFrame([col for col in val_df.columns if val_df[col].nunique()==1 and col not in drop_miss_df['feature'].tolist()+degen_drop_var_dev_df['feature'].tolist()], 
                                         columns=['feature'])
    degen_drop_var_val_df['drop_reason'] = 'degenerate_feature_val'
    drop_degen_df = degen_drop_var_dev_df.append(degen_drop_var_val_df)
    print(f'Degenerate Variables (Variables having only one value) Dropped: {drop_degen_df.index.size}')
    
    
    # Drop Variables from Data
    drop_reason_df = drop_miss_df.append(drop_degen_df)
    dev_df.drop(drop_reason_df['feature'].tolist(), axis=1, inplace=True)
    val_df.drop(drop_reason_df['feature'].tolist(), axis=1, inplace=True)
    
    return(dev_df, val_df, nmiss_df, drop_reason_df)


def create_mv_impute_input(data):
    """
    Parameters
    ----------
        data: pandas dataframe
    Returns
    -------
        pandas dataframe with missing value imputation rules: numeric variables: -9999999, character variables: 'Missing'
        This output needs to be further modified by the user if required for customised imputations
    """
    # Get Data Types of Variables
    dtypes_df = pd.DataFrame(data.dtypes, columns=['dtype']).rename_axis('feature').reset_index()
    dtypes_df['method'] = 'constant'
    dtypes_df['value'] = np.where(dtypes_df['feature'].isin(data.select_dtypes(include=np.number).columns), -9999999, 'Missing')
    
    # Add Missing %
    nmiss = data.isna().mean().rename('miss_pct')
    dtypes_df = dtypes_df.merge(nmiss, how='left', left_on='feature', right_index=True)
    dtypes_df = dtypes_df[dtypes_df['miss_pct'] > 0]
    
    return(dtypes_df)


def impute_missing_values(data, imp_df):
    """
    Parameters
    ----------
        data: pandas dataframe, that needs to be missing value imputed
        imp_df: pandas dataframe, instructions for missing value impytation at feature level
    Returns
    -------
        pandas dataframe with missing value imputed
    """
    imp_df.dropna(subset=['method'], inplace=True)
    imp_df = imp_df[imp_df['feature'].isin(data.columns)]
    method = {r.feature: r.method for i, r in imp_df.loc[imp_df.method!='constant', ['feature', 'method']].iterrows()}
    missing_values = {k: data[k].apply(v) if v!='mode' else data[k].apply(v)[0] for k, v in method.items()}
    constant_values = {r.feature: data[r.feature].dtype.type(r.value) for i, r in imp_df.loc[imp_df.method=='constant', ['feature', 'dtype', 'value']].iterrows()}
    values = {**missing_values, **constant_values}
    data.fillna(value=values, inplace=True)
    return(data)


def get_exploratory_analysis(df, features):
    """
    Parameters
    ----------
        df: pandas dataframe for which summary data needs to be created
        features: list of features for which summary data needs to be created
    Returns
    -------
        pandas dataframe of EDA of input dataframe features
    """
    nmiss = df[features].isna().sum().rename('missing')
    desc = df[features].describe(include='all').T.merge(nmiss, how='left', left_index=True, right_index=True).rename_axis('features').reset_index()
    desc['missing_pct'] = desc['missing']/desc.index.size
    return(desc)


def create_cat_groups(df, resp_var, varlist):
    """
    Parameters
    ----------
        df: pandas dataframe
        resp_var: str, name of response variable
        varlist: list of categorical variables
    Returns
    -------
        pandas dataframe with multiple 'similar' classes of the categorical variables grouped into same clusters
    """
    man_enc_map_df = pd.DataFrame()
    for var in varlist:
        summary_df = pd.DataFrame(df.groupby(var).agg({resp_var: ['count', 'sum']})).reset_index()
        summary_df.columns = ['class', 'n_cnt', 'resp_rate']
        summary_df.sort_values('n_cnt', ascending=False, inplace=True, ignore_index=True)
        summary_df['n_pct'] = summary_df['n_cnt']/summary_df['n_cnt'].sum()
        summary_df['feature'] = var
        
        # Merge Groups based on Obs %
        summary_df['grp'] = summary_df.index + 1
        for i in range(1, len(summary_df.index)):
            summary_df.loc[i, 'grp'] = np.where((summary_df.loc[i, 'n_pct'] < 0.05) & (summary_df.loc[i-1, 'class'] != 'Missing'), summary_df.loc[i-1, 'grp'], summary_df.loc[i, 'grp'])
        man_enc_map_df = man_enc_map_df.append(summary_df[['feature', 'class', 'n_cnt', 'n_pct', 'resp_rate', 'grp']])
        return(man_enc_map_df)
    
    
def create_dummy_features(dev_df, val_df, resp_var, id_varlist):
    """
    Parameters
    ----------
        dev_df: pandas dataframe - development data
        val_df: pandas datafrane - validation data
        resp_var: str, name of response variable
        id_varlist: list of ID variables
    Returns
    -------
        pandas dataframe with One-Hot Encoded variables (dev & val)
        ordinal encoder object
        one-hot encoder object
    """
    # Create List of predicting variables
    non_pred_varlist = id_varlist + [resp_var]
    varList = [x for x in dev_df.columns if x not in non_pred_varlist]
    
    # Variable Encoding Dictionary
    man_enc_dict = {col: {val: 'd'+str(idx) for idx, val in enumerate(sorted(dev_df[col].unique()))} for col in dev_df.columns if col not in non_pred_varlist}
    
    # Manual Encoding
    dev_df[list(man_enc_dict.keys())] = pd.DataFrame({col: dev_df[col].map(man_enc_dict[col]) for col in man_enc_dict.keys()})
    val_df[list(man_enc_dict.keys())] = pd.DataFrame({col: val_df[col].map(man_enc_dict[col]).fillna(statistics.mode(man_enc_dict[col].values())) for col in man_enc_dict.keys()})
    
    # One-Hot Encoding to Create Dummy Variables
    oh_enc = OneHotEncoder(handle_unknown='ignore')
    oh_enc.fit(dev_df[varList])
    
    # Transform Dev & Val Data
    dev_data_enc_df = pd.DataFrame(oh_enc.transform(dev_df[varList]).toarray())
    dev_data_enc_df.columns = oh_enc.get_feature_names_out().tolist()
    
    val_data_enc_df = pd.DataFrame(oh_enc.transform(val_df[varList]).toarray())
    val_data_enc_df.columns = oh_enc.get_feature_names_out().tolist()
    print(f'{dev_data_enc_df.shape[1]} Dummy Variables Created')
    
    # Rename variables
    new_names = [(col, 'L'+col) for col in dev_data_enc_df.columns]
    dev_data_enc_df.rename(columns=dict(new_names), inplace=True)
    val_data_enc_df.rename(columns=dict(new_names), inplace=True)
    
    # Add ID Variables
    dev_data_enc_df = pd.concat([dev_df[non_pred_varlist], dev_data_enc_df], axis=1)
    val_data_enc_df = pd.concat([val_df[non_pred_varlist], val_data_enc_df], axis=1)
    
    # Correlation Check
    print("Starting Correlation Check with 60% cut-off")
    corr_check_df = corr_check(dev_data_enc_df.drop(id_varlist, axis=1), resp_var, 0.6)
    corr_check_df['corr_val'] = corr_check_df['corr_val'].apply(lambda x: np.round(x, 4))
    
    # Drop Variables with Correlation
    corr_drop_varlist = corr_check_df['del_var'].drop_duplicates().tolist()
    dev_data_enc_df.drop(corr_drop_varlist, axis=1, inplace=True)
    val_data_enc_df.drop(corr_drop_varlist, axis=1, inplace=True)
    print(f'Variables Dropped: {len(corr_drop_varlist)}. Updated Development Data Shape: {dev_data_enc_df.shape}')
    
    return(dev_data_enc_df, val_data_enc_df, man_enc_dict, oh_enc)