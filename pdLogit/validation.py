"""
- Compute Decile Rank Orders of Model
- Compute concordance of model
- Compute Leaderboard for model
"""

import pandas as pd
import numpy as np
from bisect import bisect_left, bisect_right

from .measure import standard_metrics

def get_rank_order_data(model, X, y):
    """
    Parameters
    ----------
        model: statsmodels model object
        X: pandas dataframe of predictor variables
        y: pandas series of response variable
    Returns
    -------
        pandas dataframe of the rank ordering of the deciles
    """
    # Get Predictions
    TruthTable = pd.DataFrame()
    TruthTable['Prob_1'] = model.predict(X)
    TruthTable['resp'] = y
    
    # Rank Ordering
    TruthTable.sort_values('Prob_1', ascending=False, inplace=True)
    TruthTable.reset_index(drop=True, inplace=True)
    TruthTable['decile'] = pd.qcut(TruthTable.index, q=10, labels=[1,2,3,4,5,6,7,8,9,10])
    
    rank_order_df = TruthTable.groupby('decile').agg({'resp': ['count', 'sum']}).reset_index()
    rank_order_df.columns = ['decile', 'obs_cnt', 'resp_cnt']
    rank_order_df['resp_pct'] = rank_order_df['resp_cnt']/rank_order_df['resp_cnt'].sum()
    rank_order_df['cum_resp_pct'] = rank_order_df['resp_pct'].cumsum()
    
    # Format Columns
    rank_order_df['resp_pct'] = rank_order_df['resp_pct'].apply(lambda x: np.round(x, 4))
    rank_order_df['cum_resp_pct'] = rank_order_df['cum_resp_pct'].apply(lambda x: np.round(x, 4))
    
    return(rank_order_df)


def get_concordance(model, X, y):
    """
    Parameters
    ----------
        model: statsmodels model object
        X: pandas dataframe of predictor variables
        y: pandas series of response variable
    Returns
    -------
        python dictionary object with concordance stats
    """
    # Get Predictions
    TruthTable = pd.DataFrame()
    TruthTable['Prob_1'] = model.predict(X)
    TruthTable['resp'] = y
    
    zeros = TruthTable[(TruthTable['resp']==0)].reset_index(drop=True)
    ones = TruthTable[(TruthTable['resp']==1)].reset_index(drop=True)

    zeros_list = sorted([zeros.iloc[j]['Prob_1'] for j in zeros.index])
    zeros_length = len(zeros_list)
    disc = 0
    ties = 0
    conc = 0
    
    for i in ones.index:
        cur_conc = bisect_left(zeros_list, ones.iloc[i]['Prob_1'])
        cur_ties = bisect_right(zeros_list, ones.iloc[i]['Prob_1']) - cur_conc
        conc += cur_conc
        ties += cur_ties
    pairs_tested = zeros_length * len(ones.index)
    disc = pairs_tested - conc - ties
    
    # Create Dictionary Object
    concordance_dict = {}
    concordance_dict['pct_concordance'] = np.round((conc/pairs_tested)*100, 2)
    concordance_dict['pct_discordance'] = np.round((disc/pairs_tested)*100, 2)
    concordance_dict['pct_ties'] = np.round((ties/pairs_tested)*100, 2)
    concordance_dict['pairs_tested'] = pairs_tested

    return(concordance_dict)


def create_leaderboard(X_train, y_train, X_test, y_test, resp_var, model):
    """
    Parameters
    ----------
        X_train: pandas dataframe, predictor variables - training data
        y_train: pandas series, response variable - training data
        X_test: pandas dataframe, predictor variables - test data
        y_test: pandas series, response variable - test data
        resp_var: str, name of response variable
        model: statsmodels model object
    Returns
    -------
        python dictionary object with model performance metrics
    """
    y_pred_train = model.predict(X_train).rename('Probability')
    dev_scored_df = pd.concat([y_train, y_pred_train], axis=1)
    dev_cuts = dev_scored_df['Probability'].quantile(np.arange(0, 10+1)/10).reset_index(drop=True)
    dev_dw = _decilewise_counts(dev_scored_df, resp_var, cutpoints=dev_cuts)
    
    dev_dw_dict = {'S1_Total': dev_dw.total.sum(),
                   'S1_Resp_Total': dev_dw.resp.sum(),
                   'S1_Resp_Top2Dec': dev_dw[:2].resp.sum(),
                   'S1_Top2Dec_Capture_Rate': dev_dw[:2].resp.sum()/dev_dw.resp.sum(),
                   'S1_RO_Break_Decile': dev_dw[dev_dw.resp.diff().fillna(0)>0].index.min(),
                   'S1_Rank_Order_Flag': 1 if dev_dw.resp.diff().fillna(0).max()>0 else 0,
                   'S1_Capture_Rate_Before_RO_Break': dev_dw.resp[:int(np.nan_to_num(dev_dw[dev_dw.resp.diff().fillna(0)>0].index.min(), nan=10))].sum()/dev_dw.resp.sum()}
    dev_dw_pop_dict = {'S1CountD'+str(i+1): r.total for i, r in dev_dw.iterrows()}
    dev_dw_resp_dict = {'S1RespD'+str(i+1): r.resp for i, r in dev_dw.iterrows()}
    dev_dw_counts = {**dev_dw_dict, **dev_dw_pop_dict, **dev_dw_resp_dict}
    dev_dw_pop = (dev_dw.total/dev_dw.total.sum())
    dev_dw_resp = (dev_dw.resp/dev_dw.resp.sum())
    dev_dw_cutoffs = {'S1CutoffD'+str(i+1): v for i, v in enumerate(dev_cuts[::-1])}
    
    y_pred_test = model.predict(X_test).rename('Probability')
    val_scored_df = pd.concat([y_test, y_pred_test], axis=1)
    
    val_dw = _decilewise_counts(val_scored_df, resp_var, cutpoints=dev_cuts)
    val_dw_dict = {'S2_Total': val_dw.total.sum(),
                   'S2_Resp_Total': val_dw.resp.sum(),
                   'S2_Resp_Top2Dec': val_dw[:2].resp.sum(),
                   'S2_Top2Dec_Capture_Rate': val_dw[:2].resp.sum()/val_dw.resp.sum(),
                   'S2_RO_Break_Decile': val_dw[val_dw.resp.diff().fillna(0)>0].index.min(),
                   'S2_Rank_Order_Flag': 1 if val_dw.resp.diff().fillna(0).max()>0 else 0,
                   'S2_Capture_Rate_Before_RO_Break': val_dw.resp[:int(np.nan_to_num(val_dw[val_dw.resp.diff().fillna(0)>0].index.min(), nan=10))].sum()/val_dw.resp.sum()}
    val_dw_pop_dict = {'S1CountD'+str(i+1): r.total for i, r in val_dw.iterrows()}
    val_dw_resp_dict = {'S1RespD'+str(i+1): r.resp for i, r in val_dw.iterrows()}
    val_dw_counts = {**val_dw_dict, **val_dw_pop_dict, **val_dw_resp_dict}
    val_dw_pop = (val_dw.total/val_dw.total.sum())
    val_dw_resp = (val_dw.resp/val_dw.resp.sum())
    psi = _quick_psi(dev_dw_pop, val_dw_pop)
    
    dev_metrics = standard_metrics(dev_scored_df, resp_var, 'Probability')
    val_metrics = standard_metrics(val_scored_df, resp_var, 'Probability')
    
    metrics_dict = {'Gini_S1': dev_metrics['gini'],
                    'Gini_S2': val_metrics['gini'],
                    'GiniVariance': abs(round((dev_metrics['gini']-val_metrics['gini'])/(dev_metrics['gini']+0.00001)*100, 2)),
                    'KS_S1': dev_metrics['ks'],
                    'KS_S2': val_metrics['ks'],
                    'PSI': psi,
                    'S1_TN': dev_metrics['tn'],
                    'S1_FP': dev_metrics['fp'],
                    'S1_FN': dev_metrics['fn'],
                    'S1_TP': dev_metrics['tp'],
                    'S1_Precision': dev_metrics['precision'],
                    'S1_Recall': dev_metrics['recall'],
                    'S1_F1_Score': dev_metrics['f1_score'],
                    'S2_TN': val_metrics['tn'],
                    'S2_FP': val_metrics['fp'],
                    'S2_FN': val_metrics['fn'],
                    'S2_TP': val_metrics['tp'],
                    'S2_Precision': val_metrics['precision'],
                    'S2_Recall': val_metrics['recall'],
                    'S2_F1_Score': val_metrics['f1_score']
                   }
    kpi_master = {**metrics_dict, **dev_dw_counts, **dev_dw_cutoffs, **val_dw_counts}
    leaderboard_df = pd.DataFrame([kpi_master]).T.rename_axis('metrics').reset_index()
    leaderboard_df.columns = ['Metrics', 'Values']
    return(leaderboard_df)


def _decilewise_counts(df, resp_col, prediction_col='Probability', bins=10, cutpoints=None):
    """
    Returns a summarised pandas dataframe with total and responders for each decile based on positive probability
    """
    if cutpoints is None:
        cutpoints = df[prediction_col].quantile(np.arange(0, bins+1)/bins).reset_index(drop=True)
        cutpoints = list(set([0]+list(cutpoints[1:-1])+[1]))
    df['bins'] = pd.cut(df[prediction_col], cutpoints, duplicates='drop')
    out_df = df.groupby('bins')[resp_col].agg(['count', 'sum']).sort_values(by=['bins'], ascending=False).reset_index()
    out_df.columns = ['bins', 'total', 'resp']
    return(out_df)


def _quick_psi(dev, val):
    """Calculates PSI from 2 arrays"""
    try:
        return(sum([(a-b)*np.log(a/b) for (a, b) in zip(dev, val)]))
    except:
        return(-99.0)