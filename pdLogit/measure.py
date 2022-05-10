"""
- Calculate model performance metrcis
- Chi Square with IV
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score, precision_recall_curve

from ._woeBinningPandas import woe_binning, woe_binning2, woe_binning3

def decilewise_counts(df, resp_col, prediction_col='positive_probability', bins=10, cutpoints=None):
    """
    Returns a summarised pandas dataframe with total and responders for each decile based on positive probability
    """
    if cutpoints is None:
        cutpoints = df[prediction_col].quantile(np.arrange(0, bins+1)/bins).reset_index(drop=True)
        cutpoints = list(set([0]+list(cutpoints[1:-1])+[1]))
    df['bins'] = pd.cut(df[prediction_col], cutpoints, duplicate='drop')
    out_df = df.groupby('bins')[resp_col].agg(['count', 'sum']).sort_values(by=['bins'], ascending=False).reset_index()
    out_df.columns = ['band', 'total', 'resp']
    return(out_df)


def standard_metrics(df, resp_col, prediction_col='positive_probability'):
    """
    Returns metrics like KS, Gini, Optimal Threshold, TN, FP, FN, TP, Precision, Recall and F1 score from Metrics object
    """
    metrics = Metrics(df, resp_col, prediction_col)
    return(metrics.to_dict())


def woe_bins(df, var_name, resp_name, suffix='_dev', var_cuts=None):
    """
    Returns
    -------
        pandas dataframe for var_cuts string, total, responders, non responders, var_name (with _dev or _val suffix)
        list of interval items onto be used on val file
    """
    df1 = df[[resp_name, var_name]]
    if (np.issubdtype(df1[var_name].dtype, np.number)):
        n = df1[var_name].nunique()
        if var_cuts is None:
            suffix='_dev'
            var_cuts = woe_binning3(df1, resp_name, var_name, 0.05, 0.00001, 0, 50, 'bad', 'good')
            var_cuts = list(set(var_cuts))
            var_cuts.sort()
        df1['var_binned'] = pd.cut(df1[var_name], var_cuts, right=True, labels=None, retbins=False, precision=10, include_lowest=False)
        var_min = float(df1[var_name].min())
        var_max = float(df1[var_name].max())
        summ_df = df1.groupby('var_binned')[resp_name].agg(['count', 'sum']).reset_index()
        summ_df['delta'] = summ_df['count'] - summ_df['sum']
        summ_df['var_name'] = var_name
        summ_df.columns = ['var_cuts', 'total'+suffix, 'responders'+suffix, 'non_responders'+suffix, 'var_name']
        summ_df['var_cuts_string'+suffix] = summ_df.var_cuts.apply(lambda x: str(x.left if x.left!=-np.inf else var_min)+' To '+str(x.right if x.right!=np.inf else var_max))
    else:
        df1[var_name].fillna('Blank', inplace=True)
        summ_df = df1.groupby(var_name)[resp_name].agg(['count', 'sum']).reset_index()
        summ_df['delta'] = summ_df['count'] - summ_df['sum']
        summ_df['var_name'] = var_name
        summ_df.columns = ['var_cuts_string'+suffix, 'total'+suffix, 'responders'+suffix, 'non_responders'+suffix, 'var_name']
        summ_df['var_cuts'] = summ_df['var_cuts_string'+suffix]
    return(summ_df[summ_df['total'+suffix]!=0], var_cuts)


def iv_var(df, var_name, resp_name, var_cuts=None):
    """
    Returns IV dataframe and IV value of a given variable
    """
    suffix = '_dev' if var_cuts is None else '_val'
    iv_df, _ = iv(df, var_name, resp_name, var_cuts)
    return(iv_df, iv_df['iv'+suffix].sum())


def iv(df, var_list, resp_name, var_cuts=None):
    """
    Returns a pandas dataframe with calculated fields - resp_rate, perc_dist, perc_non_resp, perc_resp, raw_odds, ln_odds, iv, exp_resp, exp_non_resp, chi_square
    """
    dfs = []
    cuts = {}
    for var_name in var_list:
        if var_cuts is None:
            suffix = '_dev'
            summ_df, cut = woe_bins(df, var_name, resp_name, '_dev')
        else:
            suffix = '_val'
            summ_df, cut = woe_bins(df, var_name, resp_name, '_val', var_cuts[var_name])
        dfs.append(summ_df)
        cuts[var_name] = cut
    idf = pd.concat(dfs, axis=0)
    idf['resp_rate'+suffix] = (idf['responders'+suffix]*100)/idf['total'+suffix]
    idf['perc_dist'+suffix] = (idf['total'+suffix]*100)/idf.groupby('var_name')['total'+suffix].transform('sum')
    idf['perc_non_resp'+suffix] = (idf['non_responders'+suffix]*100)/idf.groupby('var_name')['non_responders'+suffix].transform('sum')
    idf['perc_resp'+suffix] = (idf['responders'+suffix]*100)/idf.groupby('var_name')['responders'+suffix].transform('sum')
    idf['raw_odds'+suffix] = idf.apply(lambda r: 0 if r['perc_resp'+suffix]==0 else r['perc_non_resp'+suffix]/r['perc_resp'+suffix], axis=1)
    idf['ln_odds'+suffix] = idf['raw_odds'+suffix].apply(lambda x: 0 if abs(np.log(x))==np.inf else np.log(x))
    idf['iv'+suffix] = (idf['perc_non_resp'+suffix]-idf['perc_resp'+suffix])*idf['ln_odds'+suffix]/100
    idf['exp_resp'+suffix] = idf['total'+suffix]*idf.groupby('var_name')['responders'+suffix].transform('sum')/idf.groupby('var_name')['total'+suffix].transform('sum')
    idf['exp_non_resp'+suffix] = idf['total'+suffix]*idf.groupby('var_name')['non_responders'+suffix].transform('sum')/idf.groupby('var_name')['total'+suffix].transform('sum')
    idf['chi_square'+suffix] = (((idf['responders'+suffix]-idf['exp_resp'+suffix])**2)/idf['exp_resp'+suffix])+(((idf['non_responders'+suffix]-idf['exp_non_resp'+suffix])**2)/idf['exp_non_resp'+suffix])
    return(idf, cuts)


def _quick_psi(dev, val):
    """
    Calculates PSI from 2 arrays - dev and val
    """
    try:
        return(sum([(a-b)*np.log(a/b) for (a,b) in zip(dev, val)]))
    except:
        return(-99.0)
    
    
def psi(dev, val, target='positive_probability', n_bins=10):
    """
    Returns a pandas dataframe with **psi** column after creating 10 deciles.
    Code includes creating score calculation using **round(500-30xlog(100x(p/(1-p))), 0)** where p is probability.
    We need to pass both dev and val at same time to apply same bins created on dev data
    """
    dev['score'] = dev[target].apply(lambda x: round(500-30*np.log2(100*(x/(1-x))), 0))
    val['score'] = val[target].apply(lambda x: round(500-30*np.log2(100*(x/(1-x))), 0))
    
    _, bins = pd.qcut(dev.score, n_bins, retbins=True, precision=0)
    bins = [int(i) if abs(i)!=np.inf else i for i in bins]
    dev['bins'] = pd.cut(dev.score, bins)
    val['bins'] = pd.cut(val.score, bins)
    
    dev_bins = dev.bins.value_counts(sort=False, normalize=True)
    val_bins = val.bins.value_counts(sort=False, normalize=True)
    
    psi_ = pd.concat([dev_bins, val_bins], axis=1)
    psi_.columns = ['dev', 'val']
    psi_['psi'] = (psi_.dev-psi_.val)*np.log(psi_.dev/psi_.val)
    return(psi_)


def csi(dev_df, val_df, var_list, resp_name):
    """
    Returns a pandas dataframe with **csi, csi_var, perc_csi** columns (Characteristic Stability Index) calculated based on both dev and val dataframes
    """
    dev, var_cuts = iv(dev_df, var_list, resp_name)
    val, _ = iv(val_df, var_list, resp_name, var_cuts)
    
    final = pd.merge(dev, val, how='left', on=['var_name', 'var_cuts'], suffixes=['_dev', '_val'])
    
    final['csi'] = ((final['perc_dist_dev']-final['perc_dist_val'])/100)*np.log(final['perc_dist_dev']/final['perc_dist_val'])
    final['csi_var'] = final.groupby('var_name')['csi'].transform('sum')
    final['perc_csi'] = (100*final.groupby('var_name')['csi'].transform('cumsum'))/final.groupby('var_name')['csi'].transform('sum')
    return(final)


class Metrics:
    def __init__(self, df, resp_col, prediction_col):
        self.df = df
        self.target = resp_col
        self.actual = df[resp_col]
        self.predicted = df[prediction_col]
        self.gains = self.calculate_gains()
        self.ks = self.ks()
        self.gini = self.gini()
        self.threshold = self.get_threshold()
        self.tn, self.fp, self.fn, self.tp, self.precision, self.recall, self.f1_score = self.precision_recall_f1_score()
        
    def calculate_gains(self):
        """
        Returns a pandas dataframe with responders, non responders, cumulative totals, percentage totals and percentage cumulative totals
        """
        self.df['scaled_score'] = (self.predicted*1000000).round(0)
        gains = self.df.groupby('scaled_score')[self.target].agg(['count', 'sum'])
        gains.columns = ['total', 'responders']
        gains.reset_index(inplace=True)
        gains = gains.sort_values(by='scaled_score', ascending=False)
        gains['non_responders'] = gains['total'] - gains['responders']
        gains['cum_resp'] = gains['responders'].cumsum()
        gains['cum_non_resp'] = gains['non_responders'].cumsum()
        gains['total_resp'] = gains['responders'].sum()
        gains['total_non_resp'] = gains['non_responders'].sum()
        gains['perc_resp'] = gains['responders']/gains['total_resp']
        gains['perc_non_resp'] = gains['non_responders']/gains['total_non_resp']
        gains['perc_cum_resp'] = gains['perc_resp'].cumsum()
        gains['perc_cum_non_resp'] = gains['perc_non_resp'].cumsum()
        gains['k_s'] = gains['perc_cum_resp']-gains['perc_cum_non_resp']
        return(gains)
    
    def get_threshold(self):
        """
        Returns a threshold cutoff value from `sklearn.metrics.roc_curve` using actual and predicted values
        """
        fpr, tpr, threshold = roc_curve(self.actual, self.predicted)
        gmean = np.sqrt(tpr*(1-fpr))
        youdenJ = tpr-fpr
        threshold_gmean = round(threshold[np.argmax(gmean)], 4)
        threshold_yJ = round(threshold[np.argmax(youdenJ)], 4)
        threshold_cutoff = max(threshold_gmean, threshold_yJ)
        return(threshold_cutoff)
    
    def ks(self):
        """
        Returns KS value calculated from Metrics.calculate_gains function
        """
        return(self.gains['k_s'].max())
    
    def gini(self):
        """
        Returns Gini value calculated from actual and predicted using `sklearn.metrics.roc_curve` and `sklearn.metrics.auc`
        """
        fpr, tpr, _ = roc_curve(self.actual, self.predicted)
        auroc = auc(fpr, tpr)
        gini = 2*auroc - 1
        return(gini)
    
    def precision_recall_f1_score(self):
        """
        Calculates TN, FP, FN, TP, Precision, Recall, F1 Score using Optimal Threshold value
        """
        threshold_cutoff = self.get_threshold()
        self.y_pred = np.where(self.predicted >= threshold_cutoff, 1, 0)
        tn, fp, fn, tp = confusion_matrix(self.actual, self.y_pred).ravel()
        precision = precision_score(self.actual, self.y_pred)
        recall = recall_score(self.actual, self.y_pred)
        f1 = f1_score(self.actual, self.y_pred)
        return(tn, fp, fn, tp, precision, recall, f1)
    
    def to_dict(self):
        """
        Returns all calculated metrics in a dict form
        """
        return({'ks': self.ks, 
                'gini': self.gini,
                'threshold': self.threshold,
                'tn': self.tn,
                'fp': self.fp,
                'fn': self.fn,
                'tp': self.tp,
                'precision': self.precision,
                'recall': self.recall,
                'f1_score': self.f1_score
               })
        