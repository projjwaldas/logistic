"""
- Fit a Decision Tree Recursive Feature Elimination (RFE) to identify top significant features
- Fit a Stepwise Logistic Regression
- Fit a Logistic Regression
"""

import pandas as pd
import numpy as np
import scipy as sp
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline
from scipy.stats import norm

from .data_processing import impute_missing_values
from .measure import standard_metrics


def get_RFE_features(data, varlist, resp_var, max_depth=5, min_samples_split=0.02, n_features=100):
    """
    Parameters
    ----------
        data: pandas.DataFrame
        varlist: list of predictor variables
        resp_var: str, name of response variable
    Returns
    -------
        list of top {n_features} variables
    """
    # Data Preparation
    y = data[[resp_var]]
    X = data.drop(resp_var, axis=1)
    mean_list = X.select_dtypes(include=np.number).mean()
    mode_list = X[X.columns.difference(mean_list.index)].mode().iloc[0]
    X.fillna(mean_list.append(mode_list), inplace=True)
    
    # Create Numeric and Character Data Separate
    char_selector = make_column_selector(dtype_include=object)
    num_selector = make_column_selector(dtype_include=np.number)
    
    # Preprocessor
    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    data_preprocessor = make_column_transformer((ord_enc, char_selector),
                                                remainder = 'passthrough')
    
    # RFE Object
    rfe = RFE(estimator=DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split), n_features_to_select=n_features)
    
    # Create Pipeline
    rfe_pipeline = make_pipeline(data_preprocessor, rfe)
    rfe_pipeline.fit(X, y)
    
    # Final Varlist
    RFE_varlist = X.columns[rfe_pipeline.get_params()['rfe'].get_support()].tolist()
    return(RFE_varlist)
    

def stepwise_selection(X, y, initial_list=[], threshold_in=0.05, threshold_out = 0.05, verbose=True):
    """
    Perform a forward-backward feature selection based on p-value from statsmodels.api.OLS
    Parameters
    ----------
        X: pandas.DataFrame with candidate features
        y: list-like with the target
        initial_list: list of features to start with (column names of X)
        threshold_in: include a feature if its p-value < threshold_in
        threshold_out: exclude a feature if its p-value > threshold_out
        verbose: whether to print the sequence of inclusions and exclusions
    Returns
    -------
        list of selected features 
    """
    stepwise_outdf = pd.DataFrame()
    included = list(initial_list)
    
    # Add Intercept Model
#     model = sm.OLS(y, sm.add_constant(pd.DataFrame(X['const']))).fit()
    model = sm.OLS(y, pd.DataFrame(X['const'])).fit()
    print('Add {:30} with p-value {:6}'.format('Intercept', model.pvalues['const']))
    included.append('const')
    
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
#             model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            model = sm.OLS(y, pd.DataFrame(X[included+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        new_pval_df = pd.DataFrame(new_pval, columns=['pval']).rename_axis('feature').reset_index()
        if best_pval < threshold_in:
            best_feature = new_pval_df.iloc[new_pval.argmin()]['feature']
#             best_feature = new_pval.argmin()
            included.append(best_feature)
            step_df = pd.DataFrame([[best_feature, best_pval, 'Add']], columns=['feature', 'p-val', 'step'])
            stepwise_outdf = stepwise_outdf.append(step_df)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
#         model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        model = sm.OLS(y, pd.DataFrame(X[included])).fit()
        # use all coefs except intercept
#         pvalues = model.pvalues.iloc[1:]
        pvalues = model.pvalues.drop('const')
        pval_df = pd.DataFrame(pvalues, columns=['pval']).rename_axis('feature').reset_index()
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pval_df.iloc[pvalues.argmax()]['feature']
            included.remove(worst_feature)
            step_df = pd.DataFrame([[worst_feature, best_pval, 'Drop']], columns=['feature', 'p-val', 'step'])
            stepwise_outdf = stepwise_outdf.append(step_df)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return(stepwise_outdf, included)


def run_logistic(X, y, method, resp_var):
    """
    Parameters
    ----------
        X: pandas dataframe of explanatory variables
        y: pandas series of dependent variable
        method: str, the logistic regression type. options are - logistic and l1
        resp_var: str, name of response variable
    Returns
    -------
        a pandas dataframe with model results
        statsmodels logistic/ l1 model object
    """
    # Fit Logistic/ L1 Model
    logit_model = sm.Logit(y, X)
    if method == 'logistic':
        result = logit_model.fit()
    elif method == 'l1':
        result = logit_model.fit_regularized(method=method)
    print(result.summary())
    
    # Development Data Performance
    y_pred = result.predict(X).rename('Probability')
    dev_scored_df = pd.concat([y, y_pred], axis=1)
    dev_metrics = standard_metrics(dev_scored_df, resp_var, 'Probability')
    
    print(f"Dev Gini: {np.round(dev_metrics['gini'], 4)}")
    print(f"Dev KS: {np.round(dev_metrics['ks'], 4)}")
    
    # Model Results
    model_results_df = result.summary2().tables[1].rename_axis('feature').reset_index()
    model_results_df.columns = ['feature', 'coeff', 'se', 'chi_sq', 'pval', 'conf_int_0.025', 'conf_int_0.975']
    
    return(model_results_df, result)