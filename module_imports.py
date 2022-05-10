# Import Required Modules
import pandas as pd, numpy as np, scipy as sp
import seaborn as sns, statsmodels.api as sm
from pandas_profiling import ProfileReport
import os, glob, warnings, time, copy, pickle, statistics, shutil, pyAesCrypt

from pdLogit.data_processing import *
from pdLogit.feature_selection import *
from pdLogit.classing import *
from pdLogit.model_build import *
from pdLogit.validation import *
from pdLogit.measure import *
from pdLogit.data_export import *

warnings.filterwarnings('ignore')

# Location Parameters
wpath = cfg.wpath
fpath = cfg.fpath
outpath = cfg.outpath

# Data Related Parameters
dev_fname = cfg.dev_fname
val_fname = cfg.val_fname

resp_var = cfg.resp_var
id_varlist = cfg.id_varlist
drop_varlist = cfg.drop_varlist

non_pred_varlist = id_varlist + [resp_var]
special_val_list = cfg.special_val_list

# Output File Names
mv_input = cfg.mv_input
feature_reduction_outfile = cfg.feature_reduction_outfile

# Encryption Credentials
encrypt_pwd = cfg.encrypt_pwd

# Cutoff Parameters
miss_cutff = cfg.miss_cutff
corr_cutoff = cfg.corr_cutoff
iv_cutoff = cfg.iv_cutoff
csi_cutoff = cfg.csi_cutoff