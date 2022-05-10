# Location Parameters
wpath = '/Users/projjwal/python_codebase/logistic_framework'
fpath = wpath + '/data'
outpath = wpath + '/out'

# Data Related Parameters
dev_fname = 'train_data.csv'
val_fname = 'test_data.csv'

resp_var = 'diagnosis'
id_varlist = ['id']
drop_varlist = []

# List of Values for which separate bin needs to be created as part of coarse classing
special_val_list = [0, -999999]

# Output File Names
mv_input = 'missing_value_imputation_input.csv'
feature_reduction_outfile = 'feature_reduction_summary.xlsx'

# Encryption Credentials
encrypt_pwd = 'Hooghly@123'

# Cutoff Parameters
miss_cutff = 0.95
corr_cutoff = 0.9
iv_cutoff = 0.03
csi_cutoff = 0.25