from PLSR_class import PLSRModel
import sys
sys.path.append('../baseline_for_training')
sys.path.append('../.')
from Dataset import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json


# creating Dataset Instance
train_file_name = 'train_data.parquet'
validation_file_name = 'validation_data.parquet'
test_file_name = 'test_data.parquet'
dataset = Dataset(train_file_name,validation_file_name,test_file_name)

# Correlation matrix - dependent variables
plt.figure(figsize=(8,5))
sns.heatmap(pd.concat([dataset.Y_train, dataset.Y_val,dataset.Y_test],axis=0).corr(),annot=True,fmt=".2f",cmap='Reds')
plt.title('Correlation matrix for X and Y')
plt.savefig('./Plots/Correlation_matrix.png')


# Pair plot - dependent variables
plt.figure(figsize=(8,5))
sns.pairplot(pd.concat([dataset.Y_train, dataset.Y_val,dataset.Y_test],axis=0))
plt.savefig('./Plots/Pair_plot.png')



# Preprocessing Data

X_scaler = StandardScaler()
Y_scaler = StandardScaler()

dataset.X_train[dataset.X_train.columns] = X_scaler.fit_transform(dataset.X_train.values)
dataset.Y_train[dataset.Y_train.columns] = Y_scaler.fit_transform(dataset.Y_train.values)

dataset.X_val[dataset.X_val.columns] = X_scaler.transform(dataset.X_val.values)
dataset.Y_val[dataset.Y_val.columns] = Y_scaler.transform(dataset.Y_val.values)

dataset.X_test[dataset.X_test.columns] = X_scaler.transform(dataset.X_test.values)
dataset.Y_test[dataset.Y_test.columns] = Y_scaler.transform(dataset.Y_test.values)


# Preparing Models

# Flags for the PLSR model
is_multi_output = True
PLSR_Tuning = True

# Define the parameter grid
param_grid = {'n_components': [i for i in range(1,51)]}

# Create the PLSR models
multi_PLSR = PLSRModel(dataset, param_grid, is_multi_output)
n_value_PLSR = PLSRModel(dataset, param_grid, not is_multi_output, "N_Value")
sc_value_PLSR = PLSRModel(dataset, param_grid, not is_multi_output, "SC_Value")
st_value_PLSR = PLSRModel(dataset, param_grid, not is_multi_output, "ST_Value")


# Hyperparameter Tuning

from Training_and_Tuning import hyperParameterTuning

# Perform hyperparameter tuning
multi_rmses = hyperParameterTuning(multi_PLSR,PLSR_Tuning = PLSR_Tuning)
n_value_rmses = hyperParameterTuning(n_value_PLSR,PLSR_Tuning = PLSR_Tuning)
sc_value_rmses = hyperParameterTuning(sc_value_PLSR,PLSR_Tuning = PLSR_Tuning)
st_value_rmses = hyperParameterTuning(st_value_PLSR,PLSR_Tuning = PLSR_Tuning)

path = './outputs/'
def create_json_file(rmses, filename):
    with open(filename, 'w') as f:
        json.dump(rmses, f)

create_json_file(multi_rmses, path + 'multi_rmses.json')
create_json_file(n_value_rmses, path + 'n_value_rmses.json')
create_json_file(sc_value_rmses, path + 'sc_value_rmses.json')
create_json_file(st_value_rmses, path + 'st_value_rmses.json')

# Get the best number of components for each model
best_n_components = sorted(multi_rmses['Avg_RMSE'], key = lambda x:x[1])[0][0]['n_components']
print('Best Multi Output PLSR Number of Components:', best_n_components)

best_n_value_n_components = sorted(n_value_rmses['N_Value'], key = lambda x:x[1])[0][0]['n_components']
print('Best N Value PLSR Number of Components:', best_n_value_n_components)


best_sc_value_n_components = sorted(sc_value_rmses['SC_Value'], key = lambda x:x[1])[0][0]['n_components']
print('Best SC Value PLSR Number of Components:', best_sc_value_n_components)

best_st_value_n_components = sorted(st_value_rmses['ST_Value'], key = lambda x:x[1])[0][0]['n_components']
print('Best ST Value PLSR Number of Components:', best_st_value_n_components)


# Training using CV10
from Training_and_Tuning import CV10

# Set the best number of components
multi_PLSR.model.estimator.set_params(n_components = best_n_components)
n_value_PLSR.model.set_params(n_components = best_n_value_n_components)
sc_value_PLSR.model.set_params(n_components = best_sc_value_n_components)
st_value_PLSR.model.set_params(n_components = best_st_value_n_components)

# Perform 10-fold cross validation
multi_rmse = CV10(multi_PLSR)
n_value_rmse = CV10(n_value_PLSR)
sc_value_rmse = CV10(sc_value_PLSR)
st_value_rmse = CV10(st_value_PLSR)


# save the results in json file

create_json_file(multi_rmse, path + 'multi_rmse_cv10.json')
create_json_file(n_value_rmse, path + 'n_value_rmse_cv10.json')
create_json_file(sc_value_rmse, path + 'sc_value_rmse_cv10.json')
create_json_file(st_value_rmse, path + 'st_value_rmse_cv10.json')


# Fit model on all of the data -  for saving only
multi_PLSR.model.estimator.fit(dataset.X_train, dataset.Y_train)
n_value_PLSR.model.fit(dataset.X_train, dataset.Y_train.iloc[:,0])
sc_value_PLSR.model.fit(dataset.X_train, dataset.Y_train.iloc[:,1])
st_value_PLSR.model.fit(dataset.X_train, dataset.Y_train.iloc[:,2])

# Eval on test data
rmses = multi_PLSR.evaluate()
# save results
create_json_file(rmses, path + 'multi_rmse_test.json')

# save model
import joblib
joblib.dump(multi_PLSR.model.estimator, './models/multi_plsr.pkl')




