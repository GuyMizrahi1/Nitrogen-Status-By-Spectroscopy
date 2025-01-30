import sys
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import json

# Load dataset
sys.path.append('../baseline_for_training')
sys.path.append('../.')

from Dataset import Dataset

# Creating Dataset Instance
train_file_name = 'train_data.parquet'
validation_file_name = 'validation_data.parquet'
test_file_name = 'test_data.parquet'
dataset = Dataset(train_file_name, validation_file_name, test_file_name)
X_scaler = StandardScaler()
dataset.X_test.loc[:,dataset.X_test.columns] = X_scaler.fit_transform(dataset.X_test)
model_name = 'pls_n_components_15.pkl'
plsr_model = joblib.load(os.path.join('./models',model_name))
errors = mean_squared_error(dataset.Y_test, plsr_model.predict(dataset.X_test),
                            multioutput='raw_values')

errorss = {}
for idx,target in enumerate(dataset.Y_test.columns):
    rng = dataset.Y_test[target].max() - dataset.Y_test[target].min()
    errorss[target] = np.sqrt(errors[idx])

with open(os.path.join('./outputs',model_name.replace('.pkl','.json')),'w') as f:
    json.dump(errorss,f)




