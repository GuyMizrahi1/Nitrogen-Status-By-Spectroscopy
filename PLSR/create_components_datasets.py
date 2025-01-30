import sys
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from constants_config import DATA_FOLDER_PATH

# Load dataset
sys.path.append('../baseline_for_training')
sys.path.append('../.')
from Dataset import Dataset

# Creating Dataset Instance
train_file_name = 'train_data.parquet'
validation_file_name = 'validation_data.parquet'
test_file_name = 'test_data.parquet'
dataset = Dataset(train_file_name, validation_file_name, test_file_name)

# Standardizing the data
X_scaler = StandardScaler()
y_scaler = StandardScaler()

dataset.X_train[dataset.X_train.columns] = X_scaler.fit_transform(dataset.X_train.values)
dataset.X_val[dataset.X_val.columns] = X_scaler.transform(dataset.X_val.values)
dataset.X_test[dataset.X_test.columns] = X_scaler.transform(dataset.X_test.values)

y_scaled = y_scaler.fit_transform(dataset.Y_train)

# Fit PLSRegression directly
n_components = 15
pls = PLSRegression(n_components=n_components)
pls.fit(dataset.X_train, y_scaled)
model_name = f'pls_n_components_{n_components}.pkl'
joblib.dump(pls, os.path.join('./models', model_name))

# Transform data
X_train_plsr = pls.transform(dataset.X_train.reset_index(drop=True))
X_val_plsr = pls.transform(dataset.X_val.reset_index(drop=True))
X_test_plsr = pls.transform(dataset.X_test.reset_index(drop=True))

# Generate feature names manually
plsr_columns = [f'PLS_Component_{i+1}' for i in range(n_components)]

# Convert to DataFrame
def create_df_structure(X, columns):
    return {columns[i]: X[:, i] for i in range(X.shape[1])}

X_train_plsr_df = pd.DataFrame(create_df_structure(X_train_plsr, plsr_columns))
X_val_plsr_df = pd.DataFrame(create_df_structure(X_val_plsr, plsr_columns))
X_test_plsr_df = pd.DataFrame(create_df_structure(X_test_plsr, plsr_columns))

# Combine with Y data
train_data_plsr = pd.concat([dataset.Y_train.reset_index(drop=True), X_train_plsr_df], axis=1)
val_data_plsr = pd.concat([dataset.Y_val.reset_index(drop=True), X_val_plsr_df], axis=1)
test_data_plsr = pd.concat([dataset.Y_test.reset_index(drop=True), X_test_plsr_df], axis=1)

# Add ID column and set index
train_data_plsr['ID'] = dataset.ID_train.reset_index(drop=True)
val_data_plsr['ID'] = dataset.ID_val.reset_index(drop=True)
test_data_plsr['ID'] = dataset.ID_test.reset_index(drop=True)

train_data_plsr.set_index('ID', inplace=True)
val_data_plsr.set_index('ID', inplace=True)
test_data_plsr.set_index('ID', inplace=True)

# Save transformed dataset
train_data_plsr.reset_index().to_parquet(os.path.join(DATA_FOLDER_PATH, f'train_data_{n_components}_plsr.parquet'))
val_data_plsr.reset_index().to_parquet(os.path.join(DATA_FOLDER_PATH, f'validation_data_{n_components}_plsr.parquet'))
test_data_plsr.reset_index().to_parquet(os.path.join(DATA_FOLDER_PATH, f'test_data_{n_components}_plsr.parquet'))
