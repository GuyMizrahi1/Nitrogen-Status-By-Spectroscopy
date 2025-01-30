# load multi plsr model

import joblib
import sys
import os
sys.path.append('../baseline_for_training')
sys.path.append('../.')
from Dataset import Dataset
import pandas as pd

multi_PLSR = joblib.load('./models/multi_plsr.pkl')

# creating Dataset Instance
train_file_name = 'train_data.parquet'
validation_file_name = 'validation_data.parquet'
test_file_name = 'test_data.parquet'
dataset = Dataset(train_file_name,validation_file_name,test_file_name)


# Preporcess again

from sklearn.preprocessing import StandardScaler

X_scaler = StandardScaler()


dataset.X_train[dataset.X_train.columns] = X_scaler.fit_transform(dataset.X_train.values)
dataset.X_val[dataset.X_val.columns] = X_scaler.transform(dataset.X_val.values)
dataset.X_test[dataset.X_test.columns] = X_scaler.transform(dataset.X_test.values)

X_train_plsr = multi_PLSR._x_scores
X_val_plsr = multi_PLSR.transform(dataset.X_val.reset_index(drop=True))
X_test_plsr = multi_PLSR.transform(dataset.X_test.reset_index(drop=True))

plsr_columns = list(multi_PLSR.get_feature_names_out())

def create_df_structure(X,columns):
    return {columns[i]:X[:,i] for i in range(X.shape[1])}

X_train_plsr_df = pd.DataFrame(create_df_structure(X_train_plsr,plsr_columns))
X_val_plsr_df = pd.DataFrame(create_df_structure(X_val_plsr,plsr_columns))
X_test_plsr_df = pd.DataFrame(create_df_structure(X_test_plsr,plsr_columns))

train_data_plsr = pd.concat([dataset.Y_train.reset_index(drop=True), X_train_plsr_df],axis=1)
val_data_plsr = pd.concat([dataset.Y_val.reset_index(drop=True), X_val_plsr_df],axis=1)
test_data_plsr = pd.concat([dataset.Y_test.reset_index(drop=True), X_test_plsr_df],axis=1)

# add ID

train_data_plsr['ID'] = dataset.ID_train.reset_index(drop=True)
val_data_plsr['ID'] = dataset.ID_val.reset_index(drop=True)
test_data_plsr['ID'] = dataset.ID_test.reset_index(drop=True)

train_data_plsr.set_index('ID',inplace=True)
val_data_plsr.set_index('ID',inplace=True)
test_data_plsr.set_index('ID',inplace=True)

from constants_config import DATA_FOLDER_PATH
train_data_plsr.reset_index().to_parquet(os.path.join(DATA_FOLDER_PATH,'train_data_plsr.parquet'))
val_data_plsr.reset_index().to_parquet(os.path.join(DATA_FOLDER_PATH,'validation_data_plsr.parquet'))
test_data_plsr.reset_index().to_parquet(os.path.join(DATA_FOLDER_PATH,'test_data_plsr.parquet'))

