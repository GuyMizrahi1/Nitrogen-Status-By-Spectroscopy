res = {"N_Value": 0.11864263575039709, "SC_Value": 2.3821409290429916, "ST_Value": 14.549582063990991, "Avg_RMSE": 5.6834552095947934}

from baseline_for_training.Dataset import  Dataset

train_file_name = 'train_data_15_plsr.parquet'
validation_file_name = 'validation_data_15_plsr.parquet'
test_file_name = 'test_data_15_plsr.parquet'

dataset = Dataset(train_file_name, validation_file_name, test_file_name)

for target in dataset.Y_test.columns:
    rng = dataset.Y_test[target].max() - dataset.Y_test[target].min()
    res[target] = res[target] / rng

print(res)