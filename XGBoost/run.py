import os
from constants_config import DATA_FOLDER
from XGBoost.xgboost_multioutput_implement import XGBoostMultiOutput


def main():
    train_path = f"{DATA_FOLDER}/train_data.parquet"
    val_path = f"{DATA_FOLDER}/validation_data.parquet"
    test_path = f"{DATA_FOLDER}/test_data.parquet"

    # Ensure the directory exists
    os.makedirs(DATA_FOLDER, exist_ok=True)

    xgb_multi_output = XGBoostMultiOutput()
    xgb_multi_output.run(train_path, val_path, test_path)


if __name__ == "__main__":
    main()
