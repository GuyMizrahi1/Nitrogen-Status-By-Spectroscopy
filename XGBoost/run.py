import os
import argparse
from constants_config import DATA_FOLDER_PATH, FIGURE_FOLDER_PATH
from XGBoost.xgboost_multioutput_implement import XGBoostMultiOutput
from utils import save_best_model_and_params, load_best_model_and_params, plot_learning_curve, \
    plot_feature_importances, plot_residuals


def main():
    parser = argparse.ArgumentParser(description="Train or load XGBoost model")
    parser.add_argument('--train', action='store_true', help="Train a new model")
    args = parser.parse_args()

    train_path = f"{DATA_FOLDER_PATH}/train_data.parquet"
    val_path = f"{DATA_FOLDER_PATH}/validation_data.parquet"
    test_path = f"{DATA_FOLDER_PATH}/test_data.parquet"

    # Ensure the directory exists
    os.makedirs(DATA_FOLDER_PATH, exist_ok=True)

    xgb_multi_output = XGBoostMultiOutput()
    if args.train:
        xgb_multi_output.run(train_path, val_path, test_path)
        save_best_model_and_params(xgb_multi_output.model, xgb_multi_output.best_params, xgb_multi_output.train_rmses,
                                   xgb_multi_output.val_rmses, directory="XGBoost_final_model")
    else:
        (xgb_multi_output.model, xgb_multi_output.best_params, xgb_multi_output.train_rmses,
         xgb_multi_output.val_rmses) = load_best_model_and_params(directory="XGBoost_final_model")

    # Create plots
    plot_learning_curve(xgb_multi_output, "best_model", FIGURE_FOLDER_PATH)
    plot_feature_importances(xgb_multi_output, FIGURE_FOLDER_PATH)
    plot_residuals(xgb_multi_output, FIGURE_FOLDER_PATH)
    # todo: add plot that shows the trees in the model?


if __name__ == "__main__":
    main()
