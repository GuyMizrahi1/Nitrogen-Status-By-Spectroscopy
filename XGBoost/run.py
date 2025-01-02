import argparse
from constants_config import DATA_FOLDER_PATH, FIGURE_FOLDER_PATH
from XGBoost.xgboost_multioutput_implement import XGBoostMultiOutput
from utils import plot_chosen_configurations_rmse, load_model, ensure_data_paths_exist, plot_learning_curves, \
    plot_feature_importances, plot_residuals


def main(train: bool = True):
    parser = argparse.ArgumentParser(description="Train or load XGBoost model")
    parser.add_argument('--train', action='store_true', help="Train a new model")
    args = parser.parse_args()

    # Ensure the directory exists and get data paths
    train_path, val_path, test_path, train_plsr_path, val_plsr_path, test_plsr_path = ensure_data_paths_exist(
        DATA_FOLDER_PATH)

    xgb_multi_output = XGBoostMultiOutput(model_name='xgboost_multi_output')
    xgb_multi_output_plsr = XGBoostMultiOutput(model_name='xgboost_multi_output_plsr')
    if train or args.train:
        xgb_multi_output.run(train_path, val_path, test_path)
        xgb_multi_output_plsr.run(train_plsr_path, val_plsr_path, test_plsr_path)
    else:
        xgb_multi_output = load_model(directory="models/xgboost_multi_output")
        xgb_multi_output_plsr = load_model(directory="models/xgboost_multi_output_plsr")

    # Create plots
    plot_chosen_configurations_rmse(xgb_multi_output, xgb_multi_output_plsr, FIGURE_FOLDER_PATH)
    plot_learning_curves(xgb_multi_output, xgb_multi_output_plsr, FIGURE_FOLDER_PATH)
    plot_feature_importances(xgb_multi_output, xgb_multi_output_plsr, FIGURE_FOLDER_PATH)
    plot_residuals(xgb_multi_output, xgb_multi_output_plsr, test_path, test_plsr_path, FIGURE_FOLDER_PATH)


if __name__ == "__main__":
    main()
