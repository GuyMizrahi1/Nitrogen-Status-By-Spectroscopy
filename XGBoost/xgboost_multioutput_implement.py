import os
import joblib
import random
import itertools
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from constants_config import TARGET_VARIABLES, NON_FEATURE_COLUMNS, MEAN


class XGBoostMultiOutput:
    def __init__(self, model_name, n_splits=2, save_dir='models/', save_figure_dir='figures/'):
        self.model_name = model_name
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.n_splits = n_splits
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_figure_dir = save_figure_dir
        os.makedirs(self.save_figure_dir, exist_ok=True)
        self.model = None
        self.best_params = None
        self.train_rmses = {}
        self.val_rmses = {}
        self.evaluated_val_rmses = {}
        self.evaluated_test_rmses = {}
        self.targets_rmses_for_best_params = {}

    def get_feature_importances(self):
        """Aggregate feature importances from all models."""
        all_importances = []
        for model in self.model.estimators_:
            all_importances.append(model.feature_importances_)
        return np.mean(all_importances, axis=0)

    def get_feature_names(self):
        """Return feature names from the first estimator."""
        first_estimator = next(iter(self.model.estimators_))
        return first_estimator.feature_names_in_

    def load_data(self, train_path, val_path, test_path):
        self.train_data = pd.read_parquet(train_path)
        self.val_data = pd.read_parquet(val_path)
        self.test_data = pd.read_parquet(test_path)

    def preprocess_data(self, dataset="train"):
        feature_columns = [col for col in self.train_data.columns if col not in NON_FEATURE_COLUMNS]
        if dataset == "train":
            X = self.train_data[feature_columns]
            y = self.train_data[TARGET_VARIABLES]
        elif dataset == "val":
            X = self.val_data[feature_columns]
            y = self.val_data[TARGET_VARIABLES]
        elif dataset == "test":
            X = self.test_data[feature_columns]
            y = self.test_data[TARGET_VARIABLES]
        return X, y

    def get_param_grid(self):
        # Define the hyperparameter grid with a maximum of 100 configurations
        param_grid = {
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "n_estimators": [50, 100, 200],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "gamma": [0, 0.1, 0.2],
            "reg_lambda": [1, 2]
        }
        keys, values = zip(*param_grid.items())
        configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # Randomly sample 100 configurations
        configurations = random.sample(configurations, 100)
        return configurations

    def train_and_evaluate_by_rmse_per_configuration(self, params, X_train, y_train, X_val, y_val):
        # Multi-output regressor model
        model_params = {k: v for k, v in params.items() if k not in ['eval_set', 'eval_metric', 'verbose']}
        model = MultiOutputRegressor(xgb.XGBRegressor(**model_params, eval_metric='rmse'))
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)

        # Calculate RMSE for each target variable of the multi_model
        multi_rmses = np.sqrt(mean_squared_error(y_val, y_pred_val, multioutput='raw_values'))
        mean_rmses = np.mean(multi_rmses)

        return mean_rmses, multi_rmses.tolist()

    def find_best_configuration_based_rmse_score(self, X_train, y_train, X_val, y_val, model_name):
        # find the best hyperparameters for each target variable and the mean
        param_grid = self.get_param_grid()

        # Initialize the best RMSE and parameters storage
        minimal_rmse = float("inf")
        best_params = {}
        best_targets_rmses = {target: None for target in TARGET_VARIABLES}

        # Hyperparameter tuning loop
        for params in tqdm(param_grid, desc="Hyperparameter tuning"):
            rmse, multi_targets_rmses = self.train_and_evaluate_by_rmse_per_configuration(params, X_train, y_train,
                                                                                          X_val, y_val)
            # Update for multi-output model
            if rmse < minimal_rmse:
                minimal_rmse = rmse
                best_params = params
                best_targets_rmses = {target: multi_targets_rmses[i] for i, target in enumerate(TARGET_VARIABLES)}

        self.best_params = best_params
        best_targets_rmses[MEAN] = minimal_rmse
        self.targets_rmses_for_best_params = best_targets_rmses
        print(f"\nBest Configurations for {model_name} raised from Hyperparameter tuning:")
        print(best_params)

        return best_targets_rmses, best_params, minimal_rmse

    def k_fold_cross_validate_model(self, X_train, y_train, model_name, params):
        """Performs k-fold cross-validation and evaluation on the validation and test sets."""
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        # Initialize dictionaries to store RMSEs for each fold for each of n_estimators
        fold_train_rmses = {key: [] for key in TARGET_VARIABLES}
        fold_val_rmses = {key: [] for key in TARGET_VARIABLES}

        # Train multi-output model
        multi_model_params = params
        for train_index, val_index in tqdm(kf.split(X_train), desc="Cross-validation for Multi", disable=False):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            model = MultiOutputRegressor(xgb.XGBRegressor(**multi_model_params, eval_metric='rmse'))

            # Fit the multi-output model
            model.fit(X_train_fold, y_train_fold)

            # Train each estimator with eval_set
            for i, estimator in enumerate(model.estimators_):
                eval_set = [(X_train_fold, y_train_fold.iloc[:, i]), (X_val_fold, y_val_fold.iloc[:, i])]
                estimator.fit(X_train_fold, y_train_fold.iloc[:, i], eval_set=eval_set, verbose=False)

            # Store the RMSEs for each fold
            for i, target in enumerate(y_train.columns):
                fold_train_rmses[f'{target}'].append(model.estimators_[i].evals_result()["validation_0"]["rmse"])
                fold_val_rmses[f'{target}'].append(model.estimators_[i].evals_result()["validation_1"]["rmse"])

        # Save the multi-output model
        self.model = model

        # Store the RMSEs for each target
        self.train_rmses = {f'{key}': np.mean(fold_train_rmses[key], axis=0) for key in TARGET_VARIABLES}
        self.val_rmses = {f'{key}': np.mean(fold_val_rmses[key], axis=0) for key in TARGET_VARIABLES}

    def evaluate_model(self, X, y, model, dataset_type='validation'):
        """Evaluates the model on the given dataset."""
        y_pred = model.predict(X)
        individual_rmses = np.sqrt(mean_squared_error(y, y_pred, multioutput='raw_values'))
        print(f"\n{dataset_type.capitalize()} RMSEs for {self.model_name}:")
        for i, target in enumerate(TARGET_VARIABLES):
            if dataset_type == 'validation':
                self.evaluated_val_rmses[f'{target}'] = individual_rmses[i]
            else:
                self.evaluated_test_rmses[f'{target}'] = individual_rmses[i]
            print(f"{target}: {individual_rmses[i]:.4f}")

        mean_rmse = np.mean(individual_rmses)
        if dataset_type == 'validation':
            self.evaluated_val_rmses[MEAN] = mean_rmse
        else:
            self.evaluated_test_rmses[MEAN] = mean_rmse
        print(f"{MEAN}: {mean_rmse:.4f}")

    def save_model_object(self):
        print("Saving the model object after evaluation...")
        directory = os.path.join(self.save_dir, self.model_name)
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self, os.path.join(directory, 'model.pkl'))

    def run(self, train_path, val_path, test_path):
        print(f"Running {self.model_name} XGBoostMultiOutput")

        # Load the data
        self.load_data(train_path, val_path, test_path)
        X_train, y_train = self.preprocess_data(dataset="train")
        X_val, y_val = self.preprocess_data(dataset="val")
        X_test, y_test = self.preprocess_data(dataset="test")

        print("Finding best configurations...")
        best_rmses, best_params, best_multi_rmses = self.find_best_configuration_based_rmse_score(
            X_train, y_train, X_val, y_val, self.model_name)
        print("\nMinimal RMSEs based on the chosen configuration:")
        for key, value in best_rmses.items():
            print(f"{key}: {value}")

        self.k_fold_cross_validate_model(X_train, y_train, self.model_name, best_params)
        self.evaluate_model(X_val, y_val, self.model, dataset_type='validation')
        self.save_model_object()
        self.evaluate_model(X_test, y_test, self.model, dataset_type='test')
