import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib
import json
import os
from constants_config import ColumnName


class XGBoostRegressorCV:
    def __init__(self, n_splits=10, save_dir="models/"):
        self.models = {}  # Store trained models for each target variable
        self.cv_scores = {}  # Store cross-validation scores for each target variable
        self.best_params = {}  # Store best hyperparameters for each target variable
        self.n_splits = n_splits  # Number of CV folds
        self.save_dir = save_dir  # Directory to save models and results
        os.makedirs(self.save_dir, exist_ok=True)  # Create save directory if not exists

    def load_data(self, train_path, val_path):
        """Load training and validation datasets from parquet files."""
        self.train_data = pd.read_parquet(train_path)
        self.val_data = pd.read_parquet(val_path)

    def preprocess_data(self, target_column, dataset="train"):
        """Split the data into features (X) and target (y) for training or validation."""
        feature_columns = [col for col in self.train_data.columns if
                           col not in [ColumnName.id, ColumnName.n_value, ColumnName.sc_value, ColumnName.st_value]]
        if dataset == "train":
            X = self.train_data[feature_columns]
            y = self.train_data[target_column]
        elif dataset == "val":
            X = self.val_data[feature_columns]
            y = self.val_data[target_column]
        return X, y

    def hyperparameter_tuning(self, X, y, target_column):
        """Perform hyperparameter tuning using GridSearchCV."""
        param_grid = {
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [4, 6, 8],
            "n_estimators": [100, 200, 300],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }

        model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        print(f"Starting hyperparameter tuning for {target_column}...")
        tuning_progress = tqdm(total=len(param_grid["learning_rate"]) *
                                     len(param_grid["max_depth"]) *
                                     len(param_grid["n_estimators"]) *
                                     self.n_splits,
                               desc=f"Tuning {target_column}")

        def update_progress(*args):
            tuning_progress.update()

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=kfold,
            verbose=0,  # Suppress verbosity to avoid cluttering
            n_jobs=-1
        )
        grid_search.fit(X, y)
        tuning_progress.close()

        best_params = grid_search.best_params_
        best_score = np.sqrt(-grid_search.best_score_)

        print(f"Best Parameters for {target_column}: {best_params}")
        print(f"Best CV RMSE for {target_column}: {best_score:.4f}")
        return best_params

    def train_model(self, target_column, params):
        """Train an XGBoost model for a specific target variable."""
        X_train, y_train = self.preprocess_data(target_column, dataset="train")
        X_val, y_val = self.preprocess_data(target_column, dataset="val")

        print(f"\nTraining model for {target_column}...")
        training_progress = tqdm(desc=f"Training {target_column}", total=1, position=0)

        model = xgb.XGBRegressor(**params)
        eval_set = [(X_train, y_train), (X_val, y_val)]
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric="rmse",
            verbose=True
        )
        training_progress.update(1)
        training_progress.close()

        # Predict on validation set
        y_pred_val = model.predict(X_val)

        # Evaluate performance on validation set
        rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
        r2_val = r2_score(y_val, y_pred_val)

        print(f"{target_column} - Validation RMSE: {rmse_val:.4f}, R²: {r2_val:.4f}")
        self.models[target_column] = model  # Save the trained model

        # Save model and metrics
        model_path = os.path.join(self.save_dir, f"{target_column}_model.pkl")
        joblib.dump(model, model_path)
        print(f"Model for {target_column} saved at: {model_path}")

        return rmse_val, r2_val

    def perform_cv_train_with_tuning(self):
        """Perform hyperparameter tuning, cross-validation, and training for each target variable."""
        results = {}
        for target_column in tqdm([ColumnName.n_value, ColumnName.sc_value, ColumnName.st_value],
                                  desc="Processing target variables"):
            # Preprocess data
            X_train, y_train = self.preprocess_data(target_column, dataset="train")

            # Hyperparameter tuning
            best_params = self.hyperparameter_tuning(X_train, y_train, target_column)
            self.best_params[target_column] = best_params

            # Train the model with the best parameters and evaluate on validation set
            rmse_val, r2_val = self.train_model(target_column, best_params)
            results[target_column] = {"RMSE": rmse_val, "R²": r2_val}

        # Save best parameters
        params_path = os.path.join(self.save_dir, "best_params.json")
        with open(params_path, "w") as f:
            json.dump(self.best_params, f)
        print(f"Best parameters saved at: {params_path}")

        return results

    def predict(self, X):
        """Predict using trained models for all target variables."""
        predictions = {}
        for target_column, model in self.models.items():
            predictions[target_column] = model.predict(X)
        return pd.DataFrame(predictions)

    def plot_learning_curve(self, model, target_column):
        """Plot learning curve for training and validation RMSE."""
        results = model.evals_result()
        epochs = len(results["validation_0"]["rmse"])
        x_axis = range(0, epochs)

        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, results["validation_0"]["rmse"], label="Train")
        plt.plot(x_axis, results["validation_1"]["rmse"], label="Validation")
        plt.title(f"Learning Curve for {target_column}")
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_feature_importance(self, target_column):
        """Plot feature importance for a trained model."""
        if target_column not in self.models:
            print(f"No trained model found for {target_column}")
            return

        model = self.models[target_column]
        importance = model.get_booster().get_score(importance_type="weight")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        keys, values = zip(*sorted_importance)

        plt.figure(figsize=(12, 6))
        plt.bar(keys[:20], values[:20])  # Top 20 features
        plt.xticks(rotation=90)
        plt.title(f"Feature Importance for {target_column}")
        plt.xlabel("Feature")
        plt.ylabel("Score")
        plt.show()
