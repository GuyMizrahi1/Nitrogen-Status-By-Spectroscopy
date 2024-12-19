import os
import shap
import json
import joblib
import itertools
import numpy as np
import pandas as pd
import xgboost as xgb
from pdpbox import pdp
from tqdm import tqdm
import matplotlib.pyplot as plt
from constants_config import ColumnName
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor


class XGBoostMultiOutput:
    def _init_(self, n_splits=4, save_dir="models/"):
        self.val_data = None
        self.train_data = None
        self.n_splits = n_splits
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.model = None
        self.best_params = None

    def load_data(self, train_path, val_path):
        self.train_data = pd.read_parquet(f"../{train_path}")
        self.val_data = pd.read_parquet(f"../{val_path}")

    def preprocess_data(self, dataset="train"):
        feature_columns = [
            col
            for col in self.train_data.columns
            if col
            not in [
                ColumnName.id,
                ColumnName.n_value,
                ColumnName.sc_value,
                ColumnName.st_value,
            ]
        ]
        if dataset == "train":
            X = self.train_data[feature_columns]
            y = self.train_data[
                [ColumnName.n_value, ColumnName.sc_value, ColumnName.st_value]
            ]
        elif dataset == "val":
            X = self.val_data[feature_columns]
            y = self.val_data[
                [ColumnName.n_value, ColumnName.sc_value, ColumnName.st_value]
            ]
        return X, y

    def get_param_grid(self):
        # Define a smaller hyperparameter grid with just 2 configurations
        param_grid = {
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5],
            "n_estimators": [50],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "gamma": [0],
            "reg_lambda": [1],
        }
        keys, values = zip(*param_grid.items())
        configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return configurations

    def train_and_evaluate(self, params, X_train, y_train, X_val, y_val):
        model = MultiOutputRegressor(xgb.XGBRegressor(**params))

        # Create separate eval_sets for each target
        eval_sets = [[(X_train, y_train.iloc[:, i]), (X_val, y_val.iloc[:, i])] for i in range(y_train.shape[1])]

        model.fit(X_train, y_train, estimator_params={'eval_set': eval_sets, 'eval_metric': 'rmse', 'verbose': False})

        y_pred_val = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred_val, multioutput='raw_values')).mean()
        return rmse, model

    def find_best_configuration(self, X_train, y_train, X_val, y_val):
        param_grid = self.get_param_grid()
        best_rmse = float("inf")
        best_params = None
        best_model = None

        for params in tqdm(param_grid, desc="Hyperparameter tuning"):
            rmse, model = self.train_and_evaluate(
                params, X_train, y_train, X_val, y_val
            )
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
                best_model = model

        self.best_params = best_params
        self.model = best_model  # Store the best model
        return best_rmse, best_params

    def cross_validate(self, X, y):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        rmses = []

        for train_index, val_index in tqdm(kf.split(X), desc="Cross-validation"):
            X_train_fold, X_val_fold = (
                X.iloc[train_index],
                X.iloc[val_index],
            )
            y_train_fold, y_val_fold = (
                y.iloc[train_index],
                y.iloc[val_index],
            )
            rmse, _ = self.train_and_evaluate(
                self.best_params,
                X_train_fold,
                y_train_fold,
                X_val_fold,
                y_val_fold,
            )
            rmses.append(rmse)
        self.plot_rmse_per_fold(rmses)  # Plot after cross-validation

        return np.mean(rmses)

    def plot_learning_curve(self, model):
        # Take the average learning curve across all outputs
        n_estimators = model.estimators_[0].n_estimators  # Assuming all estimators have the same n_estimators
        x_axis = range(1, n_estimators + 1)

        plt.figure(figsize=(10, 6))

        train_rmses = []
        val_rmses = []
        for i in range(len(model.estimators_)):
            train_rmses.append(model.estimators_[i].evals_result()["validation_0"]["rmse"])
            val_rmses.append(model.estimators_[i].evals_result()["validation_1"]["rmse"])

        plt.plot(x_axis, np.mean(train_rmses, axis=0), label="Average Train")
        plt.plot(x_axis, np.mean(val_rmses, axis=0), label="Average Validation")

        plt.title("Average Learning Curve")
        plt.xlabel("Boosting Round")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.save_dir, "average_learning_curve.png"))
        plt.show()

    def plot_feature_importances(self, model, X_train):
        """Plots mean feature importance."""
        all_importances = [
            est.feature_importances_ for est in model.estimators_
        ]
        mean_importance = np.mean(all_importances, axis=0)
        sorted_idx = np.argsort(mean_importance)[::-1]
        feature_names = X_train.columns[sorted_idx]
        mean_importance = mean_importance[sorted_idx]

        plt.figure(figsize=(12, 6))
        plt.bar(feature_names[:20], mean_importance[:20])
        plt.xticks(rotation=90)
        plt.title("Mean Feature Importance")
        plt.xlabel("Feature")
        plt.ylabel("Mean Importance Score")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "mean_feature_importance.png"))
        plt.show()

    def save_model(self, model, filename):
        joblib.dump(model, os.path.join(self.save_dir, filename))

    def save_best_params(self):
        with open(os.path.join(self.save_dir, "best_params.json"), "w") as f:
            json.dump(self.best_params, f)

    def plot_residuals(self, model, X, y):
        """Plots residuals for all target variables combined."""
        y_pred = model.predict(X)
        residuals = y - y_pred
        residuals = residuals.values.flatten()  # Flatten to a 1D array

        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred.flatten(), residuals, alpha=0.5)
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Combined Residuals Plot")
        plt.axhline(y=0, color="r", linestyle="--")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "combined_residuals_plot.png"))
        plt.show()

    def plot_predicted_vs_actual(self, model, X, y):
        """Plots predicted vs actual values for all target variables combined."""
        y_pred = model.predict(X)

        plt.figure(figsize=(8, 6))
        plt.scatter(y.values.flatten(), y_pred.flatten(), alpha=0.5)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Combined Predicted vs. Actual Plot")
        plt.plot(
            [y.min().min(), y.max().max()],
            [y.min().min(), y.max().max()],
            "k--",
            lw=2,
        )  # Add a diagonal line for reference
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, "combined_predicted_vs_actual.png")
        )
        plt.show()

    def plot_rmse_per_fold(self, rmses):
        """Plots RMSE scores for each cross-validation fold."""
        plt.figure(figsize=(8, 5))
        plt.bar(range(1, len(rmses) + 1), rmses)
        plt.xlabel("Fold")
        plt.ylabel("RMSE")
        plt.title("RMSE per Cross-Validation Fold")
        plt.savefig(os.path.join(self.save_dir, "rmse_per_fold.png"))
        plt.show()

    def plot_pdp(self, model, X, feature):
        """Plots a Partial Dependence Plot for a single feature using the first estimator."""
        pdp_feature = pdp.pdp_isolate(
            model=model.estimators_[0],
            dataset=X,
            model_features=X.columns,
            feature=feature,
        )
        pdp.pdp_plot(pdp_isolate_out=pdp_feature, feature_name=feature)
        plt.savefig(os.path.join(self.save_dir, f"pdp_{feature}.png"))
        plt.show()

    def plot_shap_summary(self, model, X):
        """Plots a SHAP summary plot using the first estimator."""
        explainer = shap.TreeExplainer(model.estimators_[0])
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X)
        plt.savefig(os.path.join(self.save_dir, "shap_summary_plot.png"))

    def run(self, train_path, val_path, test_path):
        self.load_data(train_path, val_path)
        X_train, y_train = self.preprocess_data(dataset="train")
        X_val, y_val = self.preprocess_data(dataset="val")

        print("Finding best configuration...")
        best_rmse, best_params = self.find_best_configuration(
            X_train, y_train, X_val, y_val
        )
        print(f"Best RMSE: {best_rmse}, Best Params: {best_params}")

        print("Cross-validating with best configuration...")
        mean_rmse = self.cross_validate(X_train, y_train)
        print(f"Mean RMSE from CV: {mean_rmse}")

        # Train the final model
        print("Training final model with best configuration...")
        final_model = MultiOutputRegressor(
            xgb.XGBRegressor(**best_params, eval_metric="rmse")
        )
        eval_set = [(X_train, y_train), (X_val, y_val)]
        final_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        # Evaluate on the validation set
        print("Evaluating final model on validation set...")
        y_pred_val = final_model.predict(X_val)
        val_rmse = np.sqrt(
            mean_squared_error(y_val, y_pred_val, multioutput="raw_values")
        ).mean()
        print(f"Validation RMSE: {val_rmse}")

        # Evaluate on the test set
        print("Evaluating final model on test set...")
        test_data = pd.read_parquet(f"../{test_path}")
        X_test = test_data[
            [
                col
                for col in test_data.columns
                if col
                not in [
                    ColumnName.id,
                    ColumnName.n_value,
                    ColumnName.sc_value,
                    ColumnName.st_value,
                ]
            ]
        ]
        y_test = test_data[
            [ColumnName.n_value, ColumnName.sc_value, ColumnName.st_value]
        ]
        y_pred_test = final_model.predict(X_test)
        test_rmse = np.sqrt(
            mean_squared_error(y_test, y_pred_test, multioutput="raw_values")
        ).mean()
        print(f"Test RMSE: {test_rmse}")

        # Additional Visualizations
        self.plot_learning_curve(final_model)
        self.plot_feature_importances(final_model, X_train)
        self.plot_residuals(final_model, X_val, y_val)
        self.plot_predicted_vs_actual(final_model, X_val, y_val)

        # Save the model
        self.save_model(final_model, "final_model.pkl")
        self.save_best_params()

        # Advanced Visualizations (Use the first estimator as an example)
        try:
            self.plot_pdp(final_model, X_val, feature="your_feature_name")
            self.plot_shap_summary(final_model, X_val)
        except NameError:
            print(
                "Skipping advanced visualizations (PDP and SHAP). Install pdpbox and shap for these plots."
            )
