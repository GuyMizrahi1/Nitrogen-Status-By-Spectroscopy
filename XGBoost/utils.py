import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants_config import COLOR_PALETTE_FOR_TARGET_VARIABLES, TARGET_VARIABLES_WITH_MEAN, NON_FEATURE_COLUMNS, \
    TARGET_VARIABLES, COLOR_PALETTE_FOR_TWO_MODELS


def ensure_data_paths_exist(data_folder_path):
    """Ensure the directory exists and return paths for train, validation, and test data."""
    os.makedirs(data_folder_path, exist_ok=True)

    train_path = os.path.join(data_folder_path, 'train_data.parquet')
    val_path = os.path.join(data_folder_path, 'validation_data.parquet')
    test_path = os.path.join(data_folder_path, 'test_data.parquet')
    train_plsr_path = os.path.join(data_folder_path, 'train_data_15_plsr.parquet')
    val_plsr_path = os.path.join(data_folder_path, 'validation_data_15_plsr.parquet')
    test_plsr_path = os.path.join(data_folder_path, 'test_data_15_plsr.parquet')

    return train_path, val_path, test_path, train_plsr_path, val_plsr_path, test_plsr_path


def load_model(directory):
    return joblib.load(os.path.join(directory, "model.pkl"))


def plot_chosen_configurations_rmse(model1, model2, save_dir):
    """Bar plot of RMSE scores for the chosen configuration comparing two models."""
    labels = TARGET_VARIABLES_WITH_MEAN
    model1_rmse_values = [model1.targets_rmses_for_best_params[target] for target in labels]
    model2_rmse_values = [model2.targets_rmses_for_best_params[target] for target in labels]

    x = range(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plotting the bars for both models
    bars1 = ax.bar(x, model1_rmse_values, width, label=model1.model_name, color=COLOR_PALETTE_FOR_TWO_MODELS['model1'])
    bars2 = ax.bar([p + width for p in x], model2_rmse_values, width, label=model2.model_name,
                   color=COLOR_PALETTE_FOR_TWO_MODELS['model2'])

    # Add RMSE scores on top of each bar
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom', ha='center')
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom', ha='center')

    # Adding labels, title, and legend
    ax.set_xlabel('Target Variable')
    ax.set_ylabel('RMSE')
    ax.set_title('Comparison of RMSE Scores for Two Models')
    ax.set_xticks([p + width / 2 for p in x])
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "comparison_rmse.png"))
    plt.show()


def get_X_y(data_path, dataset='train'):
    data = pd.read_parquet(data_path)
    feature_columns = [col for col in data.columns if col not in NON_FEATURE_COLUMNS]
    if dataset in ('train', 'val', 'test'):
        X = data[feature_columns]
        y = data[TARGET_VARIABLES]
    return X, y


def save_test_scores(model1, model2, test_path1, test_path2, save_dir):
    X_test1, y_test1 = get_X_y(test_path1, dataset='test')
    X_test2, y_test2 = get_X_y(test_path2, dataset='test')

    y_pred1 = model1.model.predict(X_test1)
    y_pred2 = model2.model.predict(X_test2)

    scores = {
        model1.model_name: {},
        model2.model_name: {}
    }

    for i, var in enumerate(y_test1.columns):
        rmse1 = np.sqrt(np.mean((y_test1[var].values - y_pred1[:, i]) ** 2))
        rmse2 = np.sqrt(np.mean((y_test2[var].values - y_pred2[:, i]) ** 2))
        scores[model1.model_name][var] = rmse1
        scores[model2.model_name][var] = rmse2

    scores[model1.model_name]['mean_rmse'] = np.mean(list(scores[model1.model_name].values()))
    scores[model2.model_name]['mean_rmse'] = np.mean(list(scores[model2.model_name].values()))

    with open(os.path.join(save_dir, 'test_scores.json'), 'w') as f:
        json.dump(scores, f, indent=4)


def plot_learning_curves(model1, model2, save_dir):
    """Plot learning curves for two models side by side."""

    def plot_learning_curve(ax, model, config_name):
        n_estimators = model.best_params['n_estimators']
        x_axis = range(1, n_estimators + 1)

        for var in TARGET_VARIABLES:
            train_rmse_mean = model.train_rmses[f'{var}'][:n_estimators]
            val_rmse_mean = model.val_rmses[f'{var}'][:n_estimators]

            train_color, val_color = COLOR_PALETTE_FOR_TARGET_VARIABLES.get(var, ('#D3D3D3', '#DCDCDC'))

            ax.plot(x_axis, train_rmse_mean, label=f"Train RMSE - {var}", color=train_color)
            ax.plot(x_axis, val_rmse_mean, label=f"Validation RMSE - {var}", color=val_color)

        ax.set_title(f"Learning Curve for {config_name}")
        ax.set_xlabel("Number of Estimators")
        ax.set_ylabel("RMSE")
        ax.legend()
        ax.grid()

    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    config_name1 = model1.model_name
    config_name2 = model2.model_name

    plot_learning_curve(axs[0], model1, config_name1)
    plot_learning_curve(axs[1], model2, config_name2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"learning_curves_{config_name1}_vs_{config_name2}.png"))
    plt.show()


def plot_feature_importances(model1, model2, save_dir):
    """Plot feature importances for two models side by side."""

    def plot_importance(ax, model, title, num_features, color):
        importances = model.get_feature_importances()
        indices = np.argsort(importances)[::-1][:num_features]
        features = [model.get_feature_names()[i] for i in indices]

        ax.barh(range(len(indices)), importances[indices], align='center', color=color)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel('Feature Importance')

    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Calculate the minimum number of features between the two models
    num_features = min(len(model1.get_feature_importances()), len(model2.get_feature_importances()))

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    plot_importance(axs[0], model1, f'{model1.model_name} Feature Importances', num_features,
                    COLOR_PALETTE_FOR_TWO_MODELS['model1'])
    plot_importance(axs[1], model2, f'{model2.model_name}  Feature Importances', num_features,
                    COLOR_PALETTE_FOR_TWO_MODELS['model2'])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_importances_comparison.png"))
    plt.show()


def plot_residuals(model1, model2, directory1, directory2, save_dir):
    # Get the number of target variables
    num_targets1 = len(model1.model.estimators_)
    num_targets2 = len(model2.model.estimators_)

    X_test1, y_test1 = get_X_y(directory1, dataset='val')
    X_test2, y_test2 = get_X_y(directory2, dataset='val')
    y_pred1 = np.zeros((X_test1.shape[0], num_targets1))
    y_pred2 = np.zeros((X_test2.shape[0], num_targets2))

    for i, estimator in enumerate(model1.model.estimators_):
        y_pred1[:, i] = estimator.predict(X_test1)

    for i, estimator in enumerate(model2.model.estimators_):
        y_pred2[:, i] = estimator.predict(X_test2)

    # Calculate residuals for each model
    residuals1 = y_test1.values - y_pred1
    residuals2 = y_test2.values - y_pred2

    # Plot residuals for each target variable
    for i, var in enumerate(y_test1.columns):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred1[:, i], residuals1[:, i], alpha=0.5, label=f'{model1.model_name} Residuals',
                    color=COLOR_PALETTE_FOR_TWO_MODELS['model1'])
        plt.scatter(y_pred2[:, i], residuals2[:, i], alpha=0.5, label=f'{model2.model_name} Residuals',
                    color=COLOR_PALETTE_FOR_TWO_MODELS['model2'])
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals Values")
        plt.title(f"Residuals Plot for {var}")
        plt.axhline(y=0, color="r", linestyle="--")
        plt.legend()
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"residuals_plot_{var}.png"))
        plt.show()
