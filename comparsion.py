import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from constants_config import COLOR_PALETTE_FOR_PLSR, TARGET_VARIABLES

plsr_test_path = './PLSR/outputs/test/multi_rmse_test.json'
xgboost_test_path = './XGBoost/figures/test_scores.json'
RF_test_path = './RF/models/test_results.json'

test_data = pd.read_parquet('./datasets/test_data_plsr.parquet')
targets = test_data[TARGET_VARIABLES]
ranges = {}

for target in TARGET_VARIABLES:
    ranges[target] = targets[target].max() - targets[target].min()

def load_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data

xgboost_data = load_json(xgboost_test_path)['xgboost_multi_output_plsr']
xgb_values = xgboost_data.values()
plsr_data = load_json(plsr_test_path)
rf_data = load_json(RF_test_path)
keys = rf_data.keys()

plsr_test = {}
xgboost_test = {}
rf_test = {}

for i,key in enumerate(keys):
    if i == 3:
        break
    plsr_test[key] = plsr_data[i]/ranges[key]
    xgboost_test[key] = xgboost_data[key]/ranges[key]
    rf_test[key] = rf_data[key]/ranges[key]

plsr_test['Avg_RMSE'] = float(np.mean(list(plsr_test.values())))
xgboost_test['Avg_RMSE'] = float(np.mean(list(xgboost_test.values())))
rf_test['Avg_RMSE'] = float(np.mean(list(rf_test.values())))


data = pd.DataFrame({
    "PLSR": plsr_test,
    "XGBoost": xgboost_test,
    "RF": rf_test
}).T


# Slightly darker shades of the color palette for better visibility
custom_colors_darker = ['#6495ED', '#32CD32', '#FF6347', '#FFD700']  # Cornflower blue, lime green, tomato, gold

# Plot with the legend repositioned to the bottom between XGBoost and RF
ax = data.plot(kind="bar", figsize=(10, 6), alpha=0.9, color=custom_colors_darker, edgecolor="black")
plt.title("Model Performance Comparison", fontsize=16, fontweight='bold', color="#333333")
plt.ylabel("RMSE", fontsize=14, color="#333333")
plt.xlabel("Models", fontsize=14, color="#333333")
plt.xticks(rotation=0, fontsize=12, color="#333333")
plt.yticks(fontsize=12, color="#333333")
plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

# Adding values on top of the bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', label_type='edge', fontsize=10, padding=3)

# Adjust bin spaces
plt.tight_layout()
# Positioning the legend at the bottom between XGBoost and RF
plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(0.21,0.8),ncol=1)

plt.savefig("comparsion.png",dpi=600)





