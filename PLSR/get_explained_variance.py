import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm

from constants_config import TARGET_VARIABLES

# Load dataset
sys.path.append('../baseline_for_training')
sys.path.append('../.')
from Dataset import Dataset

# Creating Dataset Instance
train_file_name = 'train_data.parquet'
validation_file_name = 'validation_data.parquet'
test_file_name = 'test_data.parquet'
dataset = Dataset(train_file_name, validation_file_name, test_file_name)

# Standardizing the data
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = X_scaler.fit_transform(dataset.X_train)

y_scaled = y_scaler.fit_transform(dataset.Y_train)

# Number of components to test
max_components = 50
explained_variance_per_target = np.zeros((max_components, y_scaled.shape[1]))

# Compute explained variance for different number of components
for n in tqdm(range(max_components + 1)):
    pls = PLSRegression(n_components=n)
    multi_pls = MultiOutputRegressor(pls)
    multi_pls.fit(X_scaled, y_scaled)
    explained_variance_per_target[n - 1, :] = [
        estimator.score(X_scaled, y_scaled[:, i]) for i, estimator in enumerate(multi_pls.estimators_)
    ]

# Plot elbow curve for each target variable

plt.figure(figsize=(10, 6))
for i in range(y_scaled.shape[1]):
    plt.plot(range(1, max_components + 1), explained_variance_per_target[:, i], label=f'{TARGET_VARIABLES[i]}')

plt.xlabel('Number of PLS Components')
plt.ylabel('Explained Variance (R²)')
plt.title('Elbow Plot of Explained Variance in Target Variables')
plt.legend()
plt.grid(True)
plt.savefig('./explained_variance.png')

