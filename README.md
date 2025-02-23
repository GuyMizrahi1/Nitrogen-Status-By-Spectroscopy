# Nitrogen-Status-By-Spectroscopy
Development and evaluation of machine learning models for identifying nitrogen status in plants using spectroscopy. The project also investigates the relationships between mineral and metabolite profiles.

## Overview

This project involves training and evaluating machine learning models using Random Forest (RF) and XGBoost. The process includes data preparation, dataset creation, model training, and comparison of results. Follow the steps below to run the models and compare their performance.

## Prerequisites

Ensure you have the following installed:
- Python 3.9
- Required Python packages (listed in `requirements.txt`)

Install the required packages using:
```bash
pip install -r requirements.txt

```

## Step-by-Step Instructions

### 1. Data Preparation

First, prepare the data by running the `data_preparation.py` script. This script will load, validate, and clean the data.

```bash
python data_preparation.py
```

### 2. Create Component Datasets

Next, divide the datasets into training, validation, and test sets, and normalize them by running the `PLSR/create_components_datasets.py` script.

```bash
python PLSR/create_components_datasets.py
```

### 3. Train PLSR Model

Train the PLSR model and create the reduced datasets by running the `PLSR/PLSR.py` script.

```bash
python PLSR/PLSR.py
```

### 4. Train Random Forest Model

Train the Random Forest model using the `RF/rf_main.py` script. This script initializes the dataset, sets the training mode, and runs the RF model.

```bash
python RF/rf_main.py --train
```

### 5. Train XGBoost Model

Train the XGBoost model using the `XGBoost/run.py` script. This script initializes the dataset, sets the training mode, and runs the XGBoost model.

```bash
python XGBoost/run.py --train
```

### 6. Compare Models

Finally, compare the models by running the `comparison.py` script. This script will evaluate and compare the performance of the trained models.

```bash
python comparison.py
```

## Additional Information

### `data_preparation.py`

This script is responsible for loading, validating, and cleaning the data. Ensure that the data files are correctly placed in the specified directories before running this script.

### `PLSR/create_components_datasets.py`

This script divides the datasets into training, validation, and test sets, and normalizes them. It is essential to run this script before training the PLSR model.

### `PLSR/PLSR.py`

This script trains the PLSR model and creates the reduced datasets. The reduced datasets are used for training the RF and XGBoost models.

### `RF/rf_main.py`

This script initializes the dataset, sets the training mode, and runs the RF model. It uses the reduced datasets created by the PLSR model.

### `XGBoost/run.py`

This script initializes the dataset, sets the training mode, and runs the XGBoost model. It uses the reduced datasets created by the PLSR model.

### `comparison.py`

This script evaluates and compares the performance of the trained models. It generates plots and saves the test scores for further analysis.

## Conclusion

By following the steps outlined above, you can prepare the data, train the models, and compare their performance. Make sure to run each script in the specified order to ensure the process completes successfully.
