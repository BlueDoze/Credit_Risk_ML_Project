# Credit Risk ML Pipeline

A comprehensive machine learning pipeline for credit risk assessment using multiple models and optimization techniques.

## Project Overview

This project implements an end-to-end machine learning pipeline for credit risk assessment. It includes data preprocessing, model training with hyperparameter optimization, and comprehensive model evaluation with automated visualization generation.

## Project Structure

```
├── config/
│   └── model_configs.py    # Model configurations and hyperparameters
├── data/
│   └── credit_risk_dataset.csv    # Credit risk dataset
├── models/
│   ├── saved_models/      # Trained model files
│   └── plots/            # Generated visualizations and metrics
│       ├── logistic_regression/
│       ├── random_forest/
│       ├── xgboost/
│       └── lightgbm/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── data_processing.py       # Data preprocessing and feature engineering
│   ├── hyperparameter_tuning.py # Hyperparameter optimization
│   ├── model_training.py        # Model training pipeline
│   └── utils.py                # Visualization and utility functions
├── Dockerfile             # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── main.py              # Main execution script
└── requirements.txt     # Python dependencies
```

## Pipeline Workflow

### 1. Data Processing (`data_processing.py`)
- Loads and inspects the credit risk dataset
- Handles missing values:
  - Numerical features: Median imputation
  - Categorical features: Mode imputation
- Encodes categorical variables using Label Encoding
- Performs feature standardization
- Splits data into training and testing sets

### 2. Model Training (`model_training.py`)
The pipeline trains multiple models with different optimization techniques:

#### Models:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

#### Optimization Methods:
1. **Grid Search**
   - Systematic search through specified parameter grid
   - Exhaustive exploration of parameter combinations

2. **Random Search**
   - Random sampling of parameter combinations
   - Efficient exploration of parameter space

3. **Optuna Optimization**
   - Advanced hyperparameter optimization
   - Bayesian optimization approach

### 3. Model Evaluation and Visualization (`utils.py`)
For each trained model, the pipeline automatically generates:

#### Performance Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

#### Visualizations
- Confusion Matrix
- ROC Curves
- Precision-Recall Curves
- Feature Importance Plots (for tree-based models)

All visualizations are automatically saved in the `models/plots` directory, organized by model and optimization method.

### 4. Results Storage
- Models are saved in serialized format
- Performance metrics are stored in JSON
- Visualizations are saved as PNG files
- Organized directory structure for easy access

## Docker Support

The project is containerized using Docker for easy deployment and reproducibility:

- Python 3.9 base image
- Automatic dependency installation
- Volume mounting for data persistence
- Configured for ML workloads

### Running with Docker

```bash
# Build and run the container
docker-compose up --build

# Stop the container
docker-compose down
```

## Model Configurations

Each model includes carefully tuned hyperparameter ranges:

### Logistic Regression
- Regularization: L1/L2
- C values: [0.001, 0.01, 0.1, 1, 10, 100]
- Solvers: liblinear, saga

### Random Forest
- n_estimators: [50, 100, 200, 300]
- max_depth: [5, 10, 15, 20, None]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]

### XGBoost
- n_estimators: [100, 200, 300]
- max_depth: [3, 4, 5, 6, 7]
- learning_rate: [0.01, 0.05, 0.1, 0.2]
- subsample: [0.8, 0.9, 1.0]

### LightGBM
- Similar parameters to XGBoost
- Optimized for efficiency

## Data Requirements

The pipeline expects a credit risk dataset with:
- Numerical features
- Categorical features
- Binary target variable (credit risk assessment)

## Results and Outputs

The pipeline generates:
1. Trained models (saved in `models/saved_models/`)
2. Performance visualizations (`models/plots/`)
   - Confusion matrices
   - ROC curves
   - Precision-Recall curves
   - Feature importance plots
3. Performance metrics (JSON format)
4. Execution logs

## Development

To set up for development:

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place your dataset in the `data/` directory
4. Run the pipeline: `python main.py`

## Project Status

The project is production-ready with:
- Containerized environment
- Automated pipeline
- Multiple model support
- Hyperparameter optimization
- Error handling and logging