# Credit Risk ML Pipeline

A comprehensive machine learning pipeline for credit risk assessment using multiple models and optimization techniques.

## Project Overview

This project implements an end-to-end machine learning pipeline for credit risk assessment. It includes data preprocessing, model training with hyperparameter optimization, and model evaluation using various machine learning algorithms.

## Project Structure

```
├── config/
│   └── model_configs.py    # Model configurations and hyperparameters
├── data/
│   └── credit_risk_dataset.csv    # Credit risk dataset
├── models/                 # Directory for saved models
├── src/
│   ├── data_processing.py         # Data preprocessing and feature engineering
│   ├── hyperparameter_tuning.py   # Hyperparameter optimization
│   └── model_training.py          # Model training pipeline
├── Dockerfile             # Docker configuration for containerization
├── docker-compose.yml    # Docker Compose configuration
├── main.py              # Main execution script
└── requirements.txt     # Python dependencies
```

## Features

### Data Processing
- Automated handling of missing values
- Feature encoding for categorical variables
- Data standardization
- Train-test split functionality

### Model Training
- Multiple model support:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
- Hyperparameter optimization techniques:
  - Grid Search
  - Random Search
- Model performance evaluation and selection

### Configuration
- Configurable model parameters
- Customizable preprocessing steps
- Environment variable support

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

The project includes pre-configured settings for multiple models:

- **Logistic Regression**: Basic linear model with L1/L2 regularization
- **Random Forest**: Ensemble learning with tree-based approach
- **XGBoost**: Gradient boosting implementation
- **LightGBM**: Light gradient boosting framework

Each model includes carefully tuned hyperparameter ranges for optimization.

## Data Requirements

The pipeline expects a credit risk dataset with:
- Numerical features
- Categorical features
- Binary target variable (credit risk assessment)

## Outputs

The pipeline generates:
- Trained models (saved in `models/` directory)
- Performance metrics
- Data preprocessing artifacts

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