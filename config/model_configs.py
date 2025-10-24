from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Model configurations
MODEL_CONFIGS = {
    'logistic_regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'grid_params': {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        },
        'random_params': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    },
    
    'random_forest': {
        'model': RandomForestClassifier(random_state=42),
        'grid_params': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        },
        'random_params': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    
    'xgboost': {
        'model': XGBClassifier(random_state=42, eval_metric='logloss'),
        'grid_params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1]
        },
        'random_params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
    },
    
    'lightgbm': {
        'model': LGBMClassifier(random_state=42),
        'grid_params': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10],
            'learning_rate': [0.01, 0.1]
        },
        'random_params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [31, 50, 100]
        }
    }
}

# Data configuration
DATA_CONFIG = {
    'target': 'loan_status',  # Binary classification: 1 = default, 0 = no default
    'numeric_features': [
        'person_age', 'person_income', 'person_emp_length', 
        'loan_amnt', 'loan_int_rate', 'loan_percent_income',
        'cb_person_cred_hist_length'
    ],
    'categorical_features': [
        'person_home_ownership', 'loan_intent', 'loan_grade', 
        'cb_person_default_on_file'
    ]
}