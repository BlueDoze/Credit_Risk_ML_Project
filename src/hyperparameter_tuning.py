import optuna
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import joblib
import time
import pandas as pd

class HyperparameterTuner:
    def __init__(self, cv=5, scoring='f1', n_jobs=-1, random_state=42):
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.results = {}
    
    def grid_search(self, model, param_grid, X_train, y_train):
        """Perform Grid Search CV"""
        print(f"Grid Search for {model.__class__.__name__}...")
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        return grid_search
    
    def random_search(self, model, param_distributions, X_train, y_train, n_iter=50):
        """Perform Random Search CV"""
        print(f"Random Search for {model.__class__.__name__}...")
        
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=0
        )
        
        random_search.fit(X_train, y_train)
        
        return random_search
    
    def optuna_optimization(self, model, X_train, y_train, n_trials=50):
        """Perform optimization using Optuna"""
        print(f"Optuna optimization for {model.__class__.__name__}...")
        
        def objective(trial):
            # Different parameter spaces for different models
            if hasattr(model, 'C'):  # Logistic Regression
                params = {
                    'C': trial.suggest_float('C', 0.001, 100, log=True),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                }
            elif hasattr(model, 'n_estimators'):  # Tree-based models
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                }
                if hasattr(model, 'subsample'):
                    params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
            
            model.set_params(**params)
            score = cross_val_score(model, X_train, y_train, cv=self.cv, scoring=self.scoring).mean()
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Train best model
        best_model = model.set_params(**study.best_params)
        best_model.fit(X_train, y_train)
        
        return best_model, study
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Comprehensive model evaluation"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        }
        
        self.results[model_name] = {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred
        }
        
        return metrics
    
    def save_model(self, model, file_path):
        """Save trained model"""
        joblib.dump(model, file_path)
    
    def compare_results(self):
        """Compare results from all models"""
        comparison = {}
        for model_name, result in self.results.items():
            comparison[model_name] = result['metrics']
        
        return pd.DataFrame(comparison).T