from .hyperparameter_tuning import HyperparameterTuner
from config.model_configs import MODEL_CONFIGS
import joblib
import os
import pandas as pd

class ModelTrainingPipeline:
    def __init__(self, data_processor, results_dir='models'):
        self.data_processor = data_processor
        self.tuner = HyperparameterTuner()
        self.results_dir = results_dir
        self.trained_models = {}
        
        os.makedirs(results_dir, exist_ok=True)
    
    def run_pipeline(self, data, models_to_train=None):
        """Run complete training pipeline"""
        if models_to_train is None:
            models_to_train = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
        
        print("Starting model training pipeline...")
        print(f"Models to train: {models_to_train}")
        print(f"Target distribution - Default rate: {data['data_info']['default_rate']:.2%}")
        
        results = {}
        
        for model_name in models_to_train:
            print(f"\n{'='*60}")
            print(f"TRAINING: {model_name.upper()}")
            print(f"{'='*60}")
            
            model_results = self.train_single_model(model_name, data)
            results[model_name] = model_results
        
        # Compare and select best model
        self._select_best_model(results)
        
        return results
    
    def train_single_model(self, model_name, data):
        """Train a single model with multiple optimization techniques"""
        model_config = MODEL_CONFIGS[model_name]
        model = model_config['model']
        
        results = {}
        
        # 1. Grid Search
        print("1. Grid Search...")
        try:
            grid_result = self.tuner.grid_search(
                model, model_config['grid_params'], 
                data['X_train'], data['y_train']
            )
            results['grid_search'] = {
                'model': grid_result.best_estimator_,
                'best_params': grid_result.best_params_,
                'best_score': grid_result.best_score_
            }
            print(f"   Best score: {grid_result.best_score_:.4f}")
        except Exception as e:
            print(f"   Grid Search failed: {e}")
            results['grid_search'] = None
        
        # 2. Random Search
        print("2. Random Search...")
        try:
            random_result = self.tuner.random_search(
                model, model_config['random_params'],
                data['X_train'], data['y_train'],
                n_iter=30
            )
            results['random_search'] = {
                'model': random_result.best_estimator_,
                'best_params': random_result.best_params_,
                'best_score': random_result.best_score_
            }
            print(f"   Best score: {random_result.best_score_:.4f}")
        except Exception as e:
            print(f"   Random Search failed: {e}")
            results['random_search'] = None
        
        # 3. Optuna Optimization
        print("3. Optuna Optimization...")
        try:
            optuna_model, optuna_study = self.tuner.optuna_optimization(
                model, data['X_train'], data['y_train'], n_trials=50
            )
            results['optuna'] = {
                'model': optuna_model,
                'best_params': optuna_study.best_params,
                'best_score': optuna_study.best_value
            }
            print(f"   Best score: {optuna_study.best_value:.4f}")
        except Exception as e:
            print(f"   Optuna failed: {e}")
            results['optuna'] = None
        
        # Evaluate all successful models
        evaluation_results = {}
        for method, result in results.items():
            if result is not None:
                metrics = self.tuner.evaluate_model(
                    result['model'], data['X_test'], data['y_test'],
                    f"{model_name}_{method}"
                )
                evaluation_results[method] = {
                    'metrics': metrics,
                    'best_params': result['best_params'],
                    'best_score': result['best_score']
                }
                
                # Save model
                model_path = os.path.join(self.results_dir, f"{model_name}_{method}.joblib")
                self.tuner.save_model(result['model'], model_path)
        
        return evaluation_results
    
    def _select_best_model(self, all_results):
        """Select the best model across all algorithms"""
        print(f"\n{'='*80}")
        print("MODEL COMPARISON")
        print(f"{'='*80}")
        
        best_overall_score = -1
        best_overall_model = None
        best_model_name = None
        
        for model_name, methods in all_results.items():
            print(f"\n{model_name.upper()}:")
            print("-" * 40)
            
            for method, results in methods.items():
                if results:  # Check if results exist
                    f1_score = results['metrics']['f1']
                    print(f"  {method:15} | F1: {f1_score:.4f} | AUC: {results['metrics']['roc_auc']:.4f}")
                    
                    if f1_score > best_overall_score:
                        best_overall_score = f1_score
                        best_overall_model = results
                        best_model_name = f"{model_name}_{method}"
        
        print(f"\n{'='*80}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"Best F1 Score: {best_overall_score:.4f}")
        print(f"{'='*80}")
        
        self.best_model = best_overall_model
        self.best_model_name = best_model_name
    
    def get_feature_importance(self, data):
        """Get feature importance from tree-based models"""
        if hasattr(self.best_model['model'], 'feature_importances_'):
            importances = self.best_model['model'].feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': data['feature_names'],
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return feature_importance_df
        return None