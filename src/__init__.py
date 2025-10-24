"""
Credit Risk ML Pipeline
A comprehensive machine learning pipeline for credit risk prediction
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Machine learning pipeline for credit risk prediction with multiple optimization techniques"

# Import key classes for easy access
from .data_processing import DataProcessor
from .model_training import ModelTrainingPipeline
from .hyperparameter_tuning import HyperparameterTuner

__all__ = [
    'DataProcessor',
    'ModelTrainingPipeline', 
    'HyperparameterTuner'
]