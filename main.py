from src.data_processing import DataProcessor
from src.model_training import ModelTrainingPipeline
from config.model_configs import DATA_CONFIG
import pandas as pd

def main():
    print("Credit Risk Prediction Pipeline")
    print("=" * 50)
    
    # Initialize data processor
    processor = DataProcessor(DATA_CONFIG)
    
    # Load and prepare data
    print("\nLoading and preprocessing data...")
    data = processor.prepare_data('data\credit_risk_dataset.csv')
    
    print(f"\nData Overview:")
    print(f"   Default rate: {data['data_info']['default_rate']:.2%}")
    print(f"   Training samples: {data['X_train'].shape[0]}")
    print(f"   Test samples: {data['X_test'].shape[0]}")
    print(f"   Features: {len(data['feature_names'])}")
    
    # Initialize and run training pipeline
    print("\nTraining models...")
    training_pipeline = ModelTrainingPipeline(processor)
    results = training_pipeline.run_pipeline(data)
    
    # Show feature importance
    print("\nFeature Importance (Best Model):")
    feature_importance = training_pipeline.get_feature_importance(data)
    if feature_importance is not None:
        print(feature_importance.head(10))
    
    print("\nPipeline completed successfully!")
    print("   Models saved in 'models/' directory")

if __name__ == "__main__":
    main()