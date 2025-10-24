import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_data(self, file_path):
        """Load and inspect data"""
        df = pd.read_csv(file_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Target distribution:\n{df[self.config['target']].value_counts()}")
        print(f"Missing values:\n{df.isnull().sum()}")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values appropriately for tabular data"""
        df = df.copy()
        
        # Fill numerical missing values with median
        for col in self.config['numeric_features']:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        for col in self.config['categorical_features']:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
                
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical features using Label Encoding"""
        df_encoded = df.copy()
        
        for col in self.config['categorical_features']:
            if col in df.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                
        return df_encoded
    
    def prepare_features(self, df):
        """Prepare features for modeling"""
        # Handle missing values
        df_clean = self.handle_missing_values(df)
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_clean)
        
        # Select features
        all_features = self.config['numeric_features'] + self.config['categorical_features']
        X = df_encoded[all_features]
        y = df_encoded[self.config['target']]
        
        self.feature_names = all_features
        
        return X, y
    
    def prepare_data(self, file_path, test_size=0.2, random_state=42):
        """Prepare complete dataset for training"""
        # Load data
        df = self.load_data(file_path)
        
        # Prepare features and target
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y
        )
        
        # Scale numerical features
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[self.config['numeric_features']] = self.scaler.fit_transform(
            X_train[self.config['numeric_features']]
        )
        X_test_scaled[self.config['numeric_features']] = self.scaler.transform(
            X_test[self.config['numeric_features']]
        )
        
        print(f"\nFinal dataset shapes:")
        print(f"X_train: {X_train_scaled.shape}, X_test: {X_test_scaled.shape}")
        print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'data_info': {
                'target_distribution': y.value_counts(),
                'default_rate': y.mean()
            }
        }