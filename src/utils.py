import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import joblib
import json
from datetime import datetime
import os

def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, model_name, save_path=None):
    """Plot and save ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_precision_recall_curve(y_true, y_pred_proba, model_name, save_path=None):
    """Plot and save Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = np.mean(precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'Avg Precision = {avg_precision:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(model, feature_names, top_n=15, save_path=None):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 8))
        plt.title("Feature Importances")
        bars = plt.bar(range(min(top_n, len(importances))), 
                      importances[indices[:top_n]])
        plt.xticks(range(min(top_n, len(importances))), 
                  [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Return feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'feature': [feature_names[i] for i in indices],
            'importance': importances[indices]
        })
        return feature_importance_df.head(top_n)
    else:
        print("Model doesn't have feature_importances_ attribute")
        return None

def generate_model_report(model, X_test, y_test, feature_names, model_name, save_dir='reports'):
    """Generate comprehensive model report with plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    print(f"📊 MODEL REPORT: {model_name}")
    print("=" * 50)
    
    # Classification report
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))
    
    # Generate plots
    if y_pred_proba is not None:
        plot_confusion_matrix(y_test, y_pred, model_name, 
                           save_path=os.path.join(save_dir, f'{model_name}_confusion_matrix.png'))
        plot_roc_curve(y_test, y_pred_proba, model_name,
                     save_path=os.path.join(save_dir, f'{model_name}_roc_curve.png'))
        plot_precision_recall_curve(y_test, y_pred_proba, model_name,
                                 save_path=os.path.join(save_dir, f'{model_name}_pr_curve.png'))
    
    # Feature importance
    feature_importance_df = plot_feature_importance(model, feature_names, 
                                                 save_path=os.path.join(save_dir, f'{model_name}_feature_importance.png'))
    
    return {
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'feature_importance': feature_importance_df
    }

def save_results_json(results, file_path):
    """Save results to JSON file"""
    # Convert numpy types to Python native types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_types(results)
    
    with open(file_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

def load_model_with_metadata(model_path):
    """Load model and its metadata"""
    model = joblib.load(model_path)
    
    # Try to load metadata if it exists
    metadata_path = model_path.replace('.joblib', '_metadata.json')
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return model, metadata

def save_model_with_metadata(model, metadata, model_path):
    """Save model with metadata"""
    joblib.dump(model, model_path)
    
    metadata_path = model_path.replace('.joblib', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def calculate_business_metrics(y_true, y_pred, y_pred_proba=None, default_cost=5, non_default_revenue=1):
    """
    Calculate business-oriented metrics
    default_cost: Cost of a false negative (missing a default)
    non_default_revenue: Revenue from a true positive (correctly identifying non-default)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'total_cost': fn * default_cost + fp * non_default_revenue,
        'cost_per_loan': (fn * default_cost + fp * non_default_revenue) / len(y_true),
        'default_miss_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
    }
    
    return metrics