import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SklearnModelTrainer:
    def __init__(self, model_type="xgboost", model_dir="./models"):
        self.model_type = model_type
        self.model_dir = model_dir
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"Initialized {model_type} model trainer")
        
    def prepare_data(self, data_path, target_column, feature_columns=None, test_size=0.2):
        """Load and prepare data for model training"""
        logger.info(f"Loading data from {data_path}")
        
        # Load data
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
            
        logger.info(f"Data loaded with shape: {df.shape}")
        
        # Select features
        if feature_columns:
            X = df[feature_columns]
        else:
            X = df.drop(columns=[target_column])
            
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        logger.info(f"Data prepared: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, X.columns.tolist()
    
    def build_model(self):
        """Build the model pipeline"""
        logger.info(f"Building {self.model_type} model")
        
        if self.model_type == "xgboost":
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', xgb.XGBClassifier(
                    objective='binary:logistic',
                    random_state=42,
                    n_jobs=-1
                ))
            ])
            
            param_grid = {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 5, 7],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__subsample': [0.8, 1.0],
                'classifier__colsample_bytree': [0.8, 1.0]
            }
        
        elif self.model_type == "lightgbm":
            # Import is inside condition to avoid requiring installation if not used
            import lightgbm as lgb
            
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', lgb.LGBMClassifier(
                    objective='binary',
                    random_state=42,
                    n_jobs=-1
                ))
            ])
            
            param_grid = {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 5, 7],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__subsample': [0.8, 1.0],
                'classifier__colsample_bytree': [0.8, 1.0],
                'classifier__num_leaves': [31, 50, 70]
            }
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        return model, param_grid
        
    def train(self, X_train, y_train, param_grid, cv=5):
        """Train the model with grid search"""
        logger.info("Training model with grid search")
        
        model, param_grid = self.build_model()
        
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
        
    def evaluate(self, model, X_test, y_test):
        """Evaluate the model"""
        logger.info("Evaluating model")
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
            
        return metrics
        
    def save_model(self, model, feature_names, metrics):
        """Save the trained model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{self.model_dir}/{self.model_type}_{timestamp}.pkl"
        
        # Save model with metadata
        model_data = {
            'model': model,
            'feature_names': feature_names,
            'metrics': metrics,
            'timestamp': timestamp,
            'model_type': self.model_type
        }
        
        logger.info(f"Saving model to {model_filename}")
        joblib.dump(model_data, model_filename)
        
        return model_filename
        
    def run_training_pipeline(self, data_path, target_column, feature_columns=None):
        """Run the complete training pipeline"""
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data(
            data_path, target_column, feature_columns
        )
        
        model, param_grid = self.build_model()
        
        trained_model = self.train(X_train, y_train, param_grid)
        
        metrics = self.evaluate(trained_model, X_test, y_test)
        
        model_path = self.save_model(trained_model, feature_names, metrics)
        
        return model_path, metrics

if __name__ == "__main__":
    trainer = SklearnModelTrainer(model_type="xgboost")
    model_path, metrics = trainer.run_training_pipeline(
        data_path="./data/processed/customer_data_processed",
        target_column="churn",
        feature_columns=None  # Use all features except target
    )
    
    print(f"Model saved to {model_path}")
    print(f"Model metrics: {metrics}")