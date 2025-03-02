import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BertClassifierTrainer:
    def __init__(self, num_labels=2, model_dir="./models/bert"):
        self.num_labels = num_labels
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Initialized BERT classifier trainer with {num_labels} labels")
        
    def prepare_data(self, data_path, text_column, label_column, test_size=0.2):
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
        
        # Split data
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        
        # Get texts and labels
        train_texts = train_df[text_column].tolist()
        train_labels = train_df[label_column].tolist()
        
        test_texts = test_df[text_column].tolist()
        test_labels = test_df[label_column].tolist()
        
        logger.info(f"Data prepared: {len(train_texts)} training samples, {len(test_texts)} test samples")
        
        return train_texts, train_labels, test_texts, test_labels
    
    def load_model_and_tokenizer(self, model_name="bert-base-uncased"):
        """Load pre-trained BERT model and tokenizer"""
        logger.info(f"Loading pre-trained BERT model: {model_name}")
        
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=self.num_labels
        )
        
        model = model.to(self.device)
        return model, tokenizer
        
    def create_data_loaders(self, train_texts, train_labels, test_texts, test_labels, tokenizer, batch_size=16):
        """Create PyTorch DataLoaders for training and testing"""
        train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)
        test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, test_loader
        
    def train(self, model, train_loader, epochs=3, learning_rate=2e-5):
        """Train the BERT model"""
        logger.info("Training BERT model")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        return model
        
    def evaluate(self, model, test_loader):
        """Evaluate the BERT model"""
        logger.info("Evaluating BERT model")
        
        model.eval()
        
        true_labels = []
        predictions = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                _, preds = torch.max(logits, dim=1)
                
                true_labels.extend(labels.cpu().tolist())
                predictions.extend(preds.cpu().tolist())
        
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
            
        return metrics
        
    def save_model(self, model, tokenizer, metrics):
        """Save the trained model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{self.model_dir}/bert_classifier_{timestamp}"
        
        logger.info(f"Saving model to {model_path}")
        
        # Save model and tokenizer
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        # Save metrics
        metrics_path = f"{model_path}/metrics.pt"
        torch.save(metrics, metrics_path)
        
        return model_path
        
    def run_training_pipeline(self, data_path, text_column, label_column, batch_size=16, epochs=3):
        """Run the complete training pipeline"""
        train_texts, train_labels, test_texts, test_labels = self.prepare_data(
            data_path, text_column, label_column
        )
        
        model, tokenizer = self.load_model_and_tokenizer()
        
        train_loader, test_loader = self.create_data_loaders(
            train_texts, train_labels, test_texts, test_labels, tokenizer, batch_size
        )
        
        trained_model = self.train(model, train_loader, epochs)
        
        metrics = self.evaluate(trained_model, test_loader)
        
        model_path = self.save_model(trained_model, tokenizer, metrics)
        
        return model_path, metrics

if __name__ == "__main__":
    trainer = BertClassifierTrainer(num_labels=2)
    model_path, metrics = trainer.run_training_pipeline(
        data_path="./data/processed/text_data_processed",
        text_column="text",
        label_column="label",
        batch_size=16,
        epochs=3
    )
    
    print(f"Model saved to {model_path}")
    print(f"Model metrics: {metrics}")