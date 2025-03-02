from flask import Flask, request, jsonify
import joblib
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("api_logs.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load ML model (scikit-learn)
def load_sklearn_model(model_path):
    logger.info(f"Loading scikit-learn model from {model_path}")
    model_data = joblib.load(model_path)
    return model_data

# Load BERT model
def load_bert_model(model_path):
    logger.info(f"Loading BERT model from {model_path}")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer, device

# Initialize models
sklearn_model_path = os.environ.get('SKLEARN_MODEL_PATH', './models/random_forest_latest.pkl')
bert_model_path = os.environ.get('BERT_MODEL_PATH', './models/bert/bert_classifier_latest')

sklearn_model_data = load_sklearn_model(sklearn_model_path)
bert_model, bert_tokenizer, device = load_bert_model(bert_model_path)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'sklearn_model': sklearn_model_path,
        'bert_model': bert_model_path
    })

@app.route('/predict/tabular', methods=['POST'])
def predict_tabular():
    """Endpoint for tabular data predictions using scikit-learn model"""
    start_time = datetime.now()
    
    # Get request data
    content_type = request.headers.get('Content-Type', '')
    
    if 'application/json' in content_type:
        data = request.json
        logger.info(f"Received tabular prediction request: {data}")
    else:
        return jsonify({'error': 'Content-Type must be application/json'}), 400
    
    try:
        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
        
        # Check if all required features are present
        required_features = sklearn_model_data['feature_names']
        missing_features = [f for f in required_features if f not in df.columns]
        
        if missing_features:
            return jsonify({
                'error': f'Missing features: {missing_features}'
            }), 400
        
        # Make predictions
        model = sklearn_model_data['model']
        predictions = model.predict(df[required_features]).tolist()
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df[required_features]).tolist()
        else:
            probabilities = None
        
        # Calculate response time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = {
            'predictions': predictions,
            'probabilities': probabilities,
            'processing_time_seconds': processing_time
        }
        
        logger.info(f"Tabular prediction completed in {processing_time:.4f} seconds")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error during tabular prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/text', methods=['POST'])
def predict_text():
    """Endpoint for text classification using BERT model"""
    start_time = datetime.now()
    
    # Get request data
    content_type = request.headers.get('Content-Type', '')
    
    if 'application/json' in content_type:
        data = request.json
        logger.info(f"Received text prediction request")
    else:
        return jsonify({'error': 'Content-Type must be application/json'}), 400
    
    try:
        # Get text from request
        if isinstance(data, dict) and 'text' in data:
            text = data['text']
        else:
            return jsonify({'error': 'Request must contain "text" field'}), 400
        
        # Tokenize
        inputs = bert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(device)
        
        # Make prediction
        bert_model.eval()
        with torch.no_grad():
            outputs = bert_model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_class].item()
        
        # Calculate response time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = {
            'prediction': predicted_class,
            'confidence': confidence,
            'processing_time_seconds': processing_time
        }
        
        logger.info(f"Text prediction completed in {processing_time:.4f} seconds")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error during text prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 