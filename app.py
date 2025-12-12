"""
Hate Speech Detection Web Application
----------------------------------
Flask application providing both web interface and REST API for hate speech detection.
"""

from flask import Flask, request, jsonify, render_template
import pickle
import os
from src.main import classify_text_with_models
from src.dl.infer import compare_models

app = Flask(__name__)

def load_models():
    """
    Load trained ML models from disk.
    Returns (vectorizer, nb_model, lr_model) or (None, None, None) if loading fails.
    """
    try:
        with open('models/ml/naive_bayes.pkl', 'rb') as f:
            nb_model = pickle.load(f)
        with open('models/ml/logistic_regression.pkl', 'rb') as f:
            lr_model = pickle.load(f)
        with open('models/ml/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return vectorizer, nb_model, lr_model
    except FileNotFoundError:
        print("Error: ML model files not found. Please run 'python src/main.py' first to train the models.")
        return None, None, None

# Load models at startup
vectorizer, nb_model, lr_model = load_models()

@app.route('/')
def home():
    """Render the home page with the classification form."""
    return render_template('index.html', input_text='')

@app.route('/classify', methods=['POST'])
def classify():
    """
    Classify text endpoint - handles both form submissions and API calls.
    
    Form submission: Returns rendered template with results
    API call: Returns JSON response
    
    Expected JSON format:
    {
        "text": "Text to classify"
    }
    
    Returns:
    {
        "text": "Original text",
        "classification": "hateful|offensive|neutral",
        "confidence": "85.0%",
        "explanation": "Classification explanation",
        "metadata": {
            "is_rule_based": bool,
            "flagged_context": str|null,
            "semantic_alert": bool
        }
    }
    """
    # Check if models are loaded
    if None in (vectorizer, nb_model, lr_model):
        return jsonify({
            'error': 'Models not loaded. Please train models first using python src/main.py'
        }), 500

    # Get text from request
    if request.is_json:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided. Please send a JSON with a text field.'
            }), 400
        text = data['text']
    else:
        text = request.form.get('text')
        if not text:
            return jsonify({
                'error': 'No text provided.'
            }), 400

    if not text.strip():
        return jsonify({
            'error': 'Empty text provided.'
        }), 400

    # Classify the text
    try:
        result, confidence, explanation, metadata = classify_text_with_models(
            text,
            vectorizer,
            nb_model,
            lr_model
        )

        response_data = {
            'text': text,
            'classification': result,
            'confidence': f'{confidence:.1%}',
            'explanation': explanation,
            'metadata': metadata
        }

        # Deep learning comparison
        comparison = compare_models(text, vectorizer, nb_model, lr_model)
        baseline_ui = {
            "label": response_data["classification"].title(),
            "confidence": float(confidence) * 100.0,
            "explanation": explanation,
            "metadata": metadata,
            "model_name": "Rules + Classical ML",
            "text": text,
        }
        
        # Get both DL models separately
        lstm_result = comparison.get("deep_lstm", {"label": "neutral", "confidence": 0.0, "model_name": "BiLSTM-Attn (unavailable)"})
        transformer_result = comparison.get("deep_transformer", {"label": "neutral", "confidence": 0.0, "model_name": "DistilBERT (unavailable)"})
        
        lstm_ui = {
            "label": str(lstm_result.get("label", "neutral")).title(),
            "confidence": float(lstm_result.get("confidence", 0.0)),
            "model_name": lstm_result.get("model_name", "BiLSTM-Attn"),
            "text": text,
        }
        
        transformer_ui = {
            "label": str(transformer_result.get("label", "neutral")).title(),
            "confidence": float(transformer_result.get("confidence", 0.0)),
            "model_name": transformer_result.get("model_name", "DistilBERT"),
            "text": text,
        }

        # Return appropriate response based on request type
        if not request.is_json:
            return render_template(
                'index.html',
                result=response_data,
                baseline_result=baseline_ui,
                lstm_result=lstm_ui,
                transformer_result=transformer_ui,
                input_text=text,
            )
        return jsonify(response_data)

    except Exception as e:
        return jsonify({
            'error': f'Classification failed: {str(e)}'
        }), 500

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    # Create required directories
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Run the application
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False) 