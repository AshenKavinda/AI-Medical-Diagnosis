from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from symptom_processor import text_to_vector
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration from .env
FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
DEFAULT_THRESHOLD = int(os.getenv('DEFAULT_THRESHOLD', 90))
MAX_PREDICTIONS = int(os.getenv('MAX_PREDICTIONS', 5))

# Load models at startup
print("Loading models...")
xgb_model = joblib.load("illness_predictor_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
print("Models loaded successfully!")
print(f"Configuration: Threshold={DEFAULT_THRESHOLD}%, Max Predictions={MAX_PREDICTIONS}")

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Process symptom text and return disease predictions"""
    try:
        data = request.get_json()
        user_text = data.get('symptoms', '')
        threshold = data.get('threshold', DEFAULT_THRESHOLD)
        
        if not user_text.strip():
            return jsonify({'error': 'Please enter symptom description'}), 400
        
        # Convert text to symptom vector
        symptom_vector, detected_symptoms = text_to_vector(user_text, threshold=threshold)
        num_symptoms = int(symptom_vector.sum())
        
        # Get predictions
        proba = xgb_model.predict_proba(symptom_vector)[0]
        class_names = label_encoder.inverse_transform(np.arange(len(proba)))
        
        # Create results dataframe
        results = pd.DataFrame({
            'disease': class_names,
            'probability': proba
        })
        
        # Get top 5 predictions without probability threshold
        top_predictions = results.sort_values(by='probability', ascending=False).head(5)
        
        # Convert to list of dictionaries
        predictions = []
        for _, row in top_predictions.iterrows():
            predictions.append({
                'disease': row['disease'],
                'probability': float(row['probability']),
                'percentage': f"{row['probability'] * 100:.2f}"
            })
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'symptoms_detected': num_symptoms,
            'total_symptoms': symptom_vector.shape[1],
            'detected_symptoms': detected_symptoms
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'message': 'API is running',
        'config': {
            'default_threshold': DEFAULT_THRESHOLD,
            'max_predictions': MAX_PREDICTIONS
        }
    })

@app.route('/config')
def config():
    """Get configuration settings"""
    return jsonify({
        'default_threshold': DEFAULT_THRESHOLD,
        'max_predictions': MAX_PREDICTIONS
    })

if __name__ == '__main__':
    app.run(debug=FLASK_DEBUG, host=FLASK_HOST, port=FLASK_PORT)
