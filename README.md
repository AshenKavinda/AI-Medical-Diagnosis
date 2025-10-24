# AI Medical Diagnosis System

A Flask-based web application that uses Machine Learning (XGBoost) and Natural Language Processing (RapidFuzz) to predict diseases from symptom descriptions.

## Features

- 🤖 AI-powered disease prediction using XGBoost
- 🔍 Natural language symptom processing with fuzzy matching
- 🎨 Modern dark-themed UI with Tailwind CSS
- ⚙️ Configurable threshold and probability settings
- 📊 Top predictions with confidence scores

## Installation

1. Install required packages:
```bash
pip install flask python-dotenv joblib numpy pandas xgboost scikit-learn rapidfuzz
```

2. Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
```

3. Ensure model files are present:
- `illness_predictor_model.pkl`
- `label_encoder.pkl`
- `symptom_features.pkl`

## Configuration (.env)

- `FLASK_PORT`: Server port (default: 5000)
- `FLASK_HOST`: Server host (default: 0.0.0.0)
- `DEFAULT_THRESHOLD`: Fuzzy matching threshold 70-100 (default: 90)
- `MIN_PROBABILITY`: Minimum confidence to display results 0.0-1.0 (default: 0.40)
- `MAX_PREDICTIONS`: Maximum predictions to show (default: 5)

## Usage

1. Start the server:
```bash
python app.py
```

2. Open browser: http://localhost:5000

3. Enter symptom description and click "Analyze Symptoms"

## API Endpoints

- `GET /` - Main application page
- `POST /predict` - Analyze symptoms and get predictions
- `GET /health` - Health check
- `GET /config` - Get current configuration

## Notes

- Only predictions above `MIN_PROBABILITY` (40%) are displayed
- Adjust `DEFAULT_THRESHOLD` for stricter/looser symptom matching
- For medical use, always consult qualified healthcare professionals
