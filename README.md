# 🏥 AI Medical Diagnosis System

An intelligent disease prediction system powered by Machine Learning that analyzes patient symptom descriptions in natural language and provides accurate disease predictions with confidence scores.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Machine Learning Model](#machine-learning-model)
- [Web Application](#web-application)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Technologies Used](#technologies-used)
- [Disclaimer](#disclaimer)

---

## 🎯 Overview

This AI-powered medical diagnosis system combines natural language processing with machine learning to predict diseases based on symptom descriptions. The system uses **XGBoost** classifier trained on a comprehensive disease-symptom dataset and employs **RapidFuzz** for intelligent symptom matching from free-text input.

### Key Highlights
- ✨ **Natural Language Input**: Describe symptoms in plain English
- 🎯 **High Accuracy**: XGBoost model trained on extensive medical data
- 🧠 **Intelligent Matching**: Fuzzy string matching for symptom detection
- 📊 **Top-5 Predictions**: Get multiple disease predictions with confidence scores
- 🌐 **Modern Web UI**: Beautiful, responsive interface built with TailwindCSS
- ⚡ **Real-time Analysis**: Instant predictions via REST API

---

## ✨ Features

### Core Functionality
- 🔍 **Symptom Text Analysis**: Parse natural language symptom descriptions
- 🎯 **Disease Prediction**: Multi-class classification with probability scores
- 📈 **Confidence Scoring**: Percentage-based confidence for each prediction
- 🏷️ **Symptom Detection**: Automatic extraction of symptoms from text
- 🔄 **Flexible Matching**: Both exact and fuzzy string matching support

### User Experience
- 💻 **Web Interface**: Clean, intuitive UI for symptom input
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile
- 🌙 **Dark Theme**: Eye-friendly dark mode interface
- ⚙️ **Configurable Settings**: Adjustable threshold and prediction limits
- 📋 **Detailed Results**: Shows detected symptoms and match confidence

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Web Browser                          │
│              (Modern UI with TailwindCSS)                   │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP/JSON
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Flask Web Server                          │
│                     (app.py)                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Routes: /, /predict, /health, /config               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Symptom Processor Module                       │
│               (symptom_processor.py)                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  • Text cleaning & normalization                      │  │
│  │  • RapidFuzz matching (exact & fuzzy)                │  │
│  │  • Symptom vector generation                         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  XGBoost ML Model                           │
│            (illness_predictor_model.pkl)                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  • Multi-class classification                         │  │
│  │  • 200 estimators, max_depth=8                       │  │
│  │  • Trained on filtered disease dataset              │  │
│  │  • Returns probability distribution                  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🤖 Machine Learning Model

### Model Details

**File**: `ml_model/cw_v1_200.ipynb`

#### Dataset
- **Source**: `disease.csv` - Comprehensive disease-symptom dataset
- **Processing**: Filtered to include diseases with minimum 30 samples
- **Features**: Binary symptom vectors (presence/absence of symptoms)
- **Target**: Disease labels (multi-class classification)

#### Model Architecture

```python
XGBoost Classifier Configuration:
├── Objective: multi:softprob
├── Tree Method: hist
├── Max Leaves: 64
├── Estimators: 200
├── Max Depth: 8
├── Learning Rate: 0.05
├── Evaluation Metric: mlogloss
└── Random State: 100
```

#### Training Process

1. **Data Loading & Exploration**
   - Load disease dataset
   - Analyze disease distribution
   - Identify rare diseases

2. **Data Preprocessing**
   ```python
   - Filter diseases with min_support >= 30 samples
   - Label encoding for target variable
   - Train-test split (80/20) with stratification
   ```

3. **Model Training**
   - XGBoost with multi-class softmax
   - 200 decision trees (n_estimators)
   - Early stopping with validation set
   - Performance monitoring with mlogloss

4. **Evaluation Metrics**
   - Accuracy score on test set
   - Classification report (precision, recall, F1)
   - Top-k accuracy analysis

5. **Model Persistence**
   ```python
   - illness_predictor_model.pkl (XGBoost model)
   - label_encoder.pkl (Disease label encoder)
   - symptom_features.pkl (Symptom feature names)
   ```

#### Prediction Pipeline

```python
Input Text → Symptom Extraction → Feature Vector → XGBoost → Probabilities → Top-5 Diseases
```

#### Model Performance
- **High Accuracy**: Trained on balanced, filtered dataset
- **Multi-class Support**: Handles multiple disease categories
- **Probability Scores**: Provides confidence for each prediction
- **Robust Predictions**: 200 trees ensure stable results

---

## 🌐 Web Application

### Application Structure

**Main File**: `app.py` - Flask web server with REST API

#### Core Components

##### 1. **Flask Server (`app.py`)**
```python
Features:
├── Model Loading (startup)
├── Environment Configuration (.env)
├── Routes:
│   ├── / → Home page
│   ├── /predict → Symptom analysis endpoint
│   ├── /health → Health check
│   └── /config → Get configuration
└── Error Handling
```

##### 2. **Symptom Processor (`symptom_processor.py`)**
```python
Functions:
├── text_to_vector(text, threshold, use_token_set)
│   ├── Text normalization
│   ├── Exact substring matching
│   ├── Fuzzy matching (RapidFuzz)
│   └── Returns: symptom vector + detected symptoms
```

**Matching Strategy**:
- **Exact Match**: Fast substring search (100% accuracy)
- **Fuzzy Match**: Token set ratio for flexible matching
- **Configurable Threshold**: Adjustable matching sensitivity (default: 90%)

##### 3. **Frontend (`templates/index.html`)**
```html
Features:
├── Modern Dark Theme
├── Responsive Design
├── Real-time Form Submission
├── Loading States
├── Error Handling
├── Results Display:
│   ├── Top-5 Disease Predictions
│   ├── Confidence Percentages
│   ├── Detected Symptoms List
│   └── Match Type Badges
└── Interactive Examples
```

#### API Endpoints

##### `POST /predict`
**Request**:
```json
{
  "symptoms": "Patient has fever, cough, and difficulty breathing...",
  "threshold": 90
}
```

**Response**:
```json
{
  "success": true,
  "predictions": [
    {
      "disease": "common_cold",
      "probability": 0.85,
      "percentage": "85.00"
    }
  ],
  "symptoms_detected": 7,
  "total_symptoms": 132,
  "detected_symptoms": [
    {
      "symptom": "fever",
      "match_type": "exact",
      "score": 100
    }
  ]
}
```

##### `GET /health`
Check API status
```json
{
  "status": "healthy",
  "message": "API is running",
  "config": {
    "default_threshold": 90,
    "max_predictions": 5
  }
}
```

##### `GET /config`
Get current configuration
```json
{
  "default_threshold": 90,
  "max_predictions": 5
}
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum
- Modern web browser

### Step 1: Clone Repository
```bash
git clone https://github.com/AshenKavinda/AI-Medical-Diagnosis.git
cd AI-Medical-Diagnosis
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install flask
pip install xgboost
pip install scikit-learn
pip install pandas
pip install numpy
pip install joblib
pip install rapidfuzz
pip install python-dotenv
```

### Step 4: Environment Configuration
Create a `.env` file in the project root:
```env
FLASK_PORT=5000
FLASK_HOST=0.0.0.0
FLASK_DEBUG=True
DEFAULT_THRESHOLD=90
MAX_PREDICTIONS=5
```

### Step 5: Verify Model Files
Ensure these files exist in the project root:
- `illness_predictor_model.pkl`
- `label_encoder.pkl`
- `symptom_features.pkl`

If missing, run the notebook `ml_model/cw_v1_200.ipynb` to generate them.

---

## 💻 Usage

### Starting the Application

1. **Activate Virtual Environment**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Run Flask Server**
   ```bash
   python app.py
   ```

3. **Access Web Interface**
   - Open browser: `http://localhost:5000`
   - Or: `http://127.0.0.1:5000`

### Using the Web Interface

1. **Enter Symptoms**
   - Type patient symptoms in the text area
   - Be as detailed as possible
   - Use natural language

2. **Analyze**
   - Click "Analyze Symptoms" button
   - Wait for AI processing

3. **View Results**
   - Top 5 disease predictions with confidence scores
   - List of detected symptoms
   - Match type indicators (exact/fuzzy)

### Testing the Model

Run the test script to verify model functionality:
```bash
python test_model.py
```

This will:
- Load the trained model
- Process sample symptom text
- Display top predictions
- Show detected symptoms

---

## 📁 Project Structure

```
AI-Medical-Diagnosis/
├── 📄 app.py                          # Flask web application
├── 📄 symptom_processor.py            # NLP symptom extraction
├── 📄 test_model.py                   # Model testing script
├── 📄 README.md                       # Project documentation
├── 📄 .env                            # Environment configuration
├── 📦 illness_predictor_model.pkl     # Trained XGBoost model
├── 📦 label_encoder.pkl               # Disease label encoder
├── 📦 symptom_features.pkl            # Symptom feature list
│
├── 📁 ml_model/                       # Machine Learning files
│   ├── 📓 cw_v1_200.ipynb            # Jupyter notebook (training)
│   └── 📊 disease.csv                 # Disease-symptom dataset
│
├── 📁 templates/                      # HTML templates
│   └── 📄 index.html                  # Main web interface
│
└── 📁 __pycache__/                    # Python cache files
```

---

## 📡 API Documentation

### Base URL
```
http://localhost:5000
```

### Authentication
No authentication required (development mode)

### Endpoints

#### 1. Predict Disease
```http
POST /predict
Content-Type: application/json

{
  "symptoms": "string",
  "threshold": number (optional, default: 90)
}
```

**Success Response (200)**:
```json
{
  "success": true,
  "predictions": [
    {
      "disease": "string",
      "probability": number,
      "percentage": "string"
    }
  ],
  "symptoms_detected": number,
  "total_symptoms": number,
  "detected_symptoms": [
    {
      "symptom": "string",
      "match_type": "exact|fuzzy",
      "score": number
    }
  ]
}
```

**Error Response (400/500)**:
```json
{
  "error": "string"
}
```

#### 2. Health Check
```http
GET /health
```

**Response (200)**:
```json
{
  "status": "healthy",
  "message": "API is running",
  "config": {
    "default_threshold": number,
    "max_predictions": number
  }
}
```

#### 3. Get Configuration
```http
GET /config
```

**Response (200)**:
```json
{
  "default_threshold": number,
  "max_predictions": number
}
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `FLASK_PORT` | int | 5000 | Server port number |
| `FLASK_HOST` | string | 0.0.0.0 | Server host address |
| `FLASK_DEBUG` | boolean | True | Debug mode |
| `DEFAULT_THRESHOLD` | int | 90 | Symptom matching threshold (0-100) |
| `MAX_PREDICTIONS` | int | 5 | Maximum predictions to return |

### Adjusting Matching Threshold

Lower threshold = More lenient matching (may include false positives)
Higher threshold = Stricter matching (may miss some symptoms)

**Recommended**: 85-95 for balanced results

---

## 🛠️ Technologies Used

### Backend
- **Flask** (2.0+) - Web framework
- **XGBoost** - Machine learning model
- **scikit-learn** - ML utilities & preprocessing
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **RapidFuzz** - Fuzzy string matching
- **Joblib** - Model serialization
- **python-dotenv** - Environment management

### Frontend
- **HTML5** - Structure
- **TailwindCSS** - Styling & UI components
- **Vanilla JavaScript** - Interactivity & AJAX
- **SVG Icons** - Vector graphics

### Machine Learning Pipeline
- **XGBoost Classifier** - Disease prediction
- **Label Encoding** - Target variable encoding
- **Train-Test Split** - Model validation
- **Stratified Sampling** - Balanced dataset split

---

## ⚠️ Disclaimer

**IMPORTANT MEDICAL NOTICE**

This AI Medical Diagnosis System is designed **for educational and research purposes only**. 

### Limitations
- ❌ **NOT a substitute** for professional medical advice
- ❌ **NOT suitable** for clinical diagnosis
- ❌ **NOT validated** for real-world medical use
- ❌ **NOT approved** by medical regulatory bodies

### Recommendations
- ✅ **Always consult** qualified healthcare professionals
- ✅ **Use only** as a learning tool or reference
- ✅ **Verify** all predictions with medical experts
- ✅ **Seek immediate** medical attention for serious symptoms

### Liability
The developers and contributors of this project:
- Accept **NO responsibility** for medical decisions made using this tool
- Provide **NO warranties** of accuracy or reliability
- Recommend **professional medical consultation** for all health concerns

**If you are experiencing a medical emergency, call your local emergency services immediately.**

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

---

## 📧 Contact

**Repository**: [AI-Medical-Diagnosis](https://github.com/AshenKavinda/AI-Medical-Diagnosis)

**Owner**: AshenKavinda

---

## 🌟 Acknowledgments

- XGBoost development team
- scikit-learn community
- Flask framework developers
- Medical dataset contributors
- Open-source community

---

**Made with ❤️ for healthcare AI education**
