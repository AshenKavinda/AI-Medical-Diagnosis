import joblib
import numpy as np
import pandas as pd
from symptom_processor import text_to_vector

# Load the trained model and label encoder
print("Loading model and encoders...")
xgb_model = joblib.load("illness_predictor_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

print("Model loaded successfully!\n")

# Test with the same user text
user_text = """I have fever, cough, sore throat, runny nose, nasal congestion, 
sneezing, fatigue, headache, and chills."""

print("=" * 80)
print("USER INPUT TEXT:")
print("=" * 80)
print(user_text)
print("\n" + "=" * 80)

# Convert text to symptom vector
print("\n🔍 Detecting symptoms in text...")
print("-" * 80)
new_patient_data, matches = text_to_vector(user_text, threshold=90)  # Increased threshold for accuracy
print("-" * 80)
print(f"\n✅ Symptom vector created: {new_patient_data.shape}")
print(f"   Total symptoms detected: {int(new_patient_data.sum())}")

# Get prediction probabilities for all classes
print("\n🤖 Running model prediction...")
proba = xgb_model.predict_proba(new_patient_data)[0]  # shape: (num_classes,)

# Get class names from label encoder
class_names = label_encoder.inverse_transform(np.arange(len(proba)))

# Combine class names and probabilities into a DataFrame
results = pd.DataFrame({
    'Disease': class_names,
    'Probability': proba
})

# Sort by probability (descending) and display top 5
top_5 = results.sort_values(by='Probability', ascending=False).head(5)

print("\n" + "=" * 80)
print("🔝 TOP 5 PREDICTED DISEASES:")
print("=" * 80)
print(top_5.to_string(index=False))

# Also show the top prediction
top_prediction = top_5.iloc[0]
print("\n" + "=" * 80)
print("🎯 MOST LIKELY DIAGNOSIS:")
print("=" * 80)
print(f"   Disease: {top_prediction['Disease']}")
print(f"   Confidence: {top_prediction['Probability']*100:.2f}%")
print("=" * 80)

# Optional: Show all predictions above 1% probability
print("\n📊 All diseases with >1% probability:")
significant_results = results[results['Probability'] > 0.01].sort_values(by='Probability', ascending=False)
print(significant_results.to_string(index=False))
