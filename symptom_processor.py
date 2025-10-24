import joblib
import numpy as np
from rapidfuzz import fuzz

# Load symptom list
symptom_features = joblib.load("symptom_features.pkl")

def text_to_vector(text, threshold=90, use_token_set=True):
    
    text = text.lower()
    vector = np.zeros(len(symptom_features))
    detected_symptoms = []
    
    for i, symptom in enumerate(symptom_features):
        symptom_clean = symptom.replace("_", " ")
        
        # Method 1: Exact substring match (fastest and most accurate)
        if symptom_clean in text:
            print(f"✓ Exact match: {symptom}")
            vector[i] = 1
            detected_symptoms.append({
                'symptom': symptom_clean,
                'match_type': 'exact',
                'score': 100
            })
            continue
        
        # Method 2: Token set ratio - better for word-based matching
        # This ignores word order and finds common words
        if use_token_set:
            score = fuzz.token_set_ratio(symptom_clean, text)
        else:
            score = fuzz.partial_ratio(symptom_clean, text)
            
        if score >= threshold:
            print(f"✓ Fuzzy match ({score:.1f}%): {symptom}")
            vector[i] = 1
            detected_symptoms.append({
                'symptom': symptom_clean,
                'match_type': 'fuzzy',
                'score': score
            })
    
    return vector.reshape(1, -1), detected_symptoms

# Example usage of your function
# Replace "some text with fever and cough" with your actual input
user_text = "The patient reports noticing an abnormal appearance of the skin over the past few weeks, along with a slightly swollen area that has gradually increased in size. There is a distinct skin lesion and raised growth on the affected region. The patient also mentions having several skin moles, one of which has changed in size and color recently. Additionally, swelling around nearby lymph nodes has been observed. The scalp appears irregular, and the patient has experienced an itchy sensation on the eyelid.These symptoms suggest a potential melanoma or other serious skin condition, warranting immediate dermatological evaluation."
vector_output = text_to_vector(user_text)

# Print the resulting vector
print(vector_output)
