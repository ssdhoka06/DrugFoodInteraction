# CORRECTED Feature Engineering - Replace your create_features function with this

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import pickle

# You need to save these during training and load them here
# Add these lines to your training code to save the vectorizers and scaler:
"""
# After training, save the fitted vectorizers and scaler:
with open('drug_tfidf.pkl', 'wb') as f:
    pickle.dump(drug_tfidf, f)
with open('food_tfidf.pkl', 'wb') as f:
    pickle.dump(food_tfidf, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
"""

@st.cache_resource
def load_preprocessing_objects():
    """Load the fitted TF-IDF vectorizers and scaler"""
    try:
        with open('drug_tfidf.pkl', 'rb') as f:
            drug_tfidf = pickle.load(f)
        with open('food_tfidf.pkl', 'rb') as f:
            food_tfidf = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return drug_tfidf, food_tfidf, scaler
    except FileNotFoundError:
        st.error("❌ Preprocessing objects not found. You need to save drug_tfidf.pkl, food_tfidf.pkl, and scaler.pkl from your training code.")
        return None, None, None

def categorize_drug(drug_name):
    """Categorize drug based on name - replicate your original logic"""
    drug_lower = drug_name.lower()
    
    # Add your original drug categorization logic here
    if any(x in drug_lower for x in ['warfarin', 'coumadin', 'heparin']):
        return 'anticoagulant'
    elif any(x in drug_lower for x in ['fentanyl', 'morphine', 'codeine', 'tramadol']):
        return 'analgesic'  # or 'pain_relief' - check your original categories
    elif any(x in drug_lower for x in ['amoxicillin', 'penicillin', 'azithromycin']):
        return 'antibiotic'
    # Add all your original drug categories here
    else:
        return 'other'

def categorize_food(food_name):
    """Categorize food based on name - replicate your original logic"""
    food_lower = food_name.lower()
    
    # Add your original food categorization logic here
    if any(x in food_lower for x in ['grapefruit', 'grape fruit', 'orange', 'lemon']):
        return 'citrus'
    elif any(x in food_lower for x in ['spinach', 'kale', 'lettuce']):
        return 'leafy_greens'  # Check your original category name
    elif any(x in food_lower for x in ['milk', 'cheese', 'yogurt']):
        return 'dairy'
    # Add all your original food categories here
    else:
        return 'other'

def get_mechanism(drug_category, food_category):
    """Determine interaction mechanism - replicate your original logic"""
    # Add your original mechanism determination logic
    if drug_category == 'anticoagulant' and food_category == 'leafy_greens':
        return 'vitamin_k_competition'
    elif food_category == 'citrus':
        return 'cyp3a4_inhibition'
    # Add all your original mechanism mappings
    else:
        return 'other'

def get_risk_level(drug_category, food_category, mechanism):
    """Determine risk level - replicate your original logic"""
    # Add your original risk level determination logic
    if drug_category == 'anticoagulant' and food_category == 'leafy_greens':
        return 'HIGH'
    elif mechanism == 'cyp3a4_inhibition':
        return 'MODERATE'
    # Add all your original risk mappings
    else:
        return 'LOW'

def create_detailed_risk_score(risk_level, mechanism, drug_category, food_category):
    """Create detailed risk score - EXACT copy from your training code"""
    base_scores = {'HIGH': 4, 'MODERATE': 2, 'LOW': 1}
    base_score = base_scores.get(risk_level, 1)
    
    if mechanism in ['cyp3a4_inhibition', 'vitamin_k_competition']:
        base_score += 1
    if drug_category == 'anticoagulant' and food_category == 'leafy_greens':
        base_score += 2
    
    return base_score

def create_correct_features(drug_input, food_input, drug_tfidf, food_tfidf, scaler):
    """Create features EXACTLY as in training"""
    
    # Step 1: Create a single row dataframe (like your training data)
    temp_df = pd.DataFrame({
        'drug': [drug_input],
        'food': [food_input]
    })
    
    # Step 2: Add categorical features
    temp_df['drug_category'] = temp_df['drug'].apply(categorize_drug)
    temp_df['food_category'] = temp_df['food'].apply(categorize_food)
    temp_df['mechanism'] = temp_df.apply(lambda x: get_mechanism(x['drug_category'], x['food_category']), axis=1)
    temp_df['risk_level'] = temp_df.apply(lambda x: get_risk_level(x['drug_category'], x['food_category'], x['mechanism']), axis=1)
    
    # Step 3: Create TF-IDF features
    drug_tfidf_features = drug_tfidf.transform(temp_df['drug']).toarray()
    food_tfidf_features = food_tfidf.transform(temp_df['food']).toarray()
    
    # Step 4: Create dummy variables (one-hot encoding)
    drug_dummies = pd.get_dummies(temp_df['drug_category'], prefix='drug').astype(int)
    food_dummies = pd.get_dummies(temp_df['food_category'], prefix='food').astype(int)
    mechanism_dummies = pd.get_dummies(temp_df['mechanism'], prefix='mechanism').astype(int)
    risk_dummies = pd.get_dummies(temp_df['risk_level'], prefix='risk').astype(int)
    
    # Step 5: Calculate additional features
    temp_df['risk_score'] = temp_df.apply(
        lambda x: create_detailed_risk_score(x['risk_level'], x['mechanism'], x['drug_category'], x['food_category']), 
        axis=1
    )
    temp_df['drug_length'] = temp_df['drug'].str.len()
    temp_df['food_length'] = temp_df['food'].str.len()
    
    # Name similarity (you might need to adjust this based on your original method)
    from difflib import SequenceMatcher
    temp_df['name_similarity'] = temp_df.apply(
        lambda x: SequenceMatcher(None, x['drug'].lower(), x['food'].lower()).ratio(), 
        axis=1
    )
    
    temp_df['same_category'] = (temp_df['drug_category'] == temp_df['food_category']).astype(int)
    temp_df['both_other'] = ((temp_df['drug_category'] == 'other') & (temp_df['food_category'] == 'other')).astype(int)
    
    # Step 6: Combine all features
    # TF-IDF features
    drug_tfidf_df = pd.DataFrame(drug_tfidf_features, columns=[f'drug_tfidf_{i}' for i in range(drug_tfidf_features.shape[1])])
    food_tfidf_df = pd.DataFrame(food_tfidf_features, columns=[f'food_tfidf_{i}' for i in range(food_tfidf_features.shape[1])])
    
    # Combine all features
    feature_df = pd.concat([
        temp_df[['risk_score', 'drug_length', 'food_length', 'name_similarity', 'same_category', 'both_other']],
        drug_dummies,
        food_dummies,
        mechanism_dummies,
        risk_dummies,
        drug_tfidf_df,
        food_tfidf_df
    ], axis=1)
    
    # Step 7: Handle missing columns (ensure all training columns are present)
    # You need to get the exact column list from your training data
    # This is crucial - the model expects EXACTLY the same columns in the same order
    
    # Step 8: Scale features (if you used StandardScaler in training)
    if scaler is not None:
        feature_df_scaled = scaler.transform(feature_df)
        feature_df = pd.DataFrame(feature_df_scaled, columns=feature_df.columns)
    
    return feature_df

# UPDATED prediction function
def predict_interaction_corrected(drug_input, food_input, model, drug_tfidf, food_tfidf, scaler):
    """Make prediction using CORRECT feature engineering"""
    try:
        # Create features exactly as in training
        features = create_correct_features(drug_input, food_input, drug_tfidf, food_tfidf, scaler)
        
        # Debug: Show feature info
        st.sidebar.write(f"Generated features shape: {features.shape}")
        st.sidebar.write(f"First 5 feature names: {list(features.columns[:5])}")
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = max(probabilities)
        
        return prediction, confidence, probabilities
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None
