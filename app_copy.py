import os
import pandas as pd
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'dfi project', 'data', 'balanced_drug_food_interactions.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'dfi project', 'models', 'best_drug_food_interaction_model.pkl')

# Load dataset and model
df = pd.read_csv(DATA_PATH)
model_package = joblib.load(MODEL_PATH)
model = model_package['model']
scaler = model_package['scaler']
feature_info = model_package['feature_info']
drug_categories = model_package['drug_categories']
food_categories = model_package['food_categories']
high_risk_interactions = model_package['high_risk_interactions']

def categorize_entity(entity, categories):
    entity_lower = str(entity).lower()
    best_match = 'other'
    max_matches = 0
    
    for category, items in categories.items():
        matches = sum(1 for item in items if item in entity_lower)
        if matches > max_matches:
            max_matches = matches
            best_match = category
    
    return best_match

def get_interaction_details(drug_cat, food_cat):
    for (d_cat, f_cat), details in high_risk_interactions.items():
        if d_cat == drug_cat and f_cat == food_cat:
            return details['mechanism'], details['risk']
    return 'unknown', 'LOW'

def predict_interaction(drug, food):
    # Create DataFrame for prediction
    new_df = pd.DataFrame({
        'drug': [drug.lower().strip()],
        'food': [food.lower().strip()],
        'interaction': [0]
    })
    
    # Categorize drug and food
    new_df['drug_category'] = new_df['drug'].apply(lambda x: categorize_entity(x, drug_categories))
    new_df['food_category'] = new_df['food'].apply(lambda x: categorize_entity(x, food_categories))
    
    # Get interaction details
    interaction_details = new_df.apply(
        lambda x: get_interaction_details(x['drug_category'], x['food_category']), 
        axis=1
    )
    new_df['mechanism'] = [details[0] for details in interaction_details]
    new_df['risk_level'] = [details[1] for details in interaction_details]
    
    # Feature engineering
    drug_dummies = pd.get_dummies(new_df['drug_category'], prefix='drug').reindex(columns=feature_info['feature_names'], fill_value=0)
    food_dummies = pd.get_dummies(new_df['food_category'], prefix='food').reindex(columns=feature_info['feature_names'], fill_value=0)
    mechanism_dummies = pd.get_dummies(new_df['mechanism'], prefix='mechanism').reindex(columns=feature_info['feature_names'], fill_value=0)
    risk_dummies = pd.get_dummies(new_df['risk_level'], prefix='risk').reindex(columns=feature_info['feature_names'], fill_value=0)
    
    # Create feature matrix
    X_new = pd.concat([
        drug_dummies,
        food_dummies,
        mechanism_dummies,
        risk_dummies
    ], axis=1)
    
    # Ensure all feature columns are present
    missing_cols = set(feature_info['feature_names']) - set(X_new.columns)
    for col in missing_cols:
        X_new[col] = 0
    
    # Select and scale features
    X_new = X_new[feature_info['feature_names']]
    X_new_scaled = scaler.transform(X_new)
    
    # Make prediction
    prediction = model.predict(X_new_scaled)[0]
    probability = model.predict_proba(X_new_scaled)[0, 1]
    
    return {
        'drug': drug,
        'food': food,
        'interaction_predicted': bool(prediction),
        'probability': float(probability),
        'drug_category': new_df['drug_category'].iloc[0],
        'food_category': new_df['food_category'].iloc[0],
        'mechanism': new_df['mechanism'].iloc[0],
        'risk_level': new_df['risk_level'].iloc[0]
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    # Get unique drugs and foods from dataset
    drugs = sorted(df['drug'].unique().tolist())
    foods = sorted(df['food'].unique().tolist())
    
    result = None
    scroll_to_results = False
    
    if request.method == 'POST':
        drug = request.form.get('drug')
        food = request.form.get('food')
        
        if drug and food:
            # First check if we have a known interaction
            known_interaction = df[
                (df['drug'].str.lower() == drug.lower()) & 
                (df['food'].str.lower() == food.lower())
            ]
            
            if not known_interaction.empty:
                interaction = known_interaction.iloc[0].to_dict()
                result = {
                    'drug': interaction['drug'],
                    'food': interaction['food'],
                    'interaction_predicted': bool(interaction['interaction']),
                    'probability': 1.0 if interaction['interaction'] else 0.0,
                    'drug_category': interaction['drug_category'],
                    'food_category': interaction['food_category'],
                    'mechanism': interaction['mechanism'],
                    'risk_level': interaction['risk_level']
                }
            else:
                # Predict new interaction
                result = predict_interaction(drug, food)
            
            # Flag to scroll to results
            scroll_to_results = True
    
    return render_template(
        'index.html',
        drugs=drugs,
        foods=foods,
        result=result,
        scroll_to_results=scroll_to_results
    )


if __name__ == '__main__':
    app.run(debug=True, port=5001)
