import os
import pandas as pd
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Configuration - Fixed paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Try relative paths first, then absolute paths as fallback
DATA_PATH_RELATIVE = os.path.join(BASE_DIR, 'data', 'balanced_drug_food_interactions.csv')
DATA_PATH_ABSOLUTE = '/Users/sachidhoka/Desktop/balanced_drug_food_interactions.csv'

MODEL_PATH_RELATIVE = os.path.join(BASE_DIR, 'models', 'best_drug_food_interaction_model.pkl')
MODEL_PATH_ABSOLUTE = '/Users/sachidhoka/Desktop/best_drug_food_interaction_model.pkl'

# Function to find the correct file path
def get_file_path(relative_path, absolute_path):
    if os.path.exists(relative_path):
        return relative_path
    elif os.path.exists(absolute_path):
        return absolute_path
    else:
        raise FileNotFoundError(f"File not found at either {relative_path} or {absolute_path}")

# Get correct paths
DATA_PATH = get_file_path(DATA_PATH_RELATIVE, DATA_PATH_ABSOLUTE)
MODEL_PATH = get_file_path(MODEL_PATH_RELATIVE, MODEL_PATH_ABSOLUTE)

# Load dataset and model
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded successfully from: {DATA_PATH}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

try:
    model_package = joblib.load(MODEL_PATH)
    model = model_package['model']
    scaler = model_package['scaler']
    feature_info = model_package['feature_info']
    drug_categories = model_package['drug_categories']
    food_categories = model_package['food_categories']
    high_risk_interactions = model_package['high_risk_interactions']
    print(f"Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

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
    drugs = sorted([drug for drug in df['drug'].unique() if pd.notna(drug)])
    foods = sorted([food for food in df['food'].unique() if pd.notna(food)])
    
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

@app.route('/api/drugs')
def get_drugs():
    """API endpoint to get all drugs"""
    drugs = sorted(df['drug'].unique().tolist())
    return {'drugs': drugs}

@app.route('/api/foods')
def get_foods():
    """API endpoint to get all foods"""
    foods = sorted(df['food'].unique().tolist())
    return {'foods': foods}

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    data = request.get_json()
    drug = data.get('drug')
    food = data.get('food')
    
    if not drug or not food:
        return {'error': 'Both drug and food must be provided'}, 400
    
    # Check for known interaction first
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
            'risk_level': interaction['risk_level'],
            'source': 'known_interaction'
        }
    else:
        # Predict new interaction
        result = predict_interaction(drug, food)
        result['source'] = 'prediction'
    
    return result

if __name__ == '__main__':
    print("Starting Flask application...")
    print(f"Dataset contains {len(df)} records")
    print(f"Unique drugs: {len(df['drug'].unique())}")
    print(f"Unique foods: {len(df['food'].unique())}")
    app.run(debug=True, port=5001)
