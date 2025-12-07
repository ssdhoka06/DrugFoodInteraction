import os
import pandas as pd
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Fixed paths - using absolute paths
DATA_PATH = '/Users/sachidhoka/Desktop/balanced_drug_food_interactions.csv'
MODEL_PATH = '/Users/sachidhoka/Desktop/College/ASEP_2(Drug-Food)/dfi project/models/best_drug_food_interaction_model.pkl'

# Load dataset and model
print("Loading dataset...")
try:
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Dataset loaded: {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    df = pd.DataFrame()

print("Loading model...")
try:
    model_package = joblib.load(MODEL_PATH)
    model = model_package['model']
    scaler = model_package.get('scaler', StandardScaler())
    feature_info = model_package.get('feature_info', {})
    drug_categories = model_package.get('drug_categories', {})
    food_categories = model_package.get('food_categories', {})
    high_risk_interactions = model_package.get('high_risk_interactions', {})
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = StandardScaler()
    feature_info = {}
    drug_categories = {}
    food_categories = {}
    high_risk_interactions = {}

def categorize_entity(entity, categories):
    """Categorize drug or food based on predefined categories"""
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
    """Get interaction mechanism and risk level"""
    for (d_cat, f_cat), details in high_risk_interactions.items():
        if d_cat == drug_cat and f_cat == food_cat:
            return details['mechanism'], details['risk']
    return 'unknown', 'LOW'

def get_clinical_explanation(mechanism, risk_level):
    """Provide patient-friendly explanation"""
    explanations = {
        'cyp3a4_inhibition': "This food can block liver enzymes that break down your medication, potentially causing it to build up to dangerous levels in your body.",
        'vitamin_k_competition': "This food contains vitamin K which can reduce the effectiveness of blood-thinning medications.",
        'calcium_chelation': "Calcium in this food can bind to your medication and prevent your body from absorbing it properly.",
        'absorption_interference': "This food may slow down or reduce how well your body absorbs the medication.",
        'cns_depression': "Combining these can increase drowsiness and impair coordination and judgment.",
        'hypoglycemia_risk': "This combination may cause dangerously low blood sugar levels.",
        'arrhythmia_risk': "This combination may affect your heart rhythm.",
        'gi_bleeding_risk': "This combination may increase the risk of stomach bleeding.",
        'bp_elevation': "This combination may reduce the effectiveness of blood pressure medication.",
        'fluid_retention': "This combination may cause increased fluid retention and swelling.",
        'unknown': "The specific interaction mechanism is not fully understood."
    }
    
    recommendations = {
        'HIGH': "🚨 AVOID this combination. Consult your healthcare provider immediately before taking these together.",
        'MODERATE': "⚠️ Use with CAUTION. Consider spacing them 2-4 hours apart and monitor for side effects.",
        'LOW': "✅ Generally safe, but follow standard precautions and monitor for any unusual symptoms."
    }
    
    return {
        'explanation': explanations.get(mechanism, explanations['unknown']),
        'recommendation': recommendations.get(risk_level, recommendations['LOW'])
    }

def predict_interaction(drug, food):
    """Predict drug-food interaction"""
    if model is None or df.empty:
        return {'error': 'Model or data not available'}
    
    # Check for known interaction first
    known_interaction = df[
        (df['drug'].str.lower() == drug.lower()) & 
        (df['food'].str.lower() == food.lower())
    ]
    
    if not known_interaction.empty:
        interaction = known_interaction.iloc[0]
        mechanism = interaction.get('mechanism', 'unknown')
        risk_level = interaction.get('risk_level', 'LOW')
        clinical_info = get_clinical_explanation(mechanism, risk_level)
        
        return {
            'drug': drug.title(),
            'food': food.title(),
            'interaction_predicted': bool(interaction.get('interaction', 0)),
            'probability': 1.0 if interaction.get('interaction', 0) else 0.0,
            'drug_category': interaction.get('drug_category', 'other'),
            'food_category': interaction.get('food_category', 'other'),
            'mechanism': mechanism,
            'risk_level': risk_level,
            'explanation': clinical_info['explanation'],
            'recommendation': clinical_info['recommendation'],
            'source': 'known_database'
        }
    
    # If not in database, use model prediction
    try:
        new_df = pd.DataFrame({
            'drug': [drug.lower().strip()],
            'food': [food.lower().strip()],
            'interaction': [0]
        })
        
        # Categorize
        new_df['drug_category'] = new_df['drug'].apply(lambda x: categorize_entity(x, drug_categories))
        new_df['food_category'] = new_df['food'].apply(lambda x: categorize_entity(x, food_categories))
        
        # Get interaction details
        interaction_details = new_df.apply(
            lambda x: get_interaction_details(x['drug_category'], x['food_category']), 
            axis=1
        )
        new_df['mechanism'] = [details[0] for details in interaction_details]
        new_df['risk_level'] = [details[1] for details in interaction_details]
        
        # Create features (simplified for compatibility)
        drug_dummies = pd.get_dummies(new_df['drug_category'], prefix='drug')
        food_dummies = pd.get_dummies(new_df['food_category'], prefix='food')
        mechanism_dummies = pd.get_dummies(new_df['mechanism'], prefix='mechanism')
        risk_dummies = pd.get_dummies(new_df['risk_level'], prefix='risk')
        
        # Combine features
        X_new = pd.concat([drug_dummies, food_dummies, mechanism_dummies, risk_dummies], axis=1)
        
        # Ensure all required features are present
        if feature_info.get('feature_names'):
            missing_cols = set(feature_info['feature_names']) - set(X_new.columns)
            for col in missing_cols:
                X_new[col] = 0
            X_new = X_new[feature_info['feature_names']]
        
        # Scale features
        X_new_scaled = scaler.transform(X_new)
        
        # Make prediction
        prediction = model.predict(X_new_scaled)[0]
        probability = model.predict_proba(X_new_scaled)[0, 1]
        
        mechanism = new_df['mechanism'].iloc[0]
        risk_level = new_df['risk_level'].iloc[0]
        clinical_info = get_clinical_explanation(mechanism, risk_level)
        
        return {
            'drug': drug.title(),
            'food': food.title(),
            'interaction_predicted': bool(prediction),
            'probability': float(probability),
            'drug_category': new_df['drug_category'].iloc[0],
            'food_category': new_df['food_category'].iloc[0],
            'mechanism': mechanism,
            'risk_level': risk_level,
            'explanation': clinical_info['explanation'],
            'recommendation': clinical_info['recommendation'],
            'source': 'model_prediction'
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return {'error': f'Prediction failed: {str(e)}'}

def find_similar_interactions(drug, food, limit=3):
    """Find similar interactions in database"""
    if df.empty:
        return []
    
    # Get drug and food categories
    drug_cat = categorize_entity(drug, drug_categories)
    food_cat = categorize_entity(food, food_categories)
    
    # Find similar interactions
    similar = df[
        ((df['drug_category'] == drug_cat) | (df['food_category'] == food_cat)) &
        (df['interaction'] == 1)
    ].head(limit)
    
    return similar[['drug', 'food', 'risk_level']].to_dict('records')

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """Check API status"""
    return jsonify({
        'status': 'online',
        'dataset_loaded': not df.empty,
        'model_loaded': model is not None,
        'total_records': len(df) if not df.empty else 0
    })

@app.route('/api/drugs')
def get_drugs():
    """Get list of drugs with optional search"""
    if df.empty:
        return jsonify({'drugs': []})
    
    search = request.args.get('q', '').lower()
    limit = int(request.args.get('limit', 20))
    
    drugs = df['drug'].unique()
    
    if search:
        drugs = [d for d in drugs if search in str(d).lower()]
    
    drugs = sorted([str(d) for d in drugs if pd.notna(d)])[:limit]
    return jsonify({'drugs': drugs})

@app.route('/api/foods')
def get_foods():
    """Get list of foods with optional search"""
    if df.empty:
        return jsonify({'foods': []})
    
    search = request.args.get('q', '').lower()
    limit = int(request.args.get('limit', 20))
    
    foods = df['food'].unique()
    
    if search:
        foods = [f for f in foods if search in str(f).lower()]
    
    foods = sorted([str(f) for f in foods if pd.notna(f)])[:limit]
    return jsonify({'foods': foods})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Predict drug-food interaction"""
    try:
        data = request.get_json()
        drug = data.get('drug', '').strip()
        food = data.get('food', '').strip()
        
        if not drug or not food:
            return jsonify({
                'success': False,
                'error': 'Both drug and food must be provided'
            }), 400
        
        # Get prediction
        result = predict_interaction(drug, food)
        
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
        
        # Get similar interactions
        similar = find_similar_interactions(drug, food)
        result['similar_interactions'] = similar
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats')
def get_stats():
    """Get dataset statistics"""
    if df.empty:
        return jsonify({'error': 'Dataset not loaded'})
    
    stats = {
        'total_interactions': len(df),
        'unique_drugs': df['drug'].nunique() if 'drug' in df.columns else 0,
        'unique_foods': df['food'].nunique() if 'food' in df.columns else 0,
        'risk_distribution': {}
    }
    
    if 'risk_level' in df.columns:
        stats['risk_distribution'] = df['risk_level'].value_counts().to_dict()
    
    return jsonify(stats)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Drug-Food Interaction Predictor")
    print("="*60)
    print(f"📊 Dataset: {len(df)} records loaded" if not df.empty else "❌ Dataset not loaded")
    print(f"🤖 Model: {'Loaded' if model else 'Not loaded'}")
    print(f"🌐 Starting server on http://localhost:5001")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5001, host='0.0.0.0')
