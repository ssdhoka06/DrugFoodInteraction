from flask import Flask, request, jsonify, render_template, session, send_file, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import json
import os
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import requests
from io import BytesIO
import random
import logging
from functools import wraps
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, origins=["http://localhost:5001", "http://127.0.0.1:5001", "*"])
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database configuration
DATABASE = 'dfi_predictor.db'

def get_db():
    """Get a database connection"""
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    """Initialize the database with required tables"""
    with get_db() as db:
        # Users table
        db.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT NOT NULL,
                      email TEXT UNIQUE NOT NULL,
                      password TEXT NOT NULL,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
        # Predictions table
        db.execute('''CREATE TABLE IF NOT EXISTS predictions
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER,
                      drug TEXT NOT NULL,
                      food TEXT NOT NULL,
                      interaction_predicted BOOLEAN,
                      probability REAL,
                      risk_level TEXT,
                      mechanism TEXT,
                      explanation TEXT,
                      recommendations TEXT,
                      timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (user_id) REFERENCES users (id))''')
        
        # Saved medications table
        db.execute('''CREATE TABLE IF NOT EXISTS user_medications
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER,
                      medication_name TEXT NOT NULL,
                      dosage TEXT,
                      frequency TEXT,
                      notes TEXT,
                      timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (user_id) REFERENCES users (id))''')
        
        # Research articles table (cached from API)
        db.execute('''CREATE TABLE IF NOT EXISTS research_articles
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      title TEXT NOT NULL,
                      source TEXT,
                      author TEXT,
                      description TEXT,
                      url TEXT UNIQUE,
                      image_url TEXT,
                      published_at TEXT,
                      category TEXT)''')

# Initialize database
init_db()

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id, name, email):
        self.id = id
        self.name = name
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    with get_db() as db:
        user = db.execute('SELECT id, name, email FROM users WHERE id = ?', (user_id,)).fetchone()
        return User(user['id'], user['name'], user['email']) if user else None

# API response decorator
def json_response(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return jsonify({
                'success': True,
                'data': result
            })
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    return wrapper

# Define paths to data files
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Try multiple possible paths for the data file
DATA_PATHS = [
    os.path.join(DATA_DIR, 'balanced_drug_food_interactions.csv'),
    os.path.join('dfi project', 'data', 'balanced_drug_food_interactions.csv'),
    'data/balanced_drug_food_interactions.csv',
    'balanced_drug_food_interactions.csv'
]

# Try multiple possible paths for the ML model
MODEL_PATHS = [
    os.path.join(MODELS_DIR, 'best_drug_food_interaction_model.pkl'),
    os.path.join('dfi project', 'models', 'best_drug_food_interaction_model.pkl'),
    'models/best_drug_food_interaction_model.pkl',
    'best_drug_food_interaction_model.pkl'
]

# Load and cache data
df = None
all_drugs = []
all_foods = []
drug_categories = {}
food_categories = {}
high_risk_interactions = []

# Load dataset
for data_path in DATA_PATHS:
    try:
        logger.info(f"Trying to load data from: {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully from {data_path}")
        
        # Clean and process data
        df = df.dropna(subset=['drug', 'food'])
        df['drug'] = df['drug'].str.strip().str.lower()
        df['food'] = df['food'].str.strip().str.lower()
        
        # Get unique drugs and foods
        all_drugs = sorted(df['drug'].unique())
        all_foods = sorted(df['food'].unique())
        
        # Extract categories from dataset
        drug_categories = {}
        food_categories = {}
        
        # Create category mappings from your dataset
        logger.info("Processing categories...")
        for idx, row in df.head(10000).iterrows():
            drug = row['drug']
            food = row['food']
            drug_cat = row['drug_category'] if 'drug_category' in df.columns and pd.notna(row['drug_category']) else 'other'
            food_cat = row['food_category'] if 'food_category' in df.columns and pd.notna(row['food_category']) else 'other'
            
            if drug_cat not in drug_categories:
                drug_categories[drug_cat] = []
            if drug.lower() not in [d.lower() for d in drug_categories[drug_cat]] and len(drug_categories[drug_cat]) < 100:
                drug_categories[drug_cat].append(drug.lower())
                
            if food_cat not in food_categories:
                food_categories[food_cat] = []
            if food.lower() not in [f.lower() for f in food_categories[food_cat]] and len(food_categories[food_cat]) < 100:
                food_categories[food_cat].append(food.lower())
        
        # Extract high risk interactions from dataset
        if 'risk_level' in df.columns:
            high_risk_data = df[df['risk_level'] == 'HIGH'].head(100)
            for _, row in high_risk_data.iterrows():
                high_risk_interactions.append({
                    'drug': row['drug'],
                    'food': row['food'],
                    'mechanism': row['mechanism'] if 'mechanism' in df.columns and pd.notna(row['mechanism']) else 'unknown',
                    'explanation': f"High risk interaction between {row['drug']} and {row['food']}",
                    'recommendations': "Avoid this combination as it may cause serious health risks",
                    'risk_level': 'HIGH'
                })
        
        logger.info(f"Data processed successfully: {len(all_drugs)} drugs, {len(all_foods)} foods")
        break
        
    except FileNotFoundError:
        logger.warning(f"CSV file not found at {data_path}")
        continue
    except Exception as e:
        logger.error(f"Error loading data file at {data_path}: {str(e)}")
        continue

if df is None:
    logger.error("Could not load data from any of the specified paths. Using mock data.")
    all_drugs = ['aspirin', 'warfarin', 'atorvastatin', 'metformin', 'lisinopril']
    all_foods = ['grapefruit', 'spinach', 'milk', 'coffee', 'alcohol']
    drug_categories = {'cardiovascular': ['aspirin', 'warfarin', 'atorvastatin', 'lisinopril'], 'diabetes': ['metformin']}
    food_categories = {'citrus': ['grapefruit'], 'leafy_greens': ['spinach'], 'dairy': ['milk'], 'beverages': ['coffee', 'alcohol']}
    high_risk_interactions = []

# Load ML model
best_model = None
feature_info = {'feature_names': []}
scaler = None

for model_path in MODEL_PATHS:
    try:
        logger.info(f"Trying to load ML model from: {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Handle different pickle formats
        if isinstance(model_data, dict):
            best_model = model_data.get('model')
            feature_info = model_data.get('feature_info', {'feature_names': []})
            scaler = model_data.get('scaler')
        else:
            best_model = model_data
            
        logger.info(f"ML model loaded successfully from {model_path}")
        logger.info(f"Model type: {type(best_model)}")
        
        if hasattr(best_model, 'feature_names_in_'):
            feature_info['feature_names'] = list(best_model.feature_names_in_)
        elif 'feature_names' in feature_info:
            logger.info(f"Feature names: {len(feature_info['feature_names'])} features")
        
        break
        
    except FileNotFoundError:
        logger.warning(f"Model file not found at {model_path}")
        continue
    except Exception as e:
        logger.error(f"Error loading model file at {model_path}: {str(e)}")
        continue

if best_model is None:
    logger.error("Could not load ML model from any of the specified paths. Using mock predictions.")

def create_feature_vector(drug, food):
    """Create feature vector for ML model prediction"""
    if df is None or best_model is None:
        return np.array([])
    
    try:
        # Basic features
        features = []
        
        # Drug and food name features (simple encoding)
        drug_lower = drug.lower().strip()
        food_lower = food.lower().strip()
        
        # One-hot encoding for common drugs and foods
        common_drugs = all_drugs[:50] if len(all_drugs) > 50 else all_drugs
        common_foods = all_foods[:50] if len(all_foods) > 50 else all_foods
        
        # Drug features
        for d in common_drugs:
            features.append(1 if d == drug_lower else 0)
        
        # Food features
        for f in common_foods:
            features.append(1 if f == food_lower else 0)
        
        # Category features
        drug_cat = categorize_entity(drug, drug_categories)
        food_cat = categorize_entity(food, food_categories)
        
        all_drug_cats = list(drug_categories.keys())
        all_food_cats = list(food_categories.keys())
        
        # Drug category features
        for cat in all_drug_cats:
            features.append(1 if cat == drug_cat else 0)
        
        # Food category features
        for cat in all_food_cats:
            features.append(1 if cat == food_cat else 0)
        
        # Additional features
        features.extend([
            len(drug),  # Drug name length
            len(food),  # Food name length
            1 if any(keyword in drug_lower for keyword in ['anti', 'inhibitor', 'blocker']) else 0,
            1 if any(keyword in food_lower for keyword in ['citrus', 'dairy', 'leafy']) else 0
        ])
        
        feature_array = np.array(features).reshape(1, -1)
        
        # Apply scaler if available
        if scaler is not None:
            feature_array = scaler.transform(feature_array)
        
        return feature_array
        
    except Exception as e:
        logger.error(f"Error creating feature vector: {str(e)}")
        return np.array([])

def categorize_entity(name, categories):
    """Categorize a drug or food based on name"""
    name_lower = name.lower()
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in name_lower:
                return category
    return 'other'

def get_interaction_explanation(mechanism, drug_category, food_category):
    """Get explanation for interaction mechanism"""
    explanations = {
        'cyp3a4_inhibition': (
            f"Components in {food_category} inhibit CYP3A4 enzymes in the liver, which are "
            f"responsible for metabolizing {drug_category} medications. This can lead to "
            "increased drug levels in the blood."
        ),
        'vitamin_k_competition': (
            f"{food_category} contains vitamin K which opposes the blood-thinning effects "
            f"of {drug_category} medications by supporting clotting factor production."
        ),
        'calcium_chelation': (
            f"Calcium in {food_category} binds to {drug_category}, forming insoluble "
            "complexes that reduce drug absorption in the gastrointestinal tract."
        ),
        'absorption_interference': (
            f"{food_category} may delay or reduce the absorption of {drug_category} by "
            "affecting gastric emptying or forming physical barriers in the GI tract."
        )
    }
    return explanations.get(mechanism, "Potential interaction through an unspecified mechanism.")

def get_recommendations(mechanism, risk_level):
    """Get recommendations based on mechanism and risk level"""
    recommendations = {
        'HIGH': {
            'cyp3a4_inhibition': 'Avoid consuming these together. Consider alternative medications or foods.',
            'vitamin_k_competition': 'Maintain consistent vitamin K intake. Monitor INR closely and adjust warfarin dose as needed.',
            'default': 'Avoid combination. Consult healthcare provider for alternatives.'
        },
        'MODERATE': {
            'calcium_chelation': 'Take medication 2 hours before or 4 hours after consuming calcium-rich foods.',
            'absorption_interference': 'Take medication on an empty stomach if possible, or at consistent times relative to meals.',
            'default': 'Space administration times. Monitor for reduced efficacy or side effects.'
        },
        'LOW': {
            'default': 'Minimal clinical significance. No special precautions required for most patients.'
        }
    }
    
    risk_recs = recommendations.get(risk_level, {})
    return risk_recs.get(mechanism, risk_recs.get('default', 'Monitor for any unexpected effects.'))

def predict_new_interaction_with_explanation(drug, food, model=None):
    """Predict interaction between drug and food with explanation"""
    
    # First, check dataset for exact matches
    drug_lower = drug.lower().strip()
    food_lower = food.lower().strip()
    
    if df is not None:
        dataset_match = df[
            (df['drug'].str.lower().str.strip() == drug_lower) & 
            (df['food'].str.lower().str.strip() == food_lower)
        ]
        
        if not dataset_match.empty:
            # Use data from dataset
            row = dataset_match.iloc[0]
            interaction = bool(row['interaction']) if 'interaction' in df.columns and pd.notna(row['interaction']) else False
            
            risk_level = row['risk_level'] if 'risk_level' in df.columns and pd.notna(row['risk_level']) else 'LOW'
            if risk_level == 'HIGH':
                probability = 0.9
            elif risk_level == 'MODERATE':
                probability = 0.7
            else:
                probability = 0.3
                
            mechanism = row['mechanism'] if 'mechanism' in df.columns and pd.notna(row['mechanism']) else 'unknown'
            drug_category = row['drug_category'] if 'drug_category' in df.columns and pd.notna(row['drug_category']) else 'other'
            food_category = row['food_category'] if 'food_category' in df.columns and pd.notna(row['food_category']) else 'other'
            
            explanation = get_interaction_explanation(mechanism, drug_category, food_category)
            recommendations = get_recommendations(mechanism, risk_level)
            
            return {
                'drug': drug,
                'food': food,
                'interaction_predicted': interaction,
                'probability': probability,
                'risk_level': risk_level,
                'mechanism': mechanism,
                'explanation': explanation,
                'recommendations': recommendations,
                'known_interaction': True
            }
    
    # Use ML model for prediction if available
    if best_model is not None:
        try:
            feature_vector = create_feature_vector(drug, food)
            
            if feature_vector.size > 0:
                # Make prediction
                if hasattr(best_model, 'predict_proba'):
                    probabilities = best_model.predict_proba(feature_vector)[0]
                    interaction_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
                else:
                    prediction = best_model.predict(feature_vector)[0]
                    interaction_prob = prediction if isinstance(prediction, float) else 0.5
                
                interaction_predicted = interaction_prob > 0.5
                
                # Determine risk level based on probability
                if interaction_prob > 0.8:
                    risk_level = 'HIGH'
                elif interaction_prob > 0.6:
                    risk_level = 'MODERATE'
                else:
                    risk_level = 'LOW'
                
                # Categorize for explanation
                drug_category = categorize_entity(drug, drug_categories)
                food_category = categorize_entity(food, food_categories)
                
                # Determine likely mechanism based on categories
                if 'citrus' in food_category.lower() or 'grapefruit' in food_lower:
                    mechanism = 'cyp3a4_inhibition'
                elif 'leafy' in food_category.lower() or 'vitamin' in food_lower:
                    mechanism = 'vitamin_k_competition'
                elif 'dairy' in food_category.lower() or 'calcium' in food_lower:
                    mechanism = 'calcium_chelation'
                else:
                    mechanism = 'absorption_interference'
                
                explanation = get_interaction_explanation(mechanism, drug_category, food_category)
                recommendations = get_recommendations(mechanism, risk_level)
                
                return {
                    'drug': drug,
                    'food': food,
                    'interaction_predicted': interaction_predicted,
                    'probability': round(interaction_prob, 3),
                    'risk_level': risk_level,
                    'mechanism': mechanism,
                    'explanation': explanation,
                    'recommendations': recommendations,
                    'known_interaction': False
                }
        except Exception as e:
            logger.error(f"Error in ML model prediction: {str(e)}")
    
    # Fallback: no interaction found
    return {
        'drug': drug,
        'food': food,
        'interaction_predicted': False,
        'probability': 0.1,
        'risk_level': 'LOW',
        'mechanism': 'unknown',
        'explanation': f"No known interaction between {drug} and {food} found in our database.",
        'recommendations': "Monitor for any unexpected effects. Consult healthcare provider if concerned.",
        'known_interaction': False
    }

def get_similar_interactions(drug, food):
    """Get similar known interactions for display"""
    drug_category = categorize_entity(drug, drug_categories)  
    food_category = categorize_entity(food, food_categories)
    
    similar = []
    for interaction in high_risk_interactions:
        if (categorize_entity(interaction['drug'], drug_categories) == drug_category or
            categorize_entity(interaction['food'], food_categories) == food_category):
            similar.append({
                'drug': interaction['drug'],
                'food': interaction['food'],
                'risk_level': interaction['risk_level']
            })
            if len(similar) >= 3:
                break
    
    return similar

# Routes
@app.route('/')
def home():
    """Serve the main frontend page"""
    return render_template('index.html')

@app.route('/api/status')
def status():
    """API status endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'Drug-Food Interaction API is running!',
        'data_loaded': df is not None,
        'model_loaded': best_model is not None,
        'drugs_count': len(all_drugs),
        'foods_count': len(all_foods),
        'sample_drugs': all_drugs[:10],
        'sample_foods': all_foods[:10]
    })

@app.route('/api/predict', methods=['POST'])
def predict_interaction():
    try:
        data = request.get_json()
        drug = data.get('drug')
        food = data.get('food')
        
        if not drug or not food:
            return jsonify({'success': False, 'error': 'Both drug and food are required'}), 400
        
        # Get prediction
        prediction = predict_new_interaction_with_explanation(drug, food, best_model)
        
        # Get similar interactions for the frontend
        similar_interactions = get_similar_interactions(drug, food)
        prediction['similar_interactions'] = similar_interactions
        
        # Save prediction to database if user is logged in
        if current_user.is_authenticated:
            try:
                with get_db() as db:
                    db.execute('''INSERT INTO predictions 
                                 (user_id, drug, food, interaction_predicted, probability, 
                                  risk_level, mechanism, explanation, recommendations)
                                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                              (current_user.id, drug, food, prediction['interaction_predicted'],
                               prediction['probability'], prediction['risk_level'],
                               prediction['mechanism'], prediction['explanation'],
                               prediction['recommendations']))
            except Exception as e:
                logger.error(f"Error saving prediction to database: {str(e)}")
        
        return jsonify({'success': True, 'data': prediction})
    except Exception as e:
        logger.error(f"Error in predict_interaction: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/drugs')
def get_drugs():
    query = request.args.get('q', '').lower().strip()
    limit = int(request.args.get('limit', 20))
    
    try:
        if query:
            # Filter drugs that contain the query
            matching_drugs = [drug for drug in all_drugs if query in drug.lower()][:limit]
        else:
            # Return first N drugs if no query
            matching_drugs = all_drugs[:limit]
        
        # Ensure we have data to return
        if not matching_drugs and len(all_drugs) > 0:
            matching_drugs = all_drugs[:5]  # Return at least some drugs
        
        return jsonify({
            'success': True,
            'drugs': matching_drugs,
            'total': len(matching_drugs)
        })
    except Exception as e:
        logger.error(f"Error in get_drugs: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'drugs': []
        }), 500

@app.route('/api/foods')
def get_foods():
    query = request.args.get('q', '').lower().strip()
    limit = int(request.args.get('limit', 20))
    
    try:
        if query:
            # Filter foods that contain the query
            matching_foods = [food for food in all_foods if query in food.lower()][:limit]
        else:
            # Return first N foods if no query
            matching_foods = all_foods[:limit]
        
        # Ensure we have data to return
        if not matching_foods and len(all_foods) > 0:
            matching_foods = all_foods[:5]  # Return at least some foods
        
        return jsonify({
            'success': True,
            'foods': matching_foods,
            'total': len(matching_foods)
        })
    except Exception as e:
        logger.error(f"Error in get_foods: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'foods': []
        }), 500
    
@app.route('/api/debug')
def debug_data():
    return jsonify({
        'data_loaded': df is not None,
        'model_loaded': best_model is not None,
        'drugs_count': len(all_drugs),
        'foods_count': len(all_foods),
        'sample_drugs': all_drugs[:5] if all_drugs else [],
        'sample_foods': all_foods[:5] if all_foods else [],
        'drug_categories': list(drug_categories.keys())[:5] if drug_categories else [],
        'food_categories': list(food_categories.keys())[:5] if food_categories else []
    })

@app.errorhandler(404)
def not_found(error):
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'API endpoint not found'}), 404
    return render_template('index.html')

@app.route('/api/test')
def test():
    return jsonify({'message': 'API is working', 'timestamp': datetime.now().isoformat()})

# Authentication routes
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        
        if not name or not email or not password:
            return jsonify({'success': False, 'error': 'All fields are required'}), 400
        
        password_hash = generate_password_hash(password)
        
        with get_db() as db:
            try:
                db.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                          (name, email, password_hash))
                return jsonify({'success': True, 'message': 'User registered successfully'})
            except sqlite3.IntegrityError:
                return jsonify({'success': False, 'error': 'Email already exists'}), 400
                
    except Exception as e:
        logger.error(f"Error in register: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'success': False, 'error': 'Email and password are required'}), 400
        
        with get_db() as db:
            user = db.execute('SELECT id, name, email, password FROM users WHERE email = ?', 
                            (email,)).fetchone()
            
            if user and check_password_hash(user['password'], password):
                user_obj = User(user['id'], user['name'], user['email'])
                login_user(user_obj)
                return jsonify({'success': True, 'user': {'id': user['id'], 'name': user['name'], 'email': user['email']}})
            else:
                return jsonify({'success': False, 'error': 'Invalid email or password'}), 401
                
    except Exception as e:
        logger.error(f"Error in login: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/user')
@login_required
def get_user():
    return jsonify({
        'success': True, 
        'user': {
            'id': current_user.id, 
            'name': current_user.name, 
            'email': current_user.email
        }
    })

@app.route('/api/predictions')
@login_required
def get_user_predictions():
    try:
        with get_db() as db:
            predictions = db.execute('''SELECT * FROM predictions 
                                      WHERE user_id = ? 
                                      ORDER BY timestamp DESC LIMIT 50''', 
                                   (current_user.id,)).fetchall()
            
            predictions_list = []
            for pred in predictions:
                predictions_list.append({
                    'id': pred['id'],
                    'drug': pred['drug'],
                    'food': pred['food'],
                    'interaction_predicted': pred['interaction_predicted'],
                    'probability': pred['probability'],
                    'risk_level': pred['risk_level'],
                    'mechanism': pred['mechanism'],
                    'explanation': pred['explanation'],
                    'recommendations': pred['recommendations'],
                    'timestamp': pred['timestamp']
                })
            
            return jsonify({'success': True, 'predictions': predictions_list})
    except Exception as e:
        logger.error(f"Error getting user predictions: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    logger.info(f"Data loaded: {df is not None}")
    logger.info(f"Model loaded: {best_model is not None}")
    logger.info(f"Available drugs: {len(all_drugs)}")
    logger.info(f"Available foods: {len(all_foods)}")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
