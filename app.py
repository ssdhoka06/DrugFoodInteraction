
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
# Commenting out PDF libraries that might not be installed
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Image
# from reportlab.lib import colors
import random
import logging
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for all routes
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
    os.path.join(DATA_DIR, 'balanced_drug_food_interactions.csv'),  # Relative path
    '/Users/sachidhoka/Desktop/ASEP_2(Drug-Food)/dfi project/data/balanced_drug_food_interactions.csv',  # Your actual path
    '/Users/sachidhoka/Desktop/balanced_drug_food_interactions.csv',  # Alternative path
    os.path.join('dfi project', 'data', 'balanced_drug_food_interactions.csv')  # Alternative relative path
]

# Load and cache data
df = None
all_drugs = []
all_foods = []
drug_categories = {}
food_categories = {}
high_risk_interactions = []

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
        
        # Create category mappings from your dataset - LIMIT THIS TO AVOID INFINITE LOOP
        logger.info("Processing categories (this may take a moment)...")
        for idx, row in df.head(10000).iterrows():  # LIMIT TO FIRST 10K ROWS
            drug = row['drug']
            food = row['food']
            drug_cat = row['drug_category'] if 'drug_category' in df.columns and pd.notna(row['drug_category']) else 'other'
            food_cat = row['food_category'] if 'food_category' in df.columns and pd.notna(row['food_category']) else 'other'
            
            if drug_cat not in drug_categories:
                drug_categories[drug_cat] = []
            if drug.lower() not in [d.lower() for d in drug_categories[drug_cat]] and len(drug_categories[drug_cat]) < 100:  # LIMIT LIST SIZE
                drug_categories[drug_cat].append(drug.lower())
                
            if food_cat not in food_categories:
                food_categories[food_cat] = []
            if food.lower() not in [f.lower() for f in food_categories[food_cat]] and len(food_categories[food_cat]) < 100:  # LIMIT LIST SIZE
                food_categories[food_cat].append(food.lower())
        
        # Extract high risk interactions from dataset - LIMIT THIS TOO
        if 'risk_level' in df.columns:
            high_risk_data = df[df['risk_level'] == 'HIGH'].head(100)  # LIMIT TO 100 HIGH RISK
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
        break  # Stop if we successfully loaded the data
        
    except FileNotFoundError:
        logger.warning(f"CSV file not found at {data_path}")
        continue
    except Exception as e:
        logger.error(f"Error loading data file at {data_path}: {str(e)}")
        continue

if df is None:
    logger.error("Could not load data from any of the specified paths. Using mock data.")
    # Add some mock data so the app can still run
    all_drugs = ['aspirin', 'warfarin', 'atorvastatin', 'metformin', 'lisinopril']
    all_foods = ['grapefruit', 'spinach', 'milk', 'coffee', 'alcohol']
    drug_categories = {'cardiovascular': ['aspirin', 'warfarin', 'atorvastatin', 'lisinopril'], 'diabetes': ['metformin']}
    food_categories = {'citrus': ['grapefruit'], 'leafy_greens': ['spinach'], 'dairy': ['milk'], 'beverages': ['coffee', 'alcohol']}
    high_risk_interactions = []

# Load ML model (commented out to avoid issues)
best_model = None
feature_info = {'feature_names': []}
scaler = None
logger.info("Using mock predictions (ML model not loaded)")

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
    
    if df is None:
        # Mock prediction when no data is available
        return {
            'drug': drug,
            'food': food,
            'interaction_predicted': True,
            'probability': 0.6,
            'risk_level': 'MODERATE',
            'mechanism': 'unknown',
            'explanation': f"Mock prediction for {drug} and {food}. Real data not loaded.",
            'recommendations': "This is a test prediction. Consult healthcare provider for real advice.",
            'known_interaction': False
        }
    
    # First, check your dataset for exact matches
    drug_lower = drug.lower().strip()
    food_lower = food.lower().strip()
    
    # Look for exact match in dataset
    dataset_match = df[
        (df['drug'].str.lower().str.strip() == drug_lower) & 
        (df['food'].str.lower().str.strip() == food_lower)
    ]
    
    if not dataset_match.empty:
        # Use data from your dataset
        row = dataset_match.iloc[0]
        interaction = bool(row['interaction']) if 'interaction' in df.columns and pd.notna(row['interaction']) else False
        
        # Convert risk level to probability
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
    return jsonify({
        'status': 'running',
        'message': 'Drug-Food Interaction API is running!',
        'data_loaded': df is not None,
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
        
        return jsonify({'success': True, 'data': prediction})
    except Exception as e:
        logger.error(f"Error in predict_interaction: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/drugs')
def get_drugs():
    query = request.args.get('q', '').lower()
    limit = int(request.args.get('limit', 20))
    
    if query:
        matching_drugs = [drug for drug in all_drugs if query in drug.lower()][:limit]
    else:
        matching_drugs = all_drugs[:limit]
    
    return jsonify({'drugs': matching_drugs})

@app.route('/api/foods')
def get_foods():
    query = request.args.get('q', '').lower()
    limit = int(request.args.get('limit', 20))
    
    if query:
        matching_foods = [food for food in all_foods if query in food.lower()][:limit]
    else:
        matching_foods = all_foods[:limit]
    
    return jsonify({'foods': matching_foods})

@app.route('/api/test')
def test():
    return jsonify({'message': 'API is working', 'timestamp': datetime.now().isoformat()})

# Commented out the research articles update function that was causing the hang
def update_research_articles():
    """Update research articles from news API"""
    try:
        # Add timeout and better error handling
        response = requests.get(
            'https://newsapi.org/v2/everything',
            params={
                'q': 'drug food interaction OR pharmacokinetics OR drug metabolism',
                'apiKey': os.environ.get('NEWS_API_KEY'),
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20
            },
            timeout=10  # Add timeout
        )
        
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            logger.info(f"Successfully fetched {len(articles)} articles")
            # Process articles here...
        else:
            logger.warning(f"NewsAPI returned status code: {response.status_code}")
    except requests.exceptions.Timeout:
        logger.error("NewsAPI request timed out")
    except Exception as e:
        logger.error(f"Error updating research articles: {str(e)}")

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    
    # COMMENTED OUT THE BLOCKING CALL THAT WAS CAUSING THE HANG
    # update_research_articles()
    
    logger.info("Flask app ready to start...")
    app.run(debug=True, host='0.0.0.0', port=5001)
