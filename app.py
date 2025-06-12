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
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Image
from reportlab.lib import colors
import random
import logging
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
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

# Load and cache data
try:
    df = pd.read_csv("data/balanced_drug_food_interactions.csv")
    all_drugs = sorted(set(df['drug'].dropna().str.strip()))
    all_foods = sorted(set(df['food'].dropna().str.strip()))
    
    # Load drug and food categories
    with open('data/drug_categories.json') as f:
        drug_categories = json.load(f)
    
    with open('data/food_categories.json') as f:
        food_categories = json.load(f)
    
    # Load high risk interactions
    with open('data/high_risk_interactions.json') as f:
        high_risk_interactions = json.load(f)
except Exception as e:
    logger.error(f"Error loading data files: {str(e)}")
    all_drugs = []
    all_foods = []
    drug_categories = {}
    food_categories = {}
    high_risk_interactions = {}

# Load ML model
try:
    with open('models/best_drug_food_interaction_model.pkl', 'rb') as f:
        model_package = pickle.load(f)
    best_model = model_package['model']
    feature_info = model_package['feature_info']
    scaler = model_package['scaler']
    logger.info("ML model loaded successfully")
except Exception as e:
    logger.error(f"Error loading ML model: {str(e)}")
    best_model = None
    feature_info = {'feature_names': []}
    scaler = None

# Mock ML functions for demonstration
def predict_new_interaction_with_explanation(drug, food, model=None):
    """Predict interaction between drug and food with explanation"""
    # Check for known high-risk interactions first
    drug_lower = drug.lower()
    food_lower = food.lower()
    
    for interaction in high_risk_interactions:
        if (drug_lower in interaction['drug'].lower() and 
            food_lower in interaction['food'].lower()):
            return {
                'drug': drug,
                'food': food,
                'interaction_predicted': True,
                'probability': 0.95,
                'risk_level': 'HIGH',
                'mechanism': interaction['mechanism'],
                'explanation': interaction['explanation'],
                'recommendations': interaction['recommendations'],
                'known_interaction': True
            }
    
    # For demo purposes, generate mock predictions
    # In a real app, this would use the actual ML model
    risk_levels = ['HIGH', 'MODERATE', 'LOW']
    mechanisms = [
        'cyp3a4_inhibition',
        'vitamin_k_competition',
        'calcium_chelation',
        'absorption_interference'
    ]
    
    # Categorize drug and food
    drug_category = categorize_entity(drug, drug_categories)
    food_category = categorize_entity(food, food_categories)
    
    # Determine interaction probability based on categories
    interaction_prob = 0.1  # Base probability
    
    # Increase probability for certain category combinations
    if drug_category == 'anticoagulant' and food_category == 'leafy_greens':
        interaction_prob = 0.9
        mechanism = 'vitamin_k_competition'
    elif drug_category == 'statin' and food_category == 'citrus':
        interaction_prob = 0.85
        mechanism = 'cyp3a4_inhibition'
    elif drug_category == 'antibiotic' and food_category == 'dairy':
        interaction_prob = 0.7
        mechanism = 'calcium_chelation'
    else:
        # Random mechanism for demo
        mechanism = random.choice(mechanisms)
        interaction_prob = min(0.9, max(0.1, interaction_prob + random.uniform(-0.2, 0.3)))
    
    interaction = interaction_prob > 0.5
    risk_level = 'HIGH' if interaction_prob > 0.8 else 'MODERATE' if interaction_prob > 0.5 else 'LOW'
    
    explanation = get_interaction_explanation(mechanism, drug_category, food_category)
    recommendations = get_recommendations(mechanism, risk_level)
    
    return {
        'drug': drug,
        'food': food,
        'interaction_predicted': interaction,
        'probability': interaction_prob,
        'risk_level': risk_level,
        'mechanism': mechanism,
        'explanation': explanation,
        'recommendations': recommendations,
        'known_interaction': False
    }

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

def check_meal_plan_compatibility(medications, meal_plan):
    """Check compatibility of medications with meal plan"""
    interactions = []
    
    for med in medications:
        for food_item in meal_plan:
            # Check if this is a known high-risk combination
            med_lower = med.lower()
            food_lower = food_item.lower()
            
            for interaction in high_risk_interactions:
                if (med_lower in interaction['drug'].lower() and 
                    food_lower in interaction['food'].lower()):
                    interactions.append({
                        'drug': med,
                        'food': food_item,
                        'risk_level': 'HIGH',
                        'mechanism': interaction['mechanism'],
                        'recommendation': interaction['recommendations']
                    })
                    break
    
    return {
        'interactions_found': len(interactions) > 0,
        'interactions': interactions,
        'summary': f"Found {len(interactions)} potential interactions in your meal plan."
    }

# Routes
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/api/search/drugs')
@json_response
def search_drugs():
    query = request.args.get('q', '').lower().strip()
    if not query or len(query) < 2:
        return {'results': []}
    
    results = []
    for drug in all_drugs:
        if query in drug.lower():
            category = categorize_entity(drug, drug_categories)
            results.append({
                'id': drug,
                'text': drug,
                'category': category.replace('_', ' ').title()
            })
            if len(results) >= 20:
                break
    
    return {'results': results}

@app.route('/api/search/foods')
@json_response
def search_foods():
    query = request.args.get('q', '').lower().strip()
    if not query or len(query) < 2:
        return {'results': []}
    
    results = []
    for food in all_foods:
        if query in food.lower():
            category = categorize_entity(food, food_categories)
            results.append({
                'id': food,
                'text': food,
                'category': category.replace('_', ' ').title()
            })
            if len(results) >= 20:
                break
    
    return {'results': results}

@app.route('/api/predict', methods=['POST'])
@login_required
@json_response
def predict_interaction():
    data = request.get_json()
    drug = data.get('drug')
    food = data.get('food')
    
    if not drug or not food:
        raise ValueError('Both drug and food are required')
    
    # Get prediction
    prediction = predict_new_interaction_with_explanation(drug, food, best_model)
    
    # Save to database
    with get_db() as db:
        db.execute('''INSERT INTO predictions 
                     (user_id, drug, food, interaction_predicted, probability, 
                      risk_level, mechanism, explanation, recommendations)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (current_user.id, drug, food, prediction['interaction_predicted'], 
                   prediction['probability'], prediction['risk_level'], 
                   prediction['mechanism'], prediction['explanation'], 
                   prediction['recommendations']))
    
    # Get similar interactions for the frontend
    similar_interactions = get_similar_interactions(drug, food)
    prediction['similar_interactions'] = similar_interactions
    
    return prediction

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

@app.route('/api/auth/login', methods=['POST'])
@json_response
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        raise ValueError('Email and password are required')
    
    with get_db() as db:
        user = db.execute(
            'SELECT id, name, email, password FROM users WHERE email = ?', 
            (email,)
        ).fetchone()
        
        if not user or not check_password_hash(user['password'], password):
            raise ValueError('Invalid email or password')
        
        login_user(User(user['id'], user['name'], user['email']))
    
    return {'message': 'Login successful'}

@app.route('/api/auth/signup', methods=['POST'])
@json_response
def signup():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    
    if not name or not email or not password:
        raise ValueError('Name, email and password are required')
    
    if len(password) < 8:
        raise ValueError('Password must be at least 8 characters')
    
    hashed_password = generate_password_hash(password)
    
    try:
        with get_db() as db:
            cursor = db.execute(
                'INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                (name, email, hashed_password)
            )
            user_id = cursor.lastrowid
            db.commit()
        
        # Log in the new user
        login_user(User(user_id, name, email))
        return {'message': 'Account created successfully'}
    except sqlite3.IntegrityError:
        raise ValueError('Email already exists')

@app.route('/api/auth/logout', methods=['POST'])
@login_required
@json_response
def logout():
    logout_user()
    return {'message': 'Logout successful'}

@app.route('/api/user/medications', methods=['GET', 'POST', 'DELETE'])
@login_required
@json_response
def user_medications():
    if request.method == 'GET':
        # Get all medications for user
        with get_db() as db:
            medications = db.execute(
                'SELECT id, medication_name, dosage, frequency, notes FROM user_medications '
                'WHERE user_id = ? ORDER BY medication_name',
                (current_user.id,)
            ).fetchall()
            return {'medications': [dict(med) for med in medications]}
    
    elif request.method == 'POST':
        # Add new medication
        data = request.get_json()
        name = data.get('name')
        
        if not name:
            raise ValueError('Medication name is required')
        
        with get_db() as db:
            db.execute(
                'INSERT INTO user_medications (user_id, medication_name, dosage, frequency, notes) '
                'VALUES (?, ?, ?, ?, ?)',
                (current_user.id, name, data.get('dosage'), data.get('frequency'), data.get('notes')))
            db.commit()
            return {'message': 'Medication added successfully'}
    
    elif request.method == 'DELETE':
        # Remove medication
        med_id = request.args.get('id')
        if not med_id:
            raise ValueError('Medication ID is required')
        
        with get_db() as db:
            db.execute(
                'DELETE FROM user_medications WHERE id = ? AND user_id = ?',
                (med_id, current_user.id)
            )
            affected = db.total_changes
            db.commit()
            
            if affected == 0:
                raise ValueError('Medication not found or not owned by user')
            
            return {'message': 'Medication removed successfully'}

@app.route('/api/user/history', methods=['GET'])
@login_required
@json_response
def prediction_history():
    limit = request.args.get('limit', 10)
    
    with get_db() as db:
        history = db.execute(
            'SELECT id, drug, food, risk_level, timestamp FROM predictions '
            'WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?',
            (current_user.id, limit)
        ).fetchall()
        
        return {'history': [dict(item) for item in history]}

@app.route('/api/research/articles', methods=['GET'])
@json_response
def research_articles():
    category = request.args.get('category', 'all')
    search_query = request.args.get('q', '')
    page = int(request.args.get('page', 1))
    per_page = 6
    
    # Base query
    query = 'SELECT * FROM research_articles WHERE 1=1'
    params = []
    
    # Apply filters
    if category != 'all':
        query += ' AND category = ?'
        params.append(category)
    
    if search_query:
        query += ' AND (title LIKE ? OR description LIKE ?)'
        params.extend([f'%{search_query}%', f'%{search_query}%'])
    
    # Add pagination
    query += ' ORDER BY published_at DESC LIMIT ? OFFSET ?'
    params.extend([per_page, (page - 1) * per_page])
    
    with get_db() as db:
        articles = db.execute(query, params).fetchall()
        total = db.execute('SELECT COUNT(*) FROM research_articles').fetchone()[0]
        
        return {
            'articles': [dict(article) for article in articles],
            'total': total,
            'page': page,
            'per_page': per_page
        }

@app.route('/api/chat', methods=['POST'])
@json_response
def chat():
    data = request.get_json()
    message = data.get('message', '').strip()
    
    if not message:
        raise ValueError('Message is required')
    
    # Simple rule-based responses (in a real app, integrate with a proper NLP system)
    responses = {
        'warfarin': {
            'message': (
                "Warfarin interacts with vitamin K-rich foods like leafy greens. "
                "Maintain consistent vitamin K intake and monitor your INR regularly."
            ),
            'sources': [
                {'title': 'Warfarin and Diet', 'url': 'https://example.com/warfarin-diet'}
            ]
        },
        'grapefruit': {
            'message': (
                "Grapefruit can interact with many medications by inhibiting CYP3A4 enzymes. "
                "Avoid grapefruit with statins, some blood pressure medications, and others."
            ),
            'sources': [
                {'title': 'Grapefruit-Drug Interactions', 'url': 'https://example.com/grapefruit-interactions'}
            ]
        },
        'dairy': {
            'message': (
                "Dairy products can interfere with absorption of some antibiotics like tetracyclines. "
                "Take these medications 2 hours before or 4 hours after dairy."
            )
        },
        'default': {
            'message': (
                "I can help answer questions about drug-food interactions. "
                "Try asking about specific medications or foods like warfarin, grapefruit, or dairy."
            )
        }
    }
    
    # Check for keywords
    message_lower = message.lower()
    response = None
    
    if 'warfarin' in message_lower or 'blood thinner' in message_lower:
        response = responses['warfarin']
    elif 'grapefruit' in message_lower:
        response = responses['grapefruit']
    elif 'dairy' in message_lower or 'milk' in message_lower:
        response = responses['dairy']
    else:
        response = responses['default']
    
    return {
        'reply': response['message'],
        'sources': response.get('sources', [])
    }

@app.route('/api/report/generate', methods=['POST'])
@login_required
@json_response
def generate_report():
    data = request.get_json()
    prediction_id = data.get('prediction_id')
    
    if not prediction_id:
        raise ValueError('Prediction ID is required')
    
    # Get prediction data
    with get_db() as db:
        prediction = db.execute(
            'SELECT * FROM predictions WHERE id = ? AND user_id = ?',
            (prediction_id, current_user.id)
        ).fetchone()
        
        if not prediction:
            raise ValueError('Prediction not found')
        
        prediction = dict(prediction)
    
    # Generate PDF
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading2']
    body_style = styles['BodyText']
    
    # Custom styles
    risk_style = ParagraphStyle(
        'RiskStyle',
        parent=body_style,
        textColor=colors.red if prediction['risk_level'] == 'HIGH' else 
                colors.orange if prediction['risk_level'] == 'MODERATE' else 
                colors.green,
        fontSize=14,
        spaceAfter=12
    )
    
    # Content
    content = []
    
    # Title
    content.append(Paragraph("Drug-Food Interaction Report", title_style))
    content.append(Spacer(1, 12))
    
    # Date
    content.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%B %d, %Y')}", 
        body_style
    ))
    content.append(Spacer(1, 24))
    
    # Interaction pair
    content.append(Paragraph(
        f"<b>Medication:</b> {prediction['drug']}<br/>"
        f"<b>Food/Supplement:</b> {prediction['food']}",
        body_style
    ))
    content.append(Spacer(1, 12))
    
    # Risk level
    content.append(Paragraph(
        f"<b>Risk Level:</b> {prediction['risk_level']}",
        risk_style
    ))
    content.append(Spacer(1, 12))
    
    # Probability
    content.append(Paragraph(
        f"<b>Probability of Interaction:</b> {prediction['probability']*100:.0f}%",
        body_style
    ))
    content.append(Spacer(1, 24))
    
    # Mechanism
    content.append(Paragraph("Interaction Mechanism", heading_style))
    content.append(Paragraph(
        prediction['explanation'],
        body_style
    ))
    content.append(Spacer(1, 24))
    
    # Recommendations
    content.append(Paragraph("Clinical Recommendations", heading_style))
    content.append(Paragraph(
        prediction['recommendations'],
        body_style
    ))
    
    # Build PDF
    doc.build(content)
    buffer.seek(0)
    
    # Return as download
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"DFI_Report_{prediction['drug']}_{prediction['food']}.pdf",
        mimetype='application/pdf'
    )

# Scheduled task to update research articles (would run via cron or similar)
def update_research_articles():
    try:
        # Use NewsAPI or similar service
        response = requests.get(
            'https://newsapi.org/v2/everything',
            params={
                'q': 'drug food interaction OR pharmacokinetics OR drug metabolism',
                'apiKey': os.environ.get('NEWS_API_KEY'),
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20
            }
        )
        
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            
            with get_db() as db:
                # Clear old articles
                db.execute('DELETE FROM research_articles')
                
                # Insert new articles
                for article in articles:
                    try:
                        db.execute(
                            'INSERT INTO research_articles '
                            '(title, source, author, description, url, image_url, published_at, category) '
                            'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                            (
                                article['title'],
                                article['source']['name'],
                                article['author'],
                                article['description'],
                                article['url'],
                                article['urlToImage'],
                                article['publishedAt'],
                                'general'  # Could categorize based on content
                            )
                        )
                    except sqlite3.IntegrityError:
                        continue  # Skip duplicates
                
                db.commit()
            logger.info(f"Updated {len(articles)} research articles")
    except Exception as e:
        logger.error(f"Error updating research articles: {str(e)}")

if __name__ == '__main__':
    # Update research articles on startup (in production, run this as a scheduled task)
    update_research_articles()
    
    app.run(debug=True, host='0.0.0.0', port=5000)