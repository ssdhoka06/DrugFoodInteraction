this is my code, XAI is in an endless loop and keeps opening excessively on chrome and also shows a lot of duplicates, this is a snippet of the output i am getting
MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
📋 Basic Information:
   Drug Category: statin
   Food Category: citrus
   Mechanism: cyp3a4_inhibition
   Risk Level: HIGH

🤖 Model Decision:
   Prediction: INTERACTION
   Confidence: 0.997
python(95242) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
📋 Basic Information:
   Drug Category: statin
   Food Category: citrus
   Mechanism: cyp3a4_inhibition
   Risk Level: HIGH

🤖 Model Decision:
   Prediction: INTERACTION
   Confidence: 0.997
python(95258) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
📋 Basic Information:
   Drug Category: statin
   Food Category: citrus
   Mechanism: cyp3a4_inhibition
   Risk Level: HIGH

🤖 Model Decision:
   Prediction: INTERACTION
   Confidence: 0.997
python(95260) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
📋 Basic Information:
   Drug Category: statin
   Food Category: citrus
   Mechanism: cyp3a4_inhibition
   Risk Level: HIGH

🤖 Model Decision:
   Prediction: INTERACTION
   Confidence: 0.997
python(95264) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
📋 Basic Information:
   Drug Category: statin
   Food Category: citrus
   Mechanism: cyp3a4_inhibition
   Risk Level: HIGH

🤖 Model Decision:
   Prediction: INTERACTION
   Confidence: 0.997
python(95269) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
📋 Basic Information:
   Drug Category: statin
   Food Category: citrus
   Mechanism: cyp3a4_inhibition
   Risk Level: HIGH

🤖 Model Decision:
   Prediction: INTERACTION
   Confidence: 0.997
python(95288) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
📋 Basic Information:
   Drug Category: statin
   Food Category: citrus
   Mechanism: cyp3a4_inhibition
   Risk Level: HIGH

🤖 Model Decision:
   Prediction: INTERACTION
   Confidence: 0.997
python(95318) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
📋 Basic Information:
   Drug Category: statin
   Food Category: citrus
   Mechanism: cyp3a4_inhibition
   Risk Level: HIGH

🤖 Model Decision:
   Prediction: INTERACTION
   Confidence: 0.997
python(95324) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
📋 Basic Information:
   Drug Category: statin
   Food Category: citrus
   Mechanism: cyp3a4_inhibition
   Risk Level: HIGH

🤖 Model Decision:
   Prediction: INTERACTION
   Confidence: 0.997
python(95327) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
📋 Basic Information:
   Drug Category: statin
   Food Category: citrus
   Mechanism: cyp3a4_inhibition
   Risk Level: HIGH

🤖 Model Decision:
   Prediction: INTERACTION
   Confidence: 0.997
python(95331) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
fix the code for that and also give me the entire code for backend using flask that integrates the best ml model for prediction, frontend that i will add later on once its done so put a placeholder there and for  XAI

import json
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, accuracy_score, precision_score, recall_score, 
                           f1_score, precision_recall_curve, average_precision_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re
import networkx as nx
from scipy.sparse import csr_matrix
import pickle
import warnings
from collections import Counter

# Install required packages if not available
try:
    import lightgbm as lgb
except ImportError:
    print("Installing lightgbm...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])
    import lightgbm as lgb

try:
    import catboost as cb
except ImportError:
    print("Installing catboost...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])
    import catboost as cb

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️ XGBoost not available. Skipping XGBoost model.")
    XGBOOST_AVAILABLE = False

# XAI Libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("Installing SHAP...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
    import shap
    SHAP_AVAILABLE = True

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    print("Installing LIME...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lime"])
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True

# Additional visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

print("🚀 ENHANCED DRUG-FOOD INTERACTION PREDICTOR WITH RISK CATEGORIZATION")
print("=" * 80)

# ENHANCED DATA PREPROCESSING
print("\n📋 PHASE 1: ENHANCED DATA PREPROCESSING")
print("-" * 40)

def load_and_clean_foodrugs(filepath='/Users/sachidhoka/Desktop/food-drug interactions.csv'):
    """Load and clean FooDrugs dataset with enhanced preprocessing"""
    print("Loading FooDrugs dataset...")
    
    try:
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(filepath, encoding=encoding, on_bad_lines='skip', low_memory=False)
                print(f"Successfully loaded with {encoding} encoding")
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        else:
            raise Exception("Could not load file with any encoding")
            
    except FileNotFoundError:
        print(f"⚠️ FooDrugs file not found at {filepath}. Creating enhanced sample data...")
        # Create comprehensive sample data
        drugs = ['warfarin', 'simvastatin', 'tetracycline', 'aspirin', 'metformin', 
                'lisinopril', 'sertraline', 'digoxin', 'amoxicillin', 'atorvastatin',
                'ibuprofen', 'omeprazole', 'losartan', 'metoprolol', 'fluoxetine',
                'amlodipine', 'levothyroxine', 'prednisone', 'gabapentin', 'tramadol'] * 150
        
        foods = ['spinach', 'grapefruit', 'milk', 'alcohol', 'bread', 'banana', 
                'coffee', 'cheese', 'broccoli', 'orange', 'yogurt', 'kale',
                'wine', 'tea', 'avocado', 'salt', 'calcium', 'fiber', 'beans', 'nuts'] * 150
        
        np.random.shuffle(drugs)
        np.random.shuffle(foods)
        
        sample_data = {'drug': drugs, 'food': foods}
        df = pd.DataFrame(sample_data)
    
    print(f"Original dataset size: {len(df)}")
    
    # Enhanced cleaning
    df_clean = df.dropna(subset=['drug', 'food'])
    df_clean = df_clean.drop_duplicates(subset=['drug', 'food'])
    
    # Advanced text cleaning
    def clean_text(text):
        text = str(text).lower().strip()
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text
    
    df_clean['drug'] = df_clean['drug'].apply(clean_text)
    df_clean['food'] = df_clean['food'].apply(clean_text)
    
    # Remove invalid entries
    df_clean = df_clean[
        (df_clean['drug'].str.len() > 2) & 
        (df_clean['food'].str.len() > 2) &
        (~df_clean['drug'].str.contains(r'^\d+$')) &
        (~df_clean['food'].str.contains(r'^\d+$'))
    ]
    
    df_clean = df_clean[['drug', 'food']].copy()
    df_clean['interaction'] = 1
    
    print(f"Clean dataset size: {len(df_clean)} interactions")
    print(f"Unique drugs: {df_clean['drug'].nunique()}")
    print(f"Unique foods: {df_clean['food'].nunique()}")
    
    return df_clean

# Load data
df_clean = load_and_clean_foodrugs()

# Enhanced knowledge base with risk levels
drug_categories = {
    'anticoagulant': ['warfarin', 'heparin', 'coumadin', 'dabigatran', 'rivaroxaban', 'apixaban'],
    'statin': ['simvastatin', 'atorvastatin', 'lovastatin', 'rosuvastatin', 'pravastatin', 'fluvastatin'],
    'antibiotic': ['amoxicillin', 'penicillin', 'tetracycline', 'doxycycline', 'ciprofloxacin', 'azithromycin', 'erythromycin'],
    'antihypertensive': ['lisinopril', 'amlodipine', 'losartan', 'metoprolol', 'atenolol', 'hydrochlorothiazide'],
    'antidepressant': ['sertraline', 'fluoxetine', 'paroxetine', 'citalopram', 'escitalopram', 'venlafaxine'],
    'diabetes': ['metformin', 'glipizide', 'insulin', 'glyburide', 'pioglitazone', 'glimepiride'],
    'pain_relief': ['aspirin', 'ibuprofen', 'acetaminophen', 'naproxen', 'celecoxib', 'tramadol'],
    'heart_rhythm': ['digoxin', 'amiodarone', 'flecainide', 'propafenone', 'quinidine'],
    'ppi': ['omeprazole', 'lansoprazole', 'pantoprazole', 'esomeprazole', 'rabeprazole'],
    'thyroid': ['levothyroxine', 'liothyronine', 'methimazole', 'propylthiouracil'],
    'steroid': ['prednisone', 'prednisolone', 'hydrocortisone', 'dexamethasone']
}

food_categories = {
    'citrus': ['grapefruit', 'orange', 'lemon', 'lime', 'tangerine', 'pomelo'],
    'dairy': ['milk', 'cheese', 'yogurt', 'calcium', 'ice cream', 'butter'],
    'alcohol': ['alcohol', 'ethanol', 'wine', 'beer', 'spirits', 'vodka'],
    'leafy_greens': ['spinach', 'kale', 'lettuce', 'broccoli', 'brussels sprouts', 'arugula'],
    'high_potassium': ['banana', 'potassium', 'avocado', 'orange juice', 'coconut water', 'tomato'],
    'high_sodium': ['salt', 'sodium', 'processed foods', 'pickles', 'soup', 'chips'],
    'caffeinated': ['coffee', 'tea', 'caffeine', 'cola', 'energy drinks', 'chocolate'],
    'fiber_rich': ['bran', 'oats', 'fiber', 'whole grains', 'beans', 'nuts'],
    'fermented': ['soy sauce', 'cheese', 'yogurt', 'kimchi', 'sauerkraut'],
    'cruciferous': ['broccoli', 'cauliflower', 'cabbage', 'brussels sprouts']
}

# Enhanced interaction mechanisms with risk levels
high_risk_interactions = {
    ('anticoagulant', 'leafy_greens'): {'mechanism': 'vitamin_k_competition', 'risk': 'HIGH'},
    ('statin', 'citrus'): {'mechanism': 'cyp3a4_inhibition', 'risk': 'HIGH'},
    ('antibiotic', 'dairy'): {'mechanism': 'calcium_chelation', 'risk': 'MODERATE'},
    ('antidepressant', 'alcohol'): {'mechanism': 'cns_depression', 'risk': 'HIGH'},
    ('diabetes', 'alcohol'): {'mechanism': 'hypoglycemia_risk', 'risk': 'HIGH'},
    ('heart_rhythm', 'high_potassium'): {'mechanism': 'arrhythmia_risk', 'risk': 'HIGH'},
    ('pain_relief', 'alcohol'): {'mechanism': 'gi_bleeding_risk', 'risk': 'MODERATE'},
    ('thyroid', 'fiber_rich'): {'mechanism': 'absorption_interference', 'risk': 'MODERATE'},
    ('ppi', 'caffeinated'): {'mechanism': 'acid_suppression_interference', 'risk': 'LOW'},
    ('antihypertensive', 'high_sodium'): {'mechanism': 'bp_elevation', 'risk': 'MODERATE'},
    ('steroid', 'high_sodium'): {'mechanism': 'fluid_retention', 'risk': 'MODERATE'}
}

def categorize_entity(entity, categories):
    """Enhanced categorization with partial matching"""
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

# Apply categorization
print("🏷️ Categorizing drugs and foods...")
df_clean['drug_category'] = df_clean['drug'].apply(lambda x: categorize_entity(x, drug_categories))
df_clean['food_category'] = df_clean['food'].apply(lambda x: categorize_entity(x, food_categories))
interaction_details = df_clean.apply(
    lambda x: get_interaction_details(x['drug_category'], x['food_category']), 
    axis=1
)
df_clean['mechanism'] = [details[0] for details in interaction_details]
df_clean['risk_level'] = [details[1] for details in interaction_details]
print("\nDrug category distribution:")
print(df_clean['drug_category'].value_counts().head(10))
print("\nFood category distribution:")
print(df_clean['food_category'].value_counts().head(10))
print("\nRisk level distribution:")
print(df_clean['risk_level'].value_counts())

# ENHANCED NEGATIVE SAMPLE GENERATION
print("\n⚖️ ENHANCED NEGATIVE SAMPLE GENERATION")
print("-" * 40)

def generate_balanced_negatives(df_positive, target_ratio=0.8):
    """Generate balanced negative samples with improved strategy"""
    
    unique_drugs = df_positive['drug'].unique()
    unique_foods = df_positive['food'].unique()
    
    # Limit entities for computational efficiency
    if len(unique_drugs) > 2000:
        unique_drugs = np.random.choice(unique_drugs, 2000, replace=False)
    if len(unique_foods) > 2000:
        unique_foods = np.random.choice(unique_foods, 2000, replace=False)
    
    print(f"Working with {len(unique_drugs)} drugs and {len(unique_foods)} foods")
    
    existing_interactions = set(zip(df_positive['drug'], df_positive['food']))
    target_negatives = int(len(df_positive) * target_ratio)
    
    negative_samples = []
    attempts = 0
    max_attempts = target_negatives * 20
    
    # Enhanced negative sampling strategy
    while len(negative_samples) < target_negatives and attempts < max_attempts:
        drug = np.random.choice(unique_drugs)
        food = np.random.choice(unique_foods)
        attempts += 1
        
        if (drug, food) not in existing_interactions:
            drug_cat = categorize_entity(drug, drug_categories)
            food_cat = categorize_entity(food, food_categories)
            mechanism, risk_level = get_interaction_details(drug_cat, food_cat)
            
            # More lenient negative sampling - include some potential interactions as negatives
            # to make the model learn nuanced differences
            probability_negative = 0.7  # 70% chance to include as negative
            
            if mechanism == 'unknown':
                probability_negative = 0.9  # Higher chance for unknown mechanisms
            elif risk_level == 'HIGH':
                probability_negative = 0.2  # Lower chance for high-risk mechanisms
            elif risk_level == 'MODERATE':
                probability_negative = 0.4  # Medium chance for moderate-risk
            
            if np.random.random() < probability_negative:
                negative_samples.append({
                    'drug': drug,
                    'food': food,
                    'interaction': 0,
                    'drug_category': drug_cat,
                    'food_category': food_cat,
                    'mechanism': mechanism,
                    'risk_level': risk_level
                })
    
    return pd.DataFrame(negative_samples)

# Generate balanced negatives
df_negatives = generate_balanced_negatives(df_clean, target_ratio=0.8)
df_final = pd.concat([df_clean, df_negatives], ignore_index=True)

print(f"Final dataset: {len(df_final)} samples")
print(f"Positive interactions: {len(df_clean)}")
print(f"Negative interactions: {len(df_negatives)}")
print(f"Balance ratio: {len(df_negatives)/len(df_clean):.2f}")

# SPECIALIZED MODELS IMPLEMENTATION
print("\n🧠 IMPLEMENTING SPECIALIZED MODELS")
print("-" * 40)

# ENHANCED FEATURE ENGINEERING
print("\n🔧 PHASE 2: ENHANCED FEATURE ENGINEERING")
print("-" * 40)

def create_enhanced_features(df):
    """Create comprehensive feature set"""
    
    # Initialize TF-IDF vectorizers
    drug_tfidf = TfidfVectorizer(max_features=50, ngram_range=(1, 2), stop_words='english')
    food_tfidf = TfidfVectorizer(max_features=50, ngram_range=(1, 2), stop_words='english')
    
    # 1. Basic categorical features
    drug_dummies = pd.get_dummies(df['drug_category'], prefix='drug').astype(int)
    food_dummies = pd.get_dummies(df['food_category'], prefix='food').astype(int)
    mechanism_dummies = pd.get_dummies(df['mechanism'], prefix='mechanism').astype(int)
    risk_dummies = pd.get_dummies(df['risk_level'], prefix='risk').astype(int)
    
    # 2. Risk scoring features
    def create_detailed_risk_score(risk_level, mechanism, drug_category, food_category):
        """Enhanced risk scoring with more granular categories"""
        base_scores = {'HIGH': 4, 'MODERATE': 2, 'LOW': 1}
        base_score = base_scores.get(risk_level, 1)
        
        # Add mechanism-specific adjustments
        if mechanism in ['cyp3a4_inhibition', 'vitamin_k_competition']:
            base_score += 1
        
        # Add category-specific adjustments
        if drug_category == 'anticoagulant' and food_category == 'leafy_greens':
            base_score += 2
        
        return min(base_score, 5)  # Cap at 5
    
    df['risk_score'] = df.apply(
        lambda x: create_detailed_risk_score(x['risk_level'], x['mechanism'], x['drug_category'], x['food_category']), 
        axis=1
    )
    
    # 3. Text-based features using TF-IDF
    print("Creating text-based features...")
    
    # Combine drug and food names for text analysis
    df['combined_text'] = df['drug'] + ' ' + df['food']
    
    # TF-IDF for drug and food names
    try:
        drug_tfidf_features = drug_tfidf.fit_transform(df['drug'])
        food_tfidf_features = food_tfidf.fit_transform(df['food'])
    except ValueError:
        # Fallback if TF-IDF fails
        drug_tfidf_features = np.zeros((len(df), 50))
        food_tfidf_features = np.zeros((len(df), 50))
    
    # Reduce dimensionality
    svd_drug = TruncatedSVD(n_components=10, random_state=42)
    svd_food = TruncatedSVD(n_components=10, random_state=42)
    
    try:
        drug_features = svd_drug.fit_transform(drug_tfidf_features)
        food_features = svd_food.fit_transform(food_tfidf_features)
    except ValueError:
        drug_features = np.zeros((len(df), 10))
        food_features = np.zeros((len(df), 10))
    
    # Convert to DataFrames
    drug_text_df = pd.DataFrame(drug_features, columns=[f'drug_text_{i}' for i in range(10)])
    food_text_df = pd.DataFrame(food_features, columns=[f'food_text_{i}' for i in range(10)])
    
    # 4. Statistical features
    df['drug_length'] = df['drug'].str.len()
    df['food_length'] = df['food'].str.len()
    df['name_similarity'] = df.apply(
        lambda x: len(set(x['drug']) & set(x['food'])) / max(len(set(x['drug']) | set(x['food'])), 1),
        axis=1
    )
    
    # 5. Category interaction features
    df['same_category'] = (df['drug_category'] == df['food_category']).astype(int)
    df['both_other'] = ((df['drug_category'] == 'other') & (df['food_category'] == 'other')).astype(int)
    
    # Combine all features
    feature_dfs = [
        drug_dummies, food_dummies, mechanism_dummies, risk_dummies,
        df[['risk_score', 'drug_length', 'food_length', 'name_similarity', 'same_category', 'both_other']],
        drug_text_df, food_text_df
    ]
    
    X = pd.concat(feature_dfs, axis=1)
    X = X.astype(float)
    
    # Store feature names and vectorizers for later use
    feature_info = {
        'drug_tfidf': drug_tfidf,
        'food_tfidf': food_tfidf,
        'svd_drug': svd_drug,
        'svd_food': svd_food,
        'feature_names': list(X.columns)
    }
    
    return X, feature_info

# Create enhanced features
X, feature_info = create_enhanced_features(df_final)
y = df_final['interaction']

print(f"Enhanced feature matrix shape: {X.shape}")
print(f"Total features: {X.shape[1]}")
print(f"Feature types: {X.dtypes.value_counts()}")

# Check for any issues
print(f"NaN values: {X.isnull().sum().sum()}")
print(f"Infinite values: {np.isinf(X).sum().sum()}")

# Class distribution
class_counts = Counter(y)
print(f"Class distribution: {dict(class_counts)}")
print(f"Balance ratio: {class_counts[0]/class_counts[1]:.2f}")

# ENHANCED MODEL TRAINING
print("\n🤖 PHASE 3: ENHANCED MODEL TRAINING")
print("-" * 40)

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Handle any remaining NaN or infinite values
X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0)
X_test = X_test.fillna(0).replace([np.inf, -np.inf], 0)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate class weights
train_counts = Counter(y_train)
weight_ratio = train_counts[0] / train_counts[1] if train_counts[1] > 0 else 1

print(f"Training class distribution: {dict(train_counts)}")
print(f"Class weight ratio: {weight_ratio:.2f}")

# Enhanced model evaluation function
def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """Comprehensive model evaluation"""
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0,
        'avg_precision': average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Interaction', 'Interaction'],
                yticklabels=['No Interaction', 'Interaction'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, model_name):
    """Plot ROC curve"""
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Initialize models dictionary
models = {}
results = []

print("🔬 Training Specified Models...")
print("-" * 30)

# 1. LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=10,
    num_leaves=31,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)
models['LightGBM'] = lgb_model

# 2. MLP (Multi-Layer Perceptron)
mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate='adaptive',
    max_iter=500,
    random_state=42
)
models['MLP'] = mlp_model

# 3. Extra Trees Classifier
et_model = ExtraTreesClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
models['Extra Trees'] = et_model

# 4. CatBoost
cb_model = cb.CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    depth=8,
    class_weights=[1, weight_ratio],
    random_seed=42,
    verbose=False
)
models['CatBoost'] = cb_model

# 5. Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
models['Random Forest'] = rf_model

# 6. XGBoost (if available)
if XGBOOST_AVAILABLE:
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=weight_ratio,
        random_state=42,
        eval_metric='logloss'
    )
    models['XGBoost'] = xgb_model

# 7. Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.8,
    random_state=42
)
models['Gradient Boosting'] = gb_model

print("🧠 Training Specialized Models...")
print("-" * 30)

# Train and evaluate all models
print("\n🚀 TRAINING AND EVALUATION")
print("=" * 50)

# First, train all individual models for the voting classifier
trained_models = {}

for model_name, model in models.items():
    print(f"\n🔧 Training {model_name}...")
    
    try:
        # Choose appropriate data based on model type
        if model_name in ['MLP']:
            X_train_use = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            X_test_use = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        else:
            # Use original data for tree-based and specialized models
            X_train_use = X_train
            X_test_use = X_test
        
        # Train model
        model.fit(X_train_use, y_train)
        trained_models[model_name] = model
        
        # Make predictions
        y_pred = model.predict(X_test_use)
        
        # Get probabilities (handle models without predict_proba)
        try:
            y_pred_proba = model.predict_proba(X_test_use)[:, 1]
        except:
            # For models without predict_proba, use decision_function or predictions
            try:
                y_pred_proba = model.decision_function(X_test_use)
                # Normalize to [0, 1] range
                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
            except:
                y_pred_proba = y_pred.astype(float)
        
        # Evaluate model
        metrics = evaluate_model(y_test, y_pred, y_pred_proba, model_name)
        results.append(metrics)
        
        # Print results
        print(f"✅ {model_name} Results:")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1']:.4f}")
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"   Avg Precision: {metrics['avg_precision']:.4f}")
        
    except Exception as e:
        print(f"❌ Error training {model_name}: {str(e)}")
        continue

# Now create and train the Voting Classifier
print(f"\n🤝 Training Voting Classifier...")
print("-" * 30)

# Select best performing models for voting (top 3-5)
voting_models = []
for model_name in ['Random Forest', 'Extra Trees', 'LightGBM', 'CatBoost']:
    if model_name in trained_models:
        voting_models.append((model_name.replace(' ', '_'), trained_models[model_name]))

# Add XGBoost if available
if 'XGBoost' in trained_models:
    voting_models.append(('XGBoost', trained_models['XGBoost']))

valid_voting_models = []
for name, model in voting_models:
    if hasattr(model, 'predict_proba'):
        valid_voting_models.append((name, model))

if len(valid_voting_models) >= 2:
    try:
        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=valid_voting_models,
            voting='soft'  # Use probability voting
        )
        
        # Train voting classifier
        voting_clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred_voting = voting_clf.predict(X_test)
        y_pred_proba_voting = voting_clf.predict_proba(X_test)[:, 1]
        
        # Evaluate voting classifier
        voting_metrics = evaluate_model(y_test, y_pred_voting, y_pred_proba_voting, 'Voting Classifier')
        results.append(voting_metrics)
        
        print(f"✅ Voting Classifier Results:")
        print(f"   Accuracy: {voting_metrics['accuracy']:.4f}")
        print(f"   Precision: {voting_metrics['precision']:.4f}")
        print(f"   Recall: {voting_metrics['recall']:.4f}")
        print(f"   F1-Score: {voting_metrics['f1']:.4f}")
        print(f"   ROC-AUC: {voting_metrics['roc_auc']:.4f}")
        print(f"   Avg Precision: {voting_metrics['avg_precision']:.4f}")
        
        # Add to models dictionary
        models['Voting Classifier'] = voting_clf
        
    except Exception as e:
        print(f"❌ Error training Voting Classifier: {str(e)}")
else:
    print("❌ Not enough models with predict_proba for Voting Classifier")

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('f1', ascending=False)

print("\n📊 COMPREHENSIVE RESULTS SUMMARY")
print("=" * 80)
print("Final Results for Selected Models:")
print("-" * 40)
print("Models included: LightGBM, MLP, Voting Classifier, Extra Trees, CatBoost,")
print("                Random Forest, XGBoost, Gradient Boosting")
print("-" * 40)
print(results_df.round(4))

# Plot comparison of all models
plt.figure(figsize=(15, 10))

# F1-Score comparison
plt.subplot(2, 3, 1)
plt.barh(results_df['model'], results_df['f1'])
plt.xlabel('F1-Score')
plt.title('F1-Score Comparison')
plt.gca().invert_yaxis()

# Accuracy comparison
plt.subplot(2, 3, 2)
plt.barh(results_df['model'], results_df['accuracy'])
plt.xlabel('Accuracy')
plt.title('Accuracy Comparison')
plt.gca().invert_yaxis()

# ROC-AUC comparison
plt.subplot(2, 3, 3)
plt.barh(results_df['model'], results_df['roc_auc'])
plt.xlabel('ROC-AUC')
plt.title('ROC-AUC Comparison')
plt.gca().invert_yaxis()

# Precision comparison
plt.subplot(2, 3, 4)
plt.barh(results_df['model'], results_df['precision'])
plt.xlabel('Precision')
plt.title('Precision Comparison')
plt.gca().invert_yaxis()

# Recall comparison
plt.subplot(2, 3, 5)
plt.barh(results_df['model'], results_df['recall'])
plt.xlabel('Recall')
plt.title('Recall Comparison')
plt.gca().invert_yaxis()

# Average Precision comparison
plt.subplot(2, 3, 6)
plt.barh(results_df['model'], results_df['avg_precision'])
plt.xlabel('Average Precision')
plt.title('Average Precision Comparison')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# Plot confusion matrices for top 3 models
print("\n🎯 CONFUSION MATRICES FOR TOP 3 MODELS")
print("=" * 50)

top_3_models = results_df.head(3)['model'].tolist()

for model_name in top_3_models:
    if model_name in models:
        model = models[model_name]
        
        # Choose appropriate test data
        if model_name == 'MLP':
            X_test_use = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        else:
            X_test_use = X_test
            
        try:
            y_pred = model.predict(X_test_use)
            plot_confusion_matrix(y_test, y_pred, model_name)
            
            # Also plot ROC curve
            try:
                y_pred_proba = model.predict_proba(X_test_use)[:, 1]
                plot_roc_curve(y_test, y_pred_proba, model_name)
            except:
                try:
                    y_pred_proba = model.decision_function(X_test_use)
                    y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
                    plot_roc_curve(y_test, y_pred_proba, model_name)
                except:
                    print(f"Could not plot ROC curve for {model_name}")
        except Exception as e:
            print(f"Error plotting for {model_name}: {str(e)}")

# XAI IMPLEMENTATION
print("\n🔍 PHASE 4: EXPLAINABLE AI (XAI) IMPLEMENTATION")
print("-" * 50)

class DrugFoodXAI:
    """Comprehensive XAI analysis for drug-food interactions"""
    
    def __init__(self, model, X_train, X_test, y_test, feature_names):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.shap_explainer = None
        self.lime_explainer = None
        
    def initialize_explainers(self):
        """Initialize SHAP and LIME explainers"""
        print("🔧 Initializing XAI explainers...")
        
        # SHAP explainer
        if hasattr(self.model, 'predict_proba'):
            try:
                # Use a subset for faster computation
                background_data = self.X_train.sample(min(100, len(self.X_train)))
                self.shap_explainer = shap.TreeExplainer(self.model, background_data)
                print("✅ SHAP TreeExplainer initialized")
            except:
                try:
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict_proba, 
                        self.X_train.sample(min(50, len(self.X_train)))
                    )
                    print("✅ SHAP KernelExplainer initialized")
                except Exception as e:
                    print(f"⚠️ SHAP initialization failed: {e}")
        
        # LIME explainer
        try:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_train.values,
                feature_names=self.feature_names,
                class_names=['No Interaction', 'Interaction'],
                mode='classification'
            )
            print("✅ LIME explainer initialized")
        except Exception as e:
            print(f"⚠️ LIME initialization failed: {e}")
    
    def explain_prediction(self, instance_idx, method='both'):
        """Explain a specific prediction"""
        instance = self.X_test.iloc[instance_idx:instance_idx+1]
        true_label = self.y_test.iloc[instance_idx]
        pred_label = self.model.predict(instance)[0]
        pred_proba = self.model.predict_proba(instance)[0, 1] if hasattr(self.model, 'predict_proba') else pred_label
        
        print(f"\n🔍 EXPLAINING PREDICTION #{instance_idx}")
        print(f"True Label: {true_label}, Predicted: {pred_label}, Probability: {pred_proba:.3f}")
        print("-" * 40)
        
        explanations = {}
        
        # SHAP explanation
        if method in ['shap', 'both'] and self.shap_explainer:
            try:
                shap_values = self.shap_explainer.shap_values(instance)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification
                
                explanations['shap'] = {
                    'values': shap_values[0],
                    'base_value': self.shap_explainer.expected_value[1] if isinstance(self.shap_explainer.expected_value, np.ndarray) else self.shap_explainer.expected_value,
                    'feature_names': self.feature_names
                }
                
                # Plot SHAP waterfall
                shap.plots.waterfall(shap.Explanation(
                    values=shap_values[0],
                    base_values=explanations['shap']['base_value'],
                    data=instance.values[0],
                    feature_names=self.feature_names
                ))
                
            except Exception as e:
                print(f"⚠️ SHAP explanation failed: {e}")
        
        # LIME explanation
        if method in ['lime', 'both'] and self.lime_explainer:
            try:
                lime_exp = self.lime_explainer.explain_instance(
                    instance.values[0],
                    self.model.predict_proba,
                    num_features=10
                )
                
                explanations['lime'] = lime_exp
                
                # Show LIME explanation
                lime_exp.show_in_notebook(show_table=True)
                
            except Exception as e:
                print(f"⚠️ LIME explanation failed: {e}")
        
        return explanations
    
    def global_feature_importance(self):
        """Global feature importance analysis using SHAP"""
        print("\n📊 GLOBAL FEATURE IMPORTANCE ANALYSIS")
        print("-" * 40)
        
        if not self.shap_explainer:
            print("⚠️ SHAP explainer not available")
            return
        
        try:
            # Calculate SHAP values for test set (subset for performance)
            test_subset = self.X_test.sample(min(100, len(self.X_test)))
            shap_values = self.shap_explainer.shap_values(test_subset)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
            
            # Summary plot
            shap.summary_plot(shap_values, test_subset, feature_names=self.feature_names)
            
            # Feature importance plot
            shap.summary_plot(shap_values, test_subset, feature_names=self.feature_names, plot_type="bar")
            
            return shap_values
            
        except Exception as e:
            print(f"⚠️ Global analysis failed: {e}")
            return None
    
    def feature_interaction_analysis(self):
        """Analyze feature interactions"""
        print("\n🔗 FEATURE INTERACTION ANALYSIS")
        print("-" * 40)
        
        if not self.shap_explainer:
            print("⚠️ SHAP explainer not available")
            return
        
        try:
            # Get interaction values
            test_subset = self.X_test.sample(min(50, len(self.X_test)))
            interaction_values = self.shap_explainer.shap_interaction_values(test_subset)
            
            # Plot interaction for top features
            shap.summary_plot(interaction_values, test_subset, feature_names=self.feature_names)
            
            return interaction_values
            
        except Exception as e:
            print(f"⚠️ Interaction analysis failed: {e}")
            return None
    
    def decision_pathway_analysis(self, drug_name, food_name):
        """Trace decision pathway for specific drug-food pair"""
        print(f"\n🛤️ DECISION PATHWAY: {drug_name.upper()} + {food_name.upper()}")
        print("-" * 50)
        
        # Get prediction for this pair
        result = predict_new_interaction_with_explanation(drug_name, food_name)
        
        if 'error' in result:
            print(f"⚠️ Error in prediction: {result['error']}")
            return
        
        print(f"📋 Basic Information:")
        print(f"   Drug Category: {result['drug_category']}")
        print(f"   Food Category: {result['food_category']}")
        print(f"   Mechanism: {result['mechanism']}")
        print(f"   Risk Level: {result['risk_level']}")
        
        print(f"\n🤖 Model Decision:")
        print(f"   Prediction: {'INTERACTION' if result['interaction_predicted'] else 'NO INTERACTION'}")
        print(f"   Confidence: {result['probability']:.3f}")
        
        # Create a pathway visualization
        pathway_data = {
            'Input': [drug_name, food_name],
            'Categories': [result['drug_category'], result['food_category']],
            'Risk_Assessment': [result['risk_level']],
            'Mechanism': [result['mechanism']],
            'Final_Prediction': [f"{'INTERACTION' if result['interaction_predicted'] else 'NO INTERACTION'} ({result['probability']:.3f})"]
        }
        
        # Create decision tree visualization
        fig = go.Figure()
        
        # Add pathway nodes
        steps = ['Input', 'Categorization', 'Risk Assessment', 'Mechanism Analysis', 'Final Prediction']
        values = [
            f"{drug_name} + {food_name}",
            f"{result['drug_category']} + {result['food_category']}",
            result['risk_level'],
            result['mechanism'],
            f"{'INTERACTION' if result['interaction_predicted'] else 'NO INTERACTION'}"
        ]
        
        colors = ['lightblue', 'lightgreen', 'yellow', 'orange', 'red' if result['interaction_predicted'] else 'green']
        
        fig.add_trace(go.Scatter(
            x=list(range(len(steps))),
            y=[1]*len(steps),
            mode='markers+text',
            marker=dict(size=60, color=colors),
            text=values,
            textposition='middle center',
            textfont=dict(size=10),
            name='Decision Pathway'
        ))
        
        fig.update_layout(
            title=f"Decision Pathway: {drug_name} + {food_name}",
            xaxis=dict(tickmode='array', tickvals=list(range(len(steps))), ticktext=steps),
            yaxis=dict(visible=False),
            showlegend=False,
            height=400
        )
        
        fig.show()
        
        return result

# Initialize XAI system
xai_system = DrugFoodXAI(
    model=models[results_df.iloc[0]['model']],
    X_train=X_train,
    X_test=X_test,
    y_test=y_test,
    feature_names=feature_info['feature_names']
)

xai_system.initialize_explainers()

def conduct_case_studies():
    """Conduct detailed case studies of high-risk interactions"""
    print("\n📚 CASE STUDY ANALYSIS")
    print("=" * 50)
    
    # Define case studies
    case_studies = [
        {
            'name': 'Warfarin-Spinach Interaction',
            'drug': 'warfarin',
            'food': 'spinach',
            'description': 'Classic vitamin K antagonist interaction'
        },
        {
            'name': 'Statin-Grapefruit Interaction',
            'drug': 'simvastatin',
            'food': 'grapefruit',
            'description': 'CYP3A4 enzyme inhibition leading to toxicity'
        },
        {
            'name': 'Antibiotic-Dairy Interaction',
            'drug': 'tetracycline',
            'food': 'milk',
            'description': 'Calcium chelation reducing absorption'
        }
    ]
    
    for i, case in enumerate(case_studies):
        print(f"\n📖 CASE STUDY {i+1}: {case['name']}")
        print("-" * 40)
        print(f"Description: {case['description']}")
        
        # Get detailed analysis
        result = xai_system.decision_pathway_analysis(case['drug'], case['food'])
        
        # Get educational insights
        insights = get_educational_insights(case['drug'], case['food'])
        print(f"\n💡 Educational Insight:")
        print(f"   {insights['patient_explanation']}")
        print(f"   Technical: {insights['professional_details']}")
        
        # Find similar cases in test set
        similar_cases = find_similar_interactions(case['drug'], case['food'])
        if similar_cases:
            print(f"   Similar cases found: {len(similar_cases)}")
        
        print("\n" + "="*50)

def find_similar_interactions(target_drug, target_food, top_n=5):
    """Find similar interactions in the dataset"""
    target_drug_cat = categorize_entity(target_drug, drug_categories)
    target_food_cat = categorize_entity(target_food, food_categories)
    
    # Find interactions with same categories
    similar = df_final[
        (df_final['drug_category'] == target_drug_cat) & 
        (df_final['food_category'] == target_food_cat)
    ].head(top_n)
    
    return similar[['drug', 'food', 'interaction', 'risk_level']].to_dict('records')

def create_interactive_dashboard():
    """Create interactive explanation dashboard"""
    print("\n📊 CREATING INTERACTIVE EXPLANATION DASHBOARD")
    print("-" * 50)
    
    # Sample predictions for dashboard
    sample_predictions = []
    for drug, food in [('warfarin', 'spinach'), ('aspirin', 'alcohol'), ('metformin', 'banana')]:
        result = predict_new_interaction_with_explanation(drug, food)
        sample_predictions.append(result)
    
    # Create dashboard with multiple visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Risk Distribution', 'Model Confidence', 'Category Interactions', 'Prediction Timeline'),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "heatmap"}, {"type": "scatter"}]]
    )
    
    # Risk distribution pie chart
    risk_counts = df_final['risk_level'].value_counts()
    fig.add_trace(
        go.Pie(labels=risk_counts.index, values=risk_counts.values, name="Risk Distribution"),
        row=1, col=1
    )
    
    # Model confidence bar chart
    confidences = [pred['probability'] for pred in sample_predictions]
    labels = [f"{pred['drug']}-{pred['food']}" for pred in sample_predictions]
    fig.add_trace(
        go.Bar(x=labels, y=confidences, name="Model Confidence"),
        row=1, col=2
    )
    
    # Category interaction heatmap
    interaction_matrix = pd.crosstab(df_final['drug_category'], df_final['food_category'], df_final['interaction'], aggfunc='mean')
    fig.add_trace(
        go.Heatmap(z=interaction_matrix.values, x=interaction_matrix.columns, y=interaction_matrix.index),
        row=2, col=1
    )
    
    # Prediction timeline (if we had timestamps)
    fig.add_trace(
        go.Scatter(x=list(range(len(sample_predictions))), y=confidences, mode='lines+markers', name="Predictions"),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Drug-Food Interaction XAI Dashboard")
    fig.show()

def explain_model_behavior():
    """Explain overall model behavior patterns"""
    print("\n🧠 MODEL BEHAVIOR ANALYSIS")
    print("-" * 40)
    
    # Analyze model predictions by categories
    behavior_analysis = df_final.groupby(['drug_category', 'food_category']).agg({
        'interaction': ['count', 'mean'],
        'risk_level': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'
    }).round(3)
    
    print("Model Decision Patterns by Category Combinations:")
    print(behavior_analysis.head(10))
    
    # Identify model biases
    print("\n⚖️ Potential Model Biases:")
    high_confidence_wrong = 0  # This would need actual implementation with confidence thresholds
    
    # Category bias analysis
    category_bias = df_final.groupby('drug_category')['interaction'].mean().sort_values(ascending=False)
    print(f"Drug categories with highest interaction rates:")
    print(category_bias.head())
    
    food_bias = df_final.groupby('food_category')['interaction'].mean().sort_values(ascending=False)
    print(f"\nFood categories with highest interaction rates:")
    print(food_bias.head())

# Create ensemble model with top performers
print("\n🏆 FINAL MODEL RANKINGS")
print("=" * 40)

# Show final results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('f1', ascending=False)
print("Final Model Performance Rankings:")
print(results_df[['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']].round(4))

print(f"\n🥇 Best Model: {results_df.iloc[0]['model']} (F1: {results_df.iloc[0]['f1']:.4f})")

# Risk Analysis
print("\n⚠️ RISK LEVEL ANALYSIS")
print("=" * 40)

# Analyze predictions by risk level
risk_analysis = df_final.groupby('risk_level').agg({
    'interaction': ['count', 'sum', 'mean']
}).round(3)

risk_analysis.columns = ['Total_Samples', 'Positive_Interactions', 'Interaction_Rate']
print("Risk Level Distribution:")
print(risk_analysis)

# Plot risk level distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
risk_counts = df_final['risk_level'].value_counts()
plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Risk Levels')

plt.subplot(1, 2, 2)
risk_interaction_rates = df_final.groupby('risk_level')['interaction'].mean()
colors = ['green', 'orange', 'red']
plt.bar(risk_interaction_rates.index, risk_interaction_rates.values, color=colors)
plt.title('Interaction Rate by Risk Level')
plt.xlabel('Risk Level')
plt.ylabel('Interaction Rate')
plt.ylim(0, 1)

# Add value labels on bars
for i, v in enumerate(risk_interaction_rates.values):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Feature importance analysis for best model
print("\n📈 FEATURE IMPORTANCE ANALYSIS")
print("=" * 40)

best_model_name = results_df.iloc[0]['model']
if best_model_name in models:
    best_model = models[best_model_name]
    
    try:
        # Get feature importance
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
        elif hasattr(best_model, 'coef_'):
            importances = np.abs(best_model.coef_[0])
        else:
            print(f"Feature importance not available for {best_model_name}")
            importances = None
        
        if importances is not None:
            # Create feature importance DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': feature_info['feature_names'],
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Plot top 20 features
            top_features = feature_importance_df.head(20)
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 20 Feature Importances - {best_model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
            print("Top 10 Most Important Features:")
            print(feature_importance_df.head(10)[['feature', 'importance']].round(4))
    
    except Exception as e:
        print(f"Error analyzing feature importance: {str(e)}")

# Enhanced prediction function
def predict_new_interaction_with_explanation(drug_name, food_name, explain=True):
    """Enhanced prediction with XAI explanations"""
    # Get base prediction
    result = predict_new_interaction(drug_name, food_name)
    
    if explain and 'error' not in result:
        # Add explanation
        try:
            # Create temporary instance for explanation
            temp_df = pd.DataFrame({
                'drug': [drug_name.lower()],
                'food': [food_name.lower()],
                'interaction': [0]
            })
            
            # Apply preprocessing
            temp_df['drug_category'] = temp_df['drug'].apply(lambda x: categorize_entity(x, drug_categories))
            temp_df['food_category'] = temp_df['food'].apply(lambda x: categorize_entity(x, food_categories))
            
            # Get decision pathway
            pathway = xai_system.decision_pathway_analysis(drug_name, food_name)
            
            result['explanation'] = {
                'decision_pathway': pathway,
                'key_factors': f"Categories: {result['drug_category']} + {result['food_category']}",
                'confidence_level': 'High' if result['probability'] > 0.7 else 'Medium' if result['probability'] > 0.4 else 'Low'
            }
            
        except Exception as e:
            result['explanation'] = f"Explanation failed: {e}"
    
    return result

def predict_new_interaction(drug_name, food_name, model=None, return_risk=True):
    """Predict interaction for new drug-food pair"""
    
    if model is None:
        model = models[results_df.iloc[0]['model']]  # Use best model
    
    # Create a temporary dataframe for the new pair
    new_df = pd.DataFrame({
        'drug': [drug_name.lower().strip()],
        'food': [food_name.lower().strip()],
        'interaction': [0]  # Placeholder
    })
    
    # Apply same preprocessing
    new_df['drug_category'] = new_df['drug'].apply(lambda x: categorize_entity(x, drug_categories))
    new_df['food_category'] = new_df['food'].apply(lambda x: categorize_entity(x, food_categories))
    new_df[['mechanism', 'risk_level']] = new_df.apply(
        lambda x: pd.Series(get_interaction_details(x['drug_category'], x['food_category'])), 
        axis=1
    )
    
    # Create features
    try:
        X_new, _ = create_enhanced_features(new_df)
        # Ensure all required columns exist and same order
        missing_cols = set(feature_info['feature_names']) - set(X_new.columns)
        for col in missing_cols:
            X_new[col] = 0
            
        # Ensure same order as training data
        X_new = X_new[feature_info['feature_names']]
        
        # Scale features if needed
        if model.__class__.__name__ == 'MLPClassifier':
            X_new = scaler.transform(X_new)
        
        # Make prediction
        prediction = model.predict(X_new)[0]
        
        try:
            probability = model.predict_proba(X_new)[0, 1]
        except:
            probability = prediction
        
        result = {
            'drug': drug_name,
            'food': food_name,
            'interaction_predicted': bool(prediction),
            'probability': float(probability),
            'drug_category': new_df['drug_category'].iloc[0],
            'food_category': new_df['food_category'].iloc[0],
            'mechanism': new_df['mechanism'].iloc[0],
            'risk_level': new_df['risk_level'].iloc[0]
        }
        
        return result
        
    except Exception as e:
        return {
            'drug': drug_name,
            'food': food_name,
            'error': str(e),
            'interaction_predicted': None,
            'probability': None
        }

def get_personalized_warning(drug_name, food_name, age=None, gender=None, conditions=None):
    """Generate personalized warnings based on patient factors"""
    base_result = predict_new_interaction_with_explanation(drug_name, food_name)
    
    # Adjust risk based on patient factors
    risk_multiplier = 1.0
    if age and age > 65:
        risk_multiplier += 0.2  # Higher risk for elderly
    if conditions and 'liver_disease' in conditions:
        risk_multiplier += 0.3
    if conditions and 'kidney_disease' in conditions:
        risk_multiplier += 0.2
    
    adjusted_probability = min(base_result['probability'] * risk_multiplier, 1.0)
    
    return {
        **base_result,
        'adjusted_probability': adjusted_probability,
        'personalized_warning': f"Risk adjusted for age: {age}, conditions: {conditions}"
    }

def check_meal_plan_compatibility(medications, meal_plan):
    """Check if meal plan is compatible with medications"""
    interactions_found = []
    
    for drug in medications:
        for food in meal_plan:
            result = predict_new_interaction_with_explanation(drug, food)
            if result['interaction_predicted'] and result['probability'] > 0.5:
                interactions_found.append(result)
    
    return {
        'safe': len(interactions_found) == 0,
        'interactions': interactions_found,
        'recommendations': f"Found {len(interactions_found)} potential interactions"
    }

def get_educational_insights(drug_name, food_name):
    """Provide educational explanations"""
    result = predict_new_interaction_with_explanation(drug_name, food_name)
    
    mechanism_explanations = {
        'cyp3a4_inhibition': "This food blocks liver enzymes that break down the medication, potentially causing dangerous buildup.",
        'calcium_chelation': "Calcium in this food binds to the medication, reducing absorption.",
        'vitamin_k_competition': "This food contains vitamin K which can interfere with blood-thinning effects.",
        'absorption_interference': "This food can slow down or reduce medication absorption in the stomach."
    }
    
    explanation = mechanism_explanations.get(result['mechanism'], "The interaction mechanism is not well understood.")
    
    return {
        **result,
        'patient_explanation': explanation,
        'professional_details': f"Mechanism: {result['mechanism']}, Category interaction: {result['drug_category']} + {result['food_category']}"
    }

# Example predictions
print("\n🔮 EXAMPLE PREDICTIONS")
print("=" * 40)

test_pairs = [
    ('warfarin', 'spinach'),      # Known high-risk interaction
    ('simvastatin', 'grapefruit'), # Known high-risk interaction
    ('aspirin', 'alcohol'),       # Known moderate-risk interaction
    ('amoxicillin', 'milk'),      # Known moderate-risk interaction
    ('metformin', 'banana'),      # Likely low-risk
    ('ibuprofen', 'coffee')       # Likely low-risk
]

for drug, food in test_pairs:
    result = predict_new_interaction_with_explanation(drug, food)
    
    if 'error' not in result:
        print(f"\n{drug.title()} + {food.title()}:")
        print(f"  Interaction: {'YES' if result['interaction_predicted'] else 'NO'}")
        print(f"  Probability: {result['probability']:.3f}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Mechanism: {result['mechanism']}")
        print(f"  Categories: {result['drug_category']} + {result['food_category']}")
    else:
        print(f"\n{drug.title()} + {food.title()}: Error - {result['error']}")

# EXECUTE XAI ANALYSIS
print("\n🔍 EXECUTING COMPREHENSIVE XAI ANALYSIS")
print("=" * 60)

# 1. Explain specific predictions
print("\n1️⃣ INDIVIDUAL PREDICTION EXPLANATIONS")
high_risk_indices = df_final[df_final['risk_level'] == 'HIGH'].index[:3]
for idx in high_risk_indices:
    if idx < len(X_test):
        try:
            xai_system.explain_prediction(idx, method='shap')
        except Exception as e:
            print(f"⚠️ Could not explain prediction {idx}: {e}")

# 2. Global feature importance
print("\n2️⃣ GLOBAL FEATURE IMPORTANCE")
try:
    global_shap = xai_system.global_feature_importance()
except Exception as e:
    print(f"⚠️ Global analysis failed: {e}")

# 3. Feature interactions
print("\n3️⃣ FEATURE INTERACTION ANALYSIS")
try:
    interactions = xai_system.feature_interaction_analysis()
except Exception as e:
    print(f"⚠️ Interaction analysis failed: {e}")

# 4. Decision pathways
print("\n4️⃣ DECISION PATHWAY ANALYSIS")
pathway_examples = [('warfarin', 'spinach'), ('simvastatin', 'grapefruit'), ('aspirin', 'coffee')]
for drug, food in pathway_examples:
    try:
        xai_system.decision_pathway_analysis(drug, food)
    except Exception as e:
        print(f"⚠️ Pathway analysis failed for {drug}-{food}: {e}")

# 5. Case studies
print("\n5️⃣ CASE STUDY ANALYSIS")
try:
    conduct_case_studies()
except Exception as e:
    print(f"⚠️ Case study analysis failed: {e}")

# 6. Interactive dashboard
print("\n6️⃣ INTERACTIVE DASHBOARD")
try:
    create_interactive_dashboard()
except Exception as e:
    print(f"⚠️ Dashboard creation failed: {e}")

# 7. Model behavior analysis
print("\n7️⃣ MODEL BEHAVIOR ANALYSIS")
try:
    explain_model_behavior()
except Exception as e:
    print(f"⚠️ Behavior analysis failed: {e}")

print("\n✅ ANALYSIS COMPLETE!")
print("=" * 80)
print("🎯 Key Findings:")
print(f"   • Best performing model: {results_df.iloc[0]['model']}")
print(f"   • Best F1-Score: {results_df.iloc[0]['f1']:.4f}")
print(f"   • Best ROC-AUC: {results_df.iloc[0]['roc_auc']:.4f}")
print(f"   • Total drug-food pairs analyzed: {len(df_final):,}")
print(f"   • High-risk interactions identified: {len(df_final[df_final['risk_level'] == 'HIGH']):,}")
print(f"   • Feature engineering created {X.shape[1]} features")
print("   • Models used: LightGBM, MLP, Voting, Extra Trees, CatBoost,")
print("                  Random Forest, XGBoost, Gradient Boosting")
print("   • Risk categorization system operational (HIGH/MODERATE/LOW)")

# Enhanced model saving with XAI components
model_package = {
    'model': best_model,
    'feature_info': feature_info,
    'scaler': scaler,
    'drug_categories': drug_categories,
    'food_categories': food_categories,
    'high_risk_interactions': high_risk_interactions,
    'model_performance': results_df.iloc[0].to_dict(),
    'training_date': datetime.now().isoformat(),
    'model_version': '2.0_XAI',
    'xai_system': xai_system,  # Save XAI system
    'xai_capabilities': ['SHAP', 'LIME', 'Decision_Pathways', 'Case_Studies', 'Interactive_Dashboard']
}

try:
    with open('best_drug_food_interaction_model.pkl', 'wb') as f:
        pickle.dump(model_package, f)
    print("✅ Best model saved as 'best_drug_food_interaction_model.pkl'")
except Exception as e:
    print(f"⚠️ Could not save model: {str(e)}")

print("\n🚀 Enhanced Drug-Food Interaction Predictor Complete! 🚀")
