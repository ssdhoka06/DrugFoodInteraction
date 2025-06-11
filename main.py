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
import streamlit as st

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
    
    df_clean = df.dropna(subset=['drug', 'food'])
    df_clean = df_clean.drop_duplicates(subset=['drug', 'food'])
    
    def clean_text(text):
        text = str(text).lower().strip()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    df_clean['drug'] = df_clean['drug'].apply(clean_text)
    df_clean['food'] = df_clean['food'].apply(clean_text)
    
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

df_clean = load_and_clean_foodrugs()

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

print("\n⚖️ ENHANCED NEGATIVE SAMPLE GENERATION")
print("-" * 40)

def generate_balanced_negatives(df_positive, target_ratio=0.8):
    unique_drugs = df_positive['drug'].unique()
    unique_foods = df_positive['food'].unique()
    
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
    
    while len(negative_samples) < target_negatives and attempts < max_attempts:
        drug = np.random.choice(unique_drugs)
        food = np.random.choice(unique_foods)
        attempts += 1
        
        if (drug, food) not in existing_interactions:
            drug_cat = categorize_entity(drug, drug_categories)
            food_cat = categorize_entity(food, food_categories)
            mechanism, risk_level = get_interaction_details(drug_cat, food_cat)
            
            probability_negative = 0.7
            if mechanism == 'unknown':
                probability_negative = 0.9
            elif risk_level == 'HIGH':
                probability_negative = 0.2
            elif risk_level == 'MODERATE':
                probability_negative = 0.4
            
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

df_negatives = generate_balanced_negatives(df_clean, target_ratio=0.8)
df_final = pd.concat([df_clean, df_negatives], ignore_index=True)

print(f"Final dataset: {len(df_final)} samples")
print(f"Positive interactions: {len(df_clean)}")
print(f"Negative interactions: {len(df_negatives)}")
print(f"Balance ratio: {len(df_negatives)/len(df_clean):.2f}")

print("\n🧠 IMPLEMENTING SPECIALIZED MODELS")
print("-" * 40)

print("\n🔧 PHASE 2: ENHANCED FEATURE ENGINEERING")
print("-" * 40)

def create_enhanced_features(df):
    drug_tfidf = TfidfVectorizer(max_features=50, ngram_range=(1, 2), stop_words='english')
    food_tfidf = TfidfVectorizer(max_features=50, ngram_range=(1, 2), stop_words='english')
    
    drug_dummies = pd.get_dummies(df['drug_category'], prefix='drug').astype(int)
    food_dummies = pd.get_dummies(df['food_category'], prefix='food').astype(int)
    mechanism_dummies = pd.get_dummies(df['mechanism'], prefix='mechanism').astype(int)
    risk_dummies = pd.get_dummies(df['risk_level'], prefix='risk').astype(int)
    
    def create_detailed_risk_score(risk_level, mechanism, drug_category, food_category):
        base_scores = {'HIGH': 4, 'MODERATE': 2, 'LOW': 1}
        base_score = base_scores.get(risk_level, 1)
        
        if mechanism in ['cyp3a4_inhibition', 'vitamin_k_competition']:
            base_score += 1
        if drug_category == 'anticoagulant' and food_category == 'leafy_greens':
            base_score += 2
        
        return min(base_score, 5)
    
    df['risk_score'] = df.apply(
        lambda x: create_detailed_risk_score(x['risk_level'], x['mechanism'], x['drug_category'], x['food_category']), 
        axis=1
    )
    
    print("Creating text-based features...")
    df['combined_text'] = df['drug'] + ' ' + df['food']
    
    try:
        drug_tfidf_features = drug_tfidf.fit_transform(df['drug'])
        food_tfidf_features = food_tfidf.fit_transform(df['food'])
    except ValueError:
        drug_tfidf_features = np.zeros((len(df), 50))
        food_tfidf_features = np.zeros((len(df), 50))
    
    svd_drug = TruncatedSVD(n_components=10, random_state=42)
    svd_food = TruncatedSVD(n_components=10, random_state=42)
    
    try:
        drug_features = svd_drug.fit_transform(drug_tfidf_features)
        food_features = svd_food.fit_transform(food_tfidf_features)
    except ValueError:
        drug_features = np.zeros((len(df), 10))
        food_features = np.zeros((len(df), 10))
    
    drug_text_df = pd.DataFrame(drug_features, columns=[f'drug_text_{i}' for i in range(10)])
    food_text_df = pd.DataFrame(food_features, columns=[f'food_text_{i}' for i in range(10)])
    
    df['drug_length'] = df['drug'].str.len()
    df['food_length'] = df['food'].str.len()
    df['name_similarity'] = df.apply(
        lambda x: len(set(x['drug']) & set(x['food'])) / max(len(set(x['drug']) | set(x['food'])), 1),
        axis=1
    )
    
    df['same_category'] = (df['drug_category'] == df['food_category']).astype(int)
    df['both_other'] = ((df['drug_category'] == 'other') & (df['food_category'] == 'other')).astype(int)
    
    feature_dfs = [
        drug_dummies, food_dummies, mechanism_dummies, risk_dummies,
        df[['risk_score', 'drug_length', 'food_length', 'name_similarity', 'same_category', 'both_other']],
        drug_text_df, food_text_df
    ]
    
    X = pd.concat(feature_dfs, axis=1)
    X = X.astype(float)
    
    feature_info = {
        'drug_tfidf': drug_tfidf,
        'food_tfidf': food_tfidf,
        'svd_drug': svd_drug,
        'svd_food': svd_food,
        'feature_names': list(X.columns)
    }
    
    return X, feature_info

X, feature_info = create_enhanced_features(df_final)
y = df_final['interaction']

print(f"Enhanced feature matrix shape: {X.shape}")
print(f"Total features: {X.shape[1]}")
print(f"Feature types: {X.dtypes.value_counts()}")
print(f"NaN values: {X.isnull().sum().sum()}")
print(f"Infinite values: {np.isinf(X).sum().sum()}")
class_counts = Counter(y)
print(f"Class distribution: {dict(class_counts)}")
print(f"Balance ratio: {class_counts[0]/class_counts[1]:.2f}")

print("\n🤖 PHASE 3: ENHANCED MODEL TRAINING")
print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0)
X_test = X_test.fillna(0).replace([np.inf, -np.inf], 0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

train_counts = Counter(y_train)
weight_ratio = train_counts[0] / train_counts[1] if train_counts[1] > 0 else 1

print(f"Training class distribution: {dict(train_counts)}")
print(f"Class weight ratio: {weight_ratio:.2f}")

def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
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

models = {}
results = []

print("🔬 Training Specified Models...")
print("-" * 30)

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

cb_model = cb.CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    depth=8,
    class_weights=[1, weight_ratio],
    random_seed=42,
    verbose=False
)
models['CatBoost'] = cb_model

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

print("\n🚀 TRAINING AND EVALUATION")
print("=" * 50)

trained_models = {}

for model_name, model in models.items():
    print(f"\n🔧 Training {model_name}...")
    
    try:
        if model_name in ['MLP']:
            X_train_use = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            X_test_use = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        else:
            X_train_use = X_train
            X_test_use = X_test
        
        model.fit(X_train_use, y_train)
        trained_models[model_name] = model
        
        y_pred = model.predict(X_test_use)
        
        try:
            y_pred_proba = model.predict_proba(X_test_use)[:, 1]
        except:
            try:
                y_pred_proba = model.decision_function(X_test_use)
                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
            except:
                y_pred_proba = y_pred.astype(float)
        
        metrics = evaluate_model(y_test, y_pred, y_pred_proba, model_name)
        results.append(metrics)
        
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

print(f"\n🤝 Training Voting Classifier...")
print("-" * 30)

voting_models = []
for model_name in ['Random Forest', 'Extra Trees', 'LightGBM', 'CatBoost']:
    if model_name in trained_models:
        voting_models.append((model_name.replace(' ', '_'), trained_models[model_name]))

if 'XGBoost' in trained_models:
    voting_models.append(('XGBoost', trained_models['XGBoost']))

valid_voting_models = []
for name, model in voting_models:
    if hasattr(model, 'predict_proba'):
        valid_voting_models.append((name, model))

if len(valid_voting_models) >= 2:
    try:
        voting_clf = VotingClassifier(
            estimators=valid_voting_models,
            voting='soft'
        )
        
        voting_clf.fit(X_train, y_train)
        
        y_pred_voting = voting_clf.predict(X_test)
        y_pred_proba_voting = voting_clf.predict_proba(X_test)[:, 1]
        
        voting_metrics = evaluate_model(y_test, y_pred_voting, y_pred_proba_voting, 'Voting Classifier')
        results.append(voting_metrics)
        
        print(f"✅ Voting Classifier Results:")
        print(f"   Accuracy: {voting_metrics['accuracy']:.4f}")
        print(f"   Precision: {voting_metrics['precision']:.4f}")
        print(f"   Recall: {voting_metrics['recall']:.4f}")
        print(f"   F1-Score: {voting_metrics['f1']:.4f}")
        print(f"   ROC-AUC: {voting_metrics['roc_auc']:.4f}")
        print(f"   Avg Precision: {voting_metrics['avg_precision']:.4f}")
        
        models['Voting Classifier'] = voting_clf
        
    except Exception as e:
        print(f"❌ Error training Voting Classifier: {str(e)}")
else:
    print("❌ Not enough models with predict_proba for Voting Classifier")

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

plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.barh(results_df['model'], results_df['f1'])
plt.xlabel('F1-Score')
plt.title('F1-Score Comparison')
plt.gca().invert_yaxis()
plt.subplot(2, 3, 2)
plt.barh(results_df['model'], results_df['accuracy'])
plt.xlabel('Accuracy')
plt.title('Accuracy Comparison')
plt.gca().invert_yaxis()
plt.subplot(2, 3, 3)
plt.barh(results_df['model'], results_df['roc_auc'])
plt.xlabel('ROC-AUC')
plt.title('ROC-AUC Comparison')
plt.gca().invert_yaxis()
plt.subplot(2, 3, 4)
plt.barh(results_df['model'], results_df['precision'])
plt.xlabel('Precision')
plt.title('Precision Comparison')
plt.gca().invert_yaxis()
plt.subplot(2, 3, 5)
plt.barh(results_df['model'], results_df['recall'])
plt.xlabel('Recall')
plt.title('Recall Comparison')
plt.gca().invert_yaxis()
plt.subplot(2, 3, 6)
plt.barh(results_df['model'], results_df['avg_precision'])
plt.xlabel('Average Precision')
plt.title('Average Precision Comparison')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\n🎯 CONFUSION MATRICES FOR TOP 3 MODELS")
print("=" * 50)

top_3_models = results_df.head(3)['model'].tolist()

for model_name in top_3_models:
    if model_name in models:
        model = models[model_name]
        if model_name == 'MLP':
            X_test_use = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        else:
            X_test_use = X_test
        try:
            y_pred = model.predict(X_test_use)
            plot_confusion_matrix(y_test, y_pred, model_name)
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

print("\n🔍 PHASE 4: EXPLAINABLE AI (XAI) IMPLEMENTATION")
print("-" * 50)

class DrugFoodXAI:
    """Optimized XAI analysis for drug-food interactions with Streamlit compatibility"""
    
    def __init__(self, model, X_train, X_test, y_test, feature_names):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.shap_explainer = None
        self.lime_explainer = None
    
    @st.cache_resource
    def initialize_explainers(self):
        """Initialize SHAP and LIME explainers with caching for performance"""
        print("🔧 Initializing XAI explainers...")
        
        if hasattr(self.model, 'predict_proba'):
            try:
                background_data = self.X_train.sample(min(50, len(self.X_train)), random_state=42)
                self.shap_explainer = shap.TreeExplainer(self.model, background_data)
                print("✅ SHAP TreeExplainer initialized")
            except:
                try:
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict_proba, 
                        self.X_train.sample(min(25, len(self.X_train)), random_state=42)
                    )
                    print("✅ SHAP KernelExplainer initialized")
                except Exception as e:
                    print(f"⚠️ SHAP initialization failed: {e}")
        
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
    
    def explain_prediction(self, instance_idx):
        """Explain a specific prediction with compact output for Streamlit"""
        instance = self.X_test.iloc[instance_idx:instance_idx+1]
        pred_label = self.model.predict(instance)[0]
        pred_proba = self.model.predict_proba(instance)[0, 1] if hasattr(self.model, 'predict_proba') else pred_label
        
        explanations = {}
        
        if self.shap_explainer:
            try:
                shap_values = self.shap_explainer.shap_values(instance)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                shap_df = pd.DataFrame({
                    'Feature': self.feature_names,
                    'SHAP Value': shap_values[0]
                }).sort_values('SHAP Value', key=abs, ascending=False).head(5)
                
                explanations['shap'] = shap_df
            except Exception as e:
                explanations['shap'] = f"SHAP explanation failed: {e}"
        
        if self.lime_explainer:
            try:
                lime_exp = self.lime_explainer.explain_instance(
                    instance.values[0],
                    self.model.predict_proba,
                    num_features=5
                )
                explanations['lime'] = pd.DataFrame(lime_exp.as_list(), columns=['Feature', 'LIME Weight'])
            except Exception as e:
                explanations['lime'] = f"LIME explanation failed: {e}"
        
        return {
            'predicted_label': 'INTERACTION' if pred_label else 'NO INTERACTION',
            'probability': float(pred_proba),
            'explanations': explanations
        }
    
    def global_feature_importance(self):
        """Global feature importance analysis optimized for Streamlit"""
        if not self.shap_explainer:
            return None
        
        try:
            test_subset = self.X_test.sample(min(50, len(self.X_test)), random_state=42)
            shap_values = self.shap_explainer.shap_values(test_subset)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            mean_shap = np.abs(shap_values).mean(0)
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': mean_shap
            }).sort_values('Importance', ascending=False).head(10)
            
            return importance_df
        except Exception as e:
            return f"Global analysis failed: {e}"
    
    def decision_pathway_analysis(self, drug_name, food_name):
        """Trace decision pathway with single, concise output"""
        result = predict_new_interaction(drug_name, food_name)
        if 'error' in result:
            return result
        
        # Consolidated output
        output = {
            'drug': drug_name.title(),
            'food': food_name.title(),
            'prediction': 'INTERACTION' if result['interaction_predicted'] else 'NO INTERACTION',
            'confidence': f"{result['probability']:.3f}",
            'drug_category': result['drug_category'],
            'food_category': result['food_category'],
            'mechanism': result['mechanism'],
            'risk_level': result['risk_level']
        }
        
        return output

xai_system = DrugFoodXAI(
    model=models[results_df.iloc[0]['model']],
    X_train=X_train,
    X_test=X_test,
    y_test=y_test,
    feature_names=feature_info['feature_names']
)

xai_system.initialize_explainers()

def conduct_case_studies():
    case_studies = [
        {'name': 'Warfarin-Spinach Interaction', 'drug': 'warfarin', 'food': 'spinach', 'description': 'Classic vitamin K antagonist interaction'},
        {'name': 'Statin-Grapefruit Interaction', 'drug': 'simvastatin', 'food': 'grapefruit', 'description': 'CYP3A4 enzyme inhibition leading to toxicity'},
        {'name': 'Antibiotic-Dairy Interaction', 'drug': 'tetracycline', 'food': 'milk', 'description': 'Calcium chelation reducing absorption'}
    ]
    
    results = []
    for i, case in enumerate(case_studies):
        result = xai_system.decision_pathway_analysis(case['drug'], case['food'])
        if 'error' not in result:
            result['case_name'] = case['name']
            result['description'] = case['description']
            results.append(result)
    
    return results

def find_similar_interactions(target_drug, target_food, top_n=5):
    target_drug_cat = categorize_entity(target_drug, drug_categories)
    target_food_cat = categorize_entity(target_food, food_categories)
    
    similar = df_final[
        (df_final['drug_category'] == target_drug_cat) & 
        (df_final['food_category'] == target_food_cat)
    ].head(top_n)
    
    return similar[['drug', 'food', 'interaction', 'risk_level']].to_dict('records')

def predict_new_interaction_with_explanation(drug_name, food_name, explain=True):
    result = predict_new_interaction(drug_name, food_name)
    
    if explain and 'error' not in result:
        try:
            temp_df = pd.DataFrame({
                'drug': [drug_name.lower()],
                'food': [food_name.lower()],
                'interaction': [0]
            })
            
            temp_df['drug_category'] = temp_df['drug'].apply(lambda x: categorize_entity(x, drug_categories))
            temp_df['food_category'] = temp_df['food'].apply(lambda x: categorize_entity(x, food_categories))
            
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
    if model is None:
        model = models[results_df.iloc[0]['model']]
    
    new_df = pd.DataFrame({
        'drug': [drug_name.lower().strip()],
        'food': [food_name.lower().strip()],
        'interaction': [0]
    })
    
    new_df['drug_category'] = new_df['drug'].apply(lambda x: categorize_entity(x, drug_categories))
    new_df['food_category'] = new_df['food'].apply(lambda x: categorize_entity(x, food_categories))
    new_df[['mechanism', 'risk_level']] = new_df.apply(
        lambda x: pd.Series(get_interaction_details(x['drug_category'], x['food_category'])), 
        axis=1
    )
    
    try:
        X_new, _ = create_enhanced_features(new_df)
        missing_cols = set(feature_info['feature_names']) - set(X_new.columns)
        for col in missing_cols:
            X_new[col] = 0
        X_new = X_new[feature_info['feature_names']]
        
        if model.__class__.__name__ == 'MLPClassifier':
            X_new = scaler.transform(X_new)
        
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
    base_result = predict_new_interaction_with_explanation(drug_name, food_name)
    
    risk_multiplier = 1.0
    if age and age > 65:
        risk_multiplier += 0.2
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

print("\n🔮 EXAMPLE PREDICTIONS")
print("=" * 40)

test_pairs = [
    ('warfarin', 'spinach'),
    ('simvastatin', 'grapefruit'),
    ('aspirin', 'alcohol'),
    ('amoxicillin', 'milk'),
    ('metformin', 'banana'),
    ('ibuprofen', 'coffee')
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

print("\n🔍 EXECUTING COMPREHENSIVE XAI ANALYSIS")
print("=" * 60)

def test_xai_simple():
    print("\n🔍 SIMPLE XAI TEST")
    print("-" * 30)
    
    test_pairs = [('warfarin', 'spinach'), ('aspirin', 'coffee')]
    
    for drug, food in test_pairs:
        result = predict_new_interaction_with_explanation(drug, food, explain=False)
        if 'error' not in result:
            print(f"{drug} + {food}: {'INTERACTION' if result['interaction_predicted'] else 'NO INTERACTION'} (p={result['probability']:.3f})")

test_xai_simple()

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

model_package = {
    'model': models[results_df.iloc[0]['model']],
    'feature_info': feature_info,
    'scaler': scaler,
    'drug_categories': drug_categories,
    'food_categories': food_categories,
    'high_risk_interactions': high_risk_interactions,
    'model_performance': results_df.iloc[0].to_dict(),
    'training_date': datetime.now().isoformat(),
    'model_version': '2.0_XAI',
    'xai_system': xai_system,
    'xai_capabilities': ['SHAP', 'LIME', 'Decision_Pathways', 'Case_Studies']
}

try:
    with open('best_drug_food_interaction_model.pkl', 'wb') as f:
        pickle.dump(model_package, f)
    print("✅ Best model saved as 'best_drug_food_interaction_model.pkl'")
except Exception as e:
    print(f"⚠️ Could not save model: {str(e)}")

print("\n🚀 Enhanced Drug-Food Interaction Predictor Complete! 🚀")
