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
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, accuracy_score, precision_score, recall_score, 
                           f1_score, precision_recall_curve, average_precision_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("Installing xgboost...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    import xgboost as xgb
    XGBOOST_AVAILABLE = True

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

# XAI Configuration
XAI_CONFIG = {
    'max_shap_samples': 100,
    'max_lime_features': 10,
    'batch_explanation_limit': 20,
    'dashboard_refresh_samples': 50,
    'enable_global_explanations': True,
    'enable_local_explanations': True,
    'cache_explanations': True
}

print("🚀 ENHANCED DRUG-FOOD INTERACTION PREDICTOR WITH XGBoost")
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

print("\n🤖 PHASE 3: XGBoost MODEL TRAINING")
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

print("🔬 Training XGBoost Model...")
print("-" * 30)

models = {}
results = []

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

print("\n🚀 TRAINING AND EVALUATION")
print("=" * 50)

trained_models = {}

model_name = 'XGBoost'
model = models['XGBoost']
print(f"\n🔧 Training {model_name}...")

try:
    model.fit(X_train, y_train)
    trained_models[model_name] = model
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_pred_proba, model_name)
    results.append(metrics)
    
    print(f"✅ {model_name} Results:")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    print(f"   F1-Score: {metrics['f1']:.4f}")
    print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"   Avg Precision: {metrics['avg_precision']:.4f}")
    
    plot_confusion_matrix(y_test, y_pred, model_name)
    plot_roc_curve(y_test, y_pred_proba, model_name)
    
except Exception as e:
    print(f"❌ Error training {model_name}: {str(e)}")

results_df = pd.DataFrame(results)

print("\n📊 XGBoost RESULTS SUMMARY")
print("=" * 80)
print("Final Results for XGBoost:")
print("-" * 40)
print(results_df.round(4))

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.barh(results_df['model'], results_df['f1'])
plt.xlabel('F1-Score')
plt.title('F1-Score')
plt.gca().invert_yaxis()
plt.subplot(1, 3, 2)
plt.barh(results_df['model'], results_df['accuracy'])
plt.xlabel('Accuracy')
plt.title('Accuracy')
plt.gca().invert_yaxis()
plt.subplot(1, 3, 3)
plt.barh(results_df['model'], results_df['roc_auc'])
plt.xlabel('ROC-AUC')
plt.title('ROC-AUC')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

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
    
    def create_shap_waterfall(self, instance_idx, save_plot=False):
        """Create SHAP waterfall plot for individual prediction"""
        if not self.shap_explainer:
            return None
        
        instance = self.X_test.iloc[instance_idx:instance_idx+1]
        
        try:
            shap_values = self.shap_explainer.shap_values(instance)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Create waterfall plot
            plt.figure(figsize=(10, 8))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=self.shap_explainer.expected_value[1] if isinstance(self.shap_explainer.expected_value, list) else self.shap_explainer.expected_value,
                    data=instance.values[0],
                    feature_names=self.feature_names
                ),
                max_display=10,
                show=False
            )
            
            if save_plot:
                plt.savefig(f'waterfall_plot_{instance_idx}.png', dpi=300, bbox_inches='tight')
            
            return plt.gcf()
        except Exception as e:
            return f"Waterfall plot failed: {e}"

    def create_shap_force_plot(self, instance_idx):
        """Create SHAP force plot for individual prediction"""
        if not self.shap_explainer:
            return None
        
        instance = self.X_test.iloc[instance_idx:instance_idx+1]
        
        try:
            shap_values = self.shap_explainer.shap_values(instance)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            return shap.force_plot(
                self.shap_explainer.expected_value[1] if isinstance(self.shap_explainer.expected_value, list) else self.shap_explainer.expected_value,
                shap_values[0],
                instance.iloc[0],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
        except Exception as e:
            return f"Force plot failed: {e}"
    
    def create_interactive_dashboard_data(self):
        """Create comprehensive data for interactive dashboard"""
        if not self.shap_explainer:
            return {'error': 'SHAP explainer not available'}
        
        try:
            # Sample for dashboard
            dashboard_sample = self.X_test.sample(min(100, len(self.X_test)), random_state=42)
            dashboard_indices = dashboard_sample.index.tolist()
            
            shap_values = self.shap_explainer.shap_values(dashboard_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Create summary data
            summary_data = {
                'feature_importance': pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': np.abs(shap_values).mean(0)
                }).sort_values('importance', ascending=False).head(15).to_dict('records'),
                
                'prediction_distribution': {
                    'high_risk': int(np.sum(self.model.predict_proba(dashboard_sample)[:, 1] > 0.7)),
                    'medium_risk': int(np.sum((self.model.predict_proba(dashboard_sample)[:, 1] > 0.3) & 
                                            (self.model.predict_proba(dashboard_sample)[:, 1] <= 0.7))),
                    'low_risk': int(np.sum(self.model.predict_proba(dashboard_sample)[:, 1] <= 0.3))
                },
                
                'feature_correlations': self.calculate_feature_correlations(),
                'sample_explanations': []
            }
            
            # Add detailed sample explanations
            for i, idx in enumerate(dashboard_indices[:20]):
                original_idx = list(self.X_test.index).index(idx)
                explanation = self.explain_prediction(original_idx)
                summary_data['sample_explanations'].append({
                    'index': idx,
                    'prediction': explanation['predicted_label'],
                    'probability': explanation['probability'],
                    'top_features': explanation['explanations'].get('shap', {}).head(3).to_dict('records') if isinstance(explanation['explanations'].get('shap'), pd.DataFrame) else []
                })
            
            return summary_data
        except Exception as e:
            return {'error': f'Dashboard data creation failed: {str(e)}'}

    def calculate_feature_correlations(self):
        """Calculate feature correlations for dashboard"""
        try:
            corr_matrix = self.X_test.corr()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': float(corr_val)
                        })
            
            return sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)[:10]
        except:
            return []
    
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
    
    def batch_explain_predictions(self, indices_list, max_explanations=10):
        """Explain multiple predictions efficiently for frontend display"""
        explanations = []
        for idx in indices_list[:max_explanations]:
            if idx < len(self.X_test):
                exp = self.explain_prediction(idx)
                explanations.append({
                    'instance': idx,
                    'actual': int(self.y_test.iloc[idx]),
                    **exp
                })
        return explanations
    
    def quantify_prediction_uncertainty(self, instance_idx, n_bootstrap=100):
        """Quantify uncertainty in predictions using bootstrap sampling"""
        instance = self.X_test.iloc[instance_idx:instance_idx+1]
        
        # Bootstrap predictions
        bootstrap_predictions = []
        bootstrap_probabilities = []
        
        for _ in range(n_bootstrap):
            # Sample with replacement from training data
            bootstrap_indices = np.random.choice(len(self.X_train), size=len(self.X_train), replace=True)
            X_bootstrap = self.X_train.iloc[bootstrap_indices]
            y_bootstrap = self.y_test.iloc[bootstrap_indices] if len(self.y_test) > max(bootstrap_indices) else [0] * len(bootstrap_indices)
            
            # Train model on bootstrap sample
            try:
                bootstrap_model = self.model.__class__(**self.model.get_params())
                bootstrap_model.fit(X_bootstrap, y_bootstrap)
                
                pred = bootstrap_model.predict(instance)[0]
                prob = bootstrap_model.predict_proba(instance)[0, 1] if hasattr(bootstrap_model, 'predict_proba') else pred
                
                bootstrap_predictions.append(pred)
                bootstrap_probabilities.append(prob)
            except:
                continue
        
        if len(bootstrap_probabilities) > 0:
            uncertainty_metrics = {
                'mean_probability': float(np.mean(bootstrap_probabilities)),
                'std_probability': float(np.std(bootstrap_probabilities)),
                'confidence_interval_95': [
                    float(np.percentile(bootstrap_probabilities, 2.5)),
                    float(np.percentile(bootstrap_probabilities, 97.5))
                ],
                'prediction_stability': float(np.mean(bootstrap_predictions)),
                'uncertainty_level': 'HIGH' if np.std(bootstrap_probabilities) > 0.2 else 'MEDIUM' if np.std(bootstrap_probabilities) > 0.1 else 'LOW'
            }
            
            return uncertainty_metrics
        
        return {'error': 'Unable to quantify uncertainty'}

    def calibration_analysis(self):
        """Analyze model calibration for uncertainty assessment"""
        if len(self.X_test) < 50:
            return {'error': 'Insufficient data for calibration analysis'}
        
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        y_true = self.y_test
        
        # Bin predictions and calculate calibration
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_data = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                
                calibration_data.append({
                    'bin_lower': float(bin_lower),
                    'bin_upper': float(bin_upper),
                    'accuracy': float(accuracy_in_bin),
                    'confidence': float(avg_confidence_in_bin),
                    'count': int(in_bin.sum())
                })
        
        return {
            'calibration_curve': calibration_data,
            'expected_calibration_error': float(np.mean([abs(d['accuracy'] - d['confidence']) for d in calibration_data if d['count'] > 0]))
        }
    
    def get_feature_impact_summary(self):
        """Get condensed feature impact for dashboard display"""
        importance_df = self.global_feature_importance()
        if isinstance(importance_df, pd.DataFrame):
            return {
                'top_features': importance_df.head(5).to_dict('records'),
                'feature_categories': {
                    'drug_features': len([f for f in importance_df['Feature'] if 'drug_' in f]),
                    'food_features': len([f for f in importance_df['Feature'] if 'food_' in f]),
                    'interaction_features': len([f for f in importance_df['Feature'] if any(x in f for x in ['risk_', 'mechanism_'])])
                }
            }
        return None
    
    def explain_interaction_mechanism(self, drug_category, food_category):
        """Provide detailed mechanism explanation for frontend"""
        mechanism, risk = get_interaction_details(drug_category, food_category)
        
        mechanism_details = {
            'cyp3a4_inhibition': {
                'description': 'Food blocks liver enzymes that metabolize the drug',
                'clinical_impact': 'Increased drug levels, potential toxicity',
                'timing': 'Avoid food 2-4 hours before/after medication'
            },
            'vitamin_k_competition': {
                'description': 'Food contains vitamin K which opposes anticoagulant effects',
                'clinical_impact': 'Reduced blood-thinning effectiveness',
                'timing': 'Maintain consistent dietary vitamin K intake'
            },
            'calcium_chelation': {
                'description': 'Calcium binds to drug, preventing absorption',
                'clinical_impact': 'Reduced drug effectiveness',
                'timing': 'Take medication 1-2 hours before dairy products'
            }
        }
        
        return mechanism_details.get(mechanism, {
            'description': 'Mechanism not fully characterized',
            'clinical_impact': 'Unknown interaction pathway',
            'timing': 'Consult healthcare provider'
        })
    
    def generate_clinical_decision_support(self, drug_name, food_name, patient_context=None):
        """Generate patient-friendly clinical decision support"""
        prediction = self.decision_pathway_analysis(drug_name, food_name)
        
        if 'error' in prediction:
            return prediction
        
        # Patient-friendly explanations
        risk_explanations = {
            'HIGH': {
                'message': '🚨 AVOID COMBINATION - High risk of serious interaction',
                'action': 'Consult your doctor immediately before combining',
                'timeline': 'Do not consume together'
            },
            'MODERATE': {
                'message': '⚠️ CAUTION REQUIRED - Monitor for side effects',
                'action': 'Space timing by 2-4 hours, monitor symptoms',
                'timeline': 'Take medication 2 hours before/after food'
            },
            'LOW': {
                'message': '✅ GENERALLY SAFE - Minimal interaction risk',
                'action': 'Safe to consume together with normal precautions',
                'timeline': 'No special timing required'
            }
        }
        
        risk_info = risk_explanations.get(prediction['risk_level'], risk_explanations['LOW'])
        
        # Mechanism explanations in simple terms
        mechanism_simple = {
            'cyp3a4_inhibition': 'This food can make your medication stronger by blocking how your body breaks it down',
            'vitamin_k_competition': 'This food contains nutrients that can make blood-thinning medication less effective',
            'calcium_chelation': 'Calcium in this food can bind to your medication and reduce how much your body absorbs',
            'absorption_interference': 'This food can slow down how well your body absorbs the medication',
            'unknown': 'The interaction pathway is not fully understood'
        }
        
        clinical_support = {
            'patient_summary': {
                'drug': drug_name.title(),
                'food': food_name.title(),
                'risk_level': prediction['risk_level'],
                'simple_explanation': mechanism_simple.get(prediction['mechanism'], 'Unknown interaction'),
                'recommendation': risk_info
            },
            'professional_details': {
                'mechanism': prediction['mechanism'],
                'drug_category': prediction['drug_category'],
                'food_category': prediction['food_category'],
                'confidence': prediction['confidence']
            },
            'monitoring_advice': self.get_monitoring_advice(prediction['risk_level'], prediction['mechanism']),
            'alternative_suggestions': self.suggest_alternatives(drug_name, food_name)
        }
        
        return clinical_support

    def get_monitoring_advice(self, risk_level, mechanism):
        """Provide specific monitoring advice based on interaction"""
        monitoring_map = {
            'HIGH': {
                'cyp3a4_inhibition': 'Monitor for signs of drug toxicity: unusual fatigue, muscle pain, digestive issues',
                'vitamin_k_competition': 'Monitor INR levels more frequently, watch for unusual bleeding/bruising',
                'default': 'Monitor for any unusual symptoms and contact healthcare provider'
            },
            'MODERATE': {
                'calcium_chelation': 'Monitor effectiveness of medication, may need dosage adjustment',
                'absorption_interference': 'Take medication on empty stomach or 2 hours before meals',
                'default': 'Monitor for reduced medication effectiveness'
            },
            'LOW': {
                'default': 'No special monitoring required, standard precautions apply'
            }
        }
        
        return monitoring_map.get(risk_level, {}).get(mechanism, monitoring_map.get(risk_level, {}).get('default', 'Consult healthcare provider'))

    def suggest_alternatives(self, drug_name, food_name):
        """Suggest safer alternatives for high-risk combinations"""
        # This would be expanded with a comprehensive database
        alternatives = {
            'warfarin': {'foods_to_avoid': ['spinach', 'kale', 'broccoli'], 'safer_options': ['carrots', 'potatoes', 'rice']},
            'simvastatin': {'foods_to_avoid': ['grapefruit', 'pomelo'], 'safer_options': ['orange', 'apple', 'banana']},
            'tetracycline': {'foods_to_avoid': ['milk', 'cheese', 'yogurt'], 'safer_options': ['water', 'juice', 'non-dairy alternatives']}
        }
        
        return alternatives.get(drug_name.lower(), {'message': 'Consult pharmacist for specific alternatives'})
    
    def decision_pathway_analysis(self, drug_name, food_name):
        """Trace decision pathway without recursion"""
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
            
            prediction = self.model.predict(X_new)[0]
            probability = self.model.predict_proba(X_new)[0, 1] if hasattr(self.model, 'predict_proba') else prediction
            
            return {
                'drug': drug_name.title(),
                'food': food_name.title(),
                'prediction': 'INTERACTION' if prediction else 'NO INTERACTION',
                'confidence': f"{probability:.3f}",
                'drug_category': new_df['drug_category'].iloc[0],
                'food_category': new_df['food_category'].iloc[0],
                'mechanism': new_df['mechanism'].iloc[0],
                'risk_level': new_df['risk_level'].iloc[0]
            }
        except Exception as e:
            return {'error': str(e)}
    
    def counterfactual_analysis(self, drug_name, food_name, change_categories=True):
        """Generate 'what-if' scenarios for counterfactual analysis"""
        base_prediction = self.decision_pathway_analysis(drug_name, food_name)
        
        if 'error' in base_prediction:
            return base_prediction
        
        counterfactuals = []
        
        # What if different drug category?
        if change_categories:
            for alt_drug_cat in drug_categories.keys():
                if alt_drug_cat != base_prediction['drug_category']:
                    alt_mechanism, alt_risk = get_interaction_details(alt_drug_cat, base_prediction['food_category'])
                    counterfactuals.append({
                        'scenario': f'If drug was {alt_drug_cat} instead',
                        'original_risk': base_prediction['risk_level'],
                        'counterfactual_risk': alt_risk,
                        'risk_change': 'INCREASED' if alt_risk == 'HIGH' and base_prediction['risk_level'] != 'HIGH' else 'DECREASED' if alt_risk == 'LOW' and base_prediction['risk_level'] != 'LOW' else 'SAME',
                        'mechanism_change': alt_mechanism
                    })
            
            # What if different food category?
            for alt_food_cat in food_categories.keys():
                if alt_food_cat != base_prediction['food_category']:
                    alt_mechanism, alt_risk = get_interaction_details(base_prediction['drug_category'], alt_food_cat)
                    counterfactuals.append({
                        'scenario': f'If food was {alt_food_cat} instead',
                        'original_risk': base_prediction['risk_level'],
                        'counterfactual_risk': alt_risk,
                        'risk_change': 'INCREASED' if alt_risk == 'HIGH' and base_prediction['risk_level'] != 'HIGH' else 'DECREASED' if alt_risk == 'LOW' and base_prediction['risk_level'] != 'LOW' else 'SAME',
                        'mechanism_change': alt_mechanism
                    })
        
        # Sort by risk change impact
        counterfactuals.sort(key=lambda x: {'INCREASED': 2, 'SAME': 1, 'DECREASED': 0}[x['risk_change']], reverse=True)
        
        return {
            'base_scenario': base_prediction,
            'counterfactuals': counterfactuals[:10],  # Top 10 most relevant
            'summary': {
                'safer_alternatives': len([c for c in counterfactuals if c['risk_change'] == 'DECREASED']),
                'riskier_alternatives': len([c for c in counterfactuals if c['risk_change'] == 'INCREASED']),
                'similar_risk': len([c for c in counterfactuals if c['risk_change'] == 'SAME'])
            }
        }

    def feature_counterfactuals(self, instance_idx, target_class=None):
        """Generate feature-level counterfactuals"""
        if not self.shap_explainer:
            return {'error': 'SHAP explainer required for feature counterfactuals'}
        
        instance = self.X_test.iloc[instance_idx:instance_idx+1]
        current_pred = self.model.predict(instance)[0]
        target_class = 1 - current_pred if target_class is None else target_class
        
        try:
            shap_values = self.shap_explainer.shap_values(instance)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Find features that most strongly push toward current prediction
            feature_impacts = pd.DataFrame({
                'feature': self.feature_names,
                'impact': shap_values[0],
                'current_value': instance.iloc[0].values
            })
            
            # Suggest changes to flip prediction
            if target_class != current_pred:
                # Features to change (opposite direction of current impact)
                features_to_change = feature_impacts.nlargest(5, 'impact') if current_pred == 1 else feature_impacts.nsmallest(5, 'impact')
                
                counterfactual_suggestions = []
                for _, row in features_to_change.iterrows():
                    suggestion = {
                        'feature': row['feature'],
                        'current_value': float(row['current_value']),
                        'suggested_direction': 'decrease' if row['impact'] > 0 else 'increase',
                        'impact_magnitude': float(abs(row['impact']))
                    }
                    counterfactual_suggestions.append(suggestion)
                
                return {
                    'current_prediction': 'INTERACTION' if current_pred else 'NO INTERACTION',
                    'target_prediction': 'INTERACTION' if target_class else 'NO INTERACTION',
                    'suggested_changes': counterfactual_suggestions,
                    'feasibility': 'Some changes may not be practically feasible'
                }
        
        except Exception as e:
            return {'error': f'Counterfactual analysis failed: {str(e)}'}

# Initialize XAI system with XGBoost
print("\n🔧 Initializing XAI system with XGBoost...")
xai_system = DrugFoodXAI(
    model=models['XGBoost'],
    X_train=X_train,
    X_test=X_test,
    y_test=y_test,
    feature_names=feature_info['feature_names']
)
xai_system.initialize_explainers()
print("✅ XAI system ready")

def safe_xai_analysis(instance_idx=0):
    """Safely run XAI analysis without infinite loops"""
    if xai_system is None:
        print("❌ XAI system not available")
        return None
    
    try:
        if instance_idx < len(X_test):
            explanation = xai_system.explain_prediction(instance_idx)
            return explanation
        else:
            print(f"❌ Instance index {instance_idx} out of range")
            return None
    except Exception as e:
        print(f"❌ XAI analysis failed: {e}")
        return None

def get_xai_dashboard_data():
    """Get all XAI data needed for frontend dashboard"""
    if xai_system is None:
        return {'error': 'XAI system not available'}
    
    try:
        return {
            'model_info': {
                'best_model': results_df.iloc[0]['model'],
                'performance': results_df.iloc[0].to_dict(),
                'total_features': len(feature_info['feature_names'])
            },
            'feature_importance': xai_system.get_feature_impact_summary(),
            'sample_explanations': xai_system.batch_explain_predictions(list(range(min(5, len(X_test))))),
            'interaction_stats': {
                'high_risk_count': len(df_final[df_final['risk_level'] == 'HIGH']),
                'moderate_risk_count': len(df_final[df_final['risk_level'] == 'MODERATE']),
                'low_risk_count': len(df_final[df_final['risk_level'] == 'LOW'])
            }
        }
    except Exception as e:
        return {'error': f'XAI dashboard generation failed: {str(e)}'}

def create_comprehensive_xai_visualizations(xai_system, save_plots=True):
    """Create comprehensive XAI visualizations"""
    if not xai_system or not xai_system.shap_explainer:
        print("❌ XAI system not available for visualizations")
        return
    
    # 1. Summary plot
    try:
        test_sample = xai_system.X_test.sample(min(100, len(xai_system.X_test)), random_state=42)
        shap_values = xai_system.shap_explainer.shap_values(test_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, test_sample, feature_names=xai_system.feature_names, show=False)
        if save_plots:
            plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Feature importance bar plot
        feature_importance = pd.DataFrame({
            'feature': xai_system.feature_names,
            'importance': np.abs(shap_values).mean(0)
        }).sort_values('importance', ascending=True).tail(15)
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Top 15 Feature Importance (SHAP)')
        plt.tight_layout()
        if save_plots:
            plt.savefig('feature_importance_shap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Dependence plots for top features
        top_features = feature_importance.tail(3)['feature'].tolist()
        for feature in top_features:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(feature, shap_values, test_sample, 
                               feature_names=xai_system.feature_names, show=False)
            if save_plots:
                plt.savefig(f'dependence_plot_{feature}.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print("✅ Comprehensive XAI visualizations created")
        
    except Exception as e:
        print(f"❌ Visualization creation failed: {str(e)}")

def explain_prediction_for_frontend(drug_name, food_name):
    """Complete explanation package for frontend display"""
    prediction = predict_new_interaction(drug_name, food_name)
    
    if 'error' in prediction:
        return prediction
    
    mechanism_details = xai_system.explain_interaction_mechanism(
        prediction['drug_category'], 
        prediction['food_category']
    ) if xai_system else {}
    
    similar = find_similar_interactions(drug_name, food_name, top_n=3)
    
    return {
        **prediction,
        'mechanism_details': mechanism_details,
        'similar_interactions': similar,
        'confidence_level': 'High' if prediction['probability'] > 0.7 else 'Medium' if prediction['probability'] > 0.4 else 'Low',
        'clinical_significance': 'Critical' if prediction['risk_level'] == 'HIGH' else 'Important' if prediction['risk_level'] == 'MODERATE' else 'Monitor'
    }

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

def batch_predict_interactions(drug_list, food_list, max_combinations=100):
    """Efficiently predict multiple drug-food combinations for frontend"""
    results = []
    combinations = [(d, f) for d in drug_list for f in food_list]
    
    combinations = combinations[:max_combinations]
    
    for drug, food in combinations:
        result = predict_new_interaction(drug, food)
        if 'error' not in result:
            results.append({
                'drug': drug,
                'food': food,
                'interaction': result['interaction_predicted'],
                'probability': result['probability'],
                'risk_level': result['risk_level']
            })
    
    return {
        'total_combinations': len(combinations),
        'interactions_found': len([r for r in results if r['interaction']]),
        'results': results
    }

def predict_new_interaction_with_explanation(drug_name, food_name, explain=True):
    result = predict_new_interaction(drug_name, food_name)
    
    if explain and 'error' not in result and xai_system is not None:
        try:
            result['explanation'] = {
                'key_factors': f"Categories: {result['drug_category']} + {result['food_category']}",
                'confidence_level': 'High' if result['probability'] > 0.7 else 'Medium' if result['probability'] > 0.4 else 'Low',
                'mechanism': result['mechanism'],
                'risk_assessment': result['risk_level']
            }
        except Exception as e:
            result['explanation'] = f"Explanation failed: {e}"
    
    return result

def predict_new_interaction(drug_name, food_name, model=None, return_risk=True):
    if model is None:
        model = models['XGBoost']
    
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
        
        prediction = model.predict(X_new)[0]
        probability = model.predict_proba(X_new)[0, 1]
        
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
    result = predict_new_interaction(drug, food)
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
    
    if xai_system is not None:
        importance = xai_system.global_feature_importance()
        if importance is not None and isinstance(importance, pd.DataFrame):
            print("Top 5 Important Features:")
            print(importance.head(5))
        
        if len(X_test) > 0:
            sample_explanation = safe_xai_analysis(0)
            if sample_explanation:
                print(f"Sample prediction: {sample_explanation['predicted_label']}")
    else:
        print("❌ XAI system not available")

test_xai_simple()

# Enhanced XAI Analysis
print("\n🔍 ENHANCED XAI ANALYSIS")
print("=" * 60)

# Test waterfall plots
sample_waterfall = xai_system.create_shap_waterfall(0)
if sample_waterfall and hasattr(sample_waterfall, 'show'):
    plt.show()

# Test uncertainty quantification
uncertainty_result = xai_system.quantify_prediction_uncertainty(0, n_bootstrap=20)
print("Uncertainty Analysis:", uncertainty_result)

# Test counterfactual analysis
counterfactual_result = xai_system.counterfactual_analysis('warfarin', 'spinach')
print("Counterfactual Analysis:")
print(f"  Base risk: {counterfactual_result['base_scenario']['risk_level']}")
print(f"  Safer alternatives: {counterfactual_result['summary']['safer_alternatives']}")

# Test clinical decision support
clinical_support = xai_system.generate_clinical_decision_support('warfarin', 'spinach')
print("Clinical Decision Support:")
print(f"  Risk Level: {clinical_support['patient_summary']['risk_level']}")
print(f"  Recommendation: {clinical_support['patient_summary']['recommendation']['message']}")

# Create comprehensive visualizations
create_comprehensive_xai_visualizations(xai_system, save_plots=True)

# Test interactive dashboard data
dashboard_data = xai_system.create_interactive_dashboard_data()
print("Dashboard Data Generated:", 'error' not in dashboard_data)

print("\n✅ ANALYSIS COMPLETE!")
print("=" * 80)
print("🎯 Key Findings:")
print(f"   • Model: XGBoost")
print(f"   • F1-Score: {results_df.iloc[0]['f1']:.4f}")
print(f"   • ROC-AUC: {results_df.iloc[0]['roc_auc']:.4f}")
print(f"   • Total drug-food pairs analyzed: {len(df_final):,}")
print(f"   • High-risk interactions identified: {len(df_final[df_final['risk_level'] == 'HIGH']):,}")
print(f"   • Feature engineering created {X.shape[1]} features")
print("   • Risk categorization system operational (HIGH/MODERATE/LOW)")

def create_serializable_model_package():
    """Create a model package that can be properly serialized"""
    serializable_package = {
        'model': models['XGBoost'],
        'feature_info': {
            'feature_names': feature_info['feature_names'],
        },
        'scaler': scaler,
        'drug_categories': drug_categories,
        'food_categories': food_categories,
        'high_risk_interactions': high_risk_interactions,
        'model_performance': results_df.iloc[0].to_dict(),
        'training_date': datetime.now().isoformat(),
        'model_version': '2.0_XAI_XGBoost'
    }
    return serializable_package

model_package = create_serializable_model_package()

try:
    with open('xgboost_drug_food_interaction_model.pkl', 'wb') as f:
        pickle.dump(model_package, f)
    print("✅ Model saved as 'xgboost_drug_food_interaction_model.pkl'")
except Exception as e:
    print(f"⚠️ Could not save model: {str(e)}")

print("\n🚀 Enhanced Drug-Food Interaction Predictor with XGBoost Complete! 🚀")
