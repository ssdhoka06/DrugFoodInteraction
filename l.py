
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, accuracy_score, precision_score, recall_score, f1_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import xgboost as xgb
from collections import Counter
import pickle
import warnings
warnings.filterwarnings('ignore')

# BioBERT and Transformers imports
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import subprocess
import sys

# Set up device for M3 GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if device.type == "mps":
    torch.mps.manual_seed(42)

# Dataset path
df2 = '/Users/sachidhoka/Desktop/food-drug interactions.csv'

class BioBERTEmbedder:
    """BioBERT embedder for drug and food entities"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.tokenizer = None
        self.model = None
        self.load_models()
    
    def install_requirements(self):
        """Install required transformers package"""
        try:
            import transformers
            print("✅ Transformers already installed")
            return True
        except ImportError:
            print("📦 Installing transformers...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'transformers'])
                print("✅ Transformers installed successfully")
                return True
            except Exception as e:
                print(f"❌ Failed to install transformers: {e}")
                return False
    
    def load_models(self):
        """Load BioBERT models with fallback options"""
        print("🧠 Loading BioBERT models...")
        
        if not self.install_requirements():
            raise RuntimeError("Failed to install required packages")
        
        try:
            biobert_models = [
                ('dmis-lab/biobert-base-cased-v1.1', 'BioBERT-v1.1'),
                ('dmis-lab/biobert-v1.1-pubmed', 'BioBERT-PubMed'),
                ('allenai/scibert_scivocab_uncased', 'SciBERT'),
                ('bert-base-uncased', 'BERT-base')
            ]
            
            for model_name, model_type in biobert_models:
                try:
                    print(f"🔄 Loading {model_type}...")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModel.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32
                    )
                    self.model.eval()
                    self.model.to(self.device)
                    print(f"✅ {model_type} loaded successfully")
                    self.model_name = model_type
                    break
                except Exception as e:
                    print(f"❌ {model_type} failed: {e}")
                    continue
            else:
                raise RuntimeError("All BioBERT models failed to load")
            
        except Exception as e:
            print(f"❌ Critical error loading models: {e}")
            raise
    
    def get_embeddings(self, texts, batch_size=32):
        """Get BioBERT embeddings for texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def get_similarity_features(self, drug_embeddings, food_embeddings):
        """Calculate similarity features between drug and food embeddings"""
        cos_sim = np.sum(drug_embeddings * food_embeddings, axis=1) / (
            np.linalg.norm(drug_embeddings, axis=1) * np.linalg.norm(food_embeddings, axis=1)
        )
        euclidean_dist = np.linalg.norm(drug_embeddings - food_embeddings, axis=1)
        manhattan_dist = np.sum(np.abs(drug_embeddings - food_embeddings), axis=1)
        dot_product = np.sum(drug_embeddings * food_embeddings, axis=1)
        
        return np.column_stack([cos_sim, euclidean_dist, manhattan_dist, dot_product])

print("🚀 STARTING DRUG-FOOD INTERACTION PREDICTOR WITH BIOBERT")
print("=" * 70)

# Initialize BioBERT embedder
biobert_embedder = BioBERTEmbedder(device=device)

# Data preprocessing
print("\n📋 DATA PREPROCESSING")
print("-" * 40)

def load_and_clean_foodrugs(filepath):
    """Load and clean FooDrugs dataset"""
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
        print("⚠️ FooDrugs file not found. Creating sample data...")
        drugs = ['warfarin', 'simvastatin', 'tetracycline', 'aspirin', 'metformin', 
                'lisinopril', 'sertraline', 'digoxin', 'amoxicillin', 'atorvastatin',
                'ibuprofen', 'omeprazole', 'losartan', 'metoprolol', 'fluoxetine'] * 50
        foods = ['spinach', 'grapefruit', 'milk', 'alcohol', 'bread', 'banana', 
                'coffee', 'cheese', 'broccoli', 'orange', 'yogurt', 'kale',
                'wine', 'tea', 'avocado'] * 50
        
        np.random.shuffle(drugs)
        np.random.shuffle(foods)
        
        sample_data = {'drug': drugs, 'food': foods}
        df = pd.DataFrame(sample_data)
    
    print(f"Original dataset size: {len(df)}")
    
    df_clean = df.dropna(subset=['drug', 'food'])
    df_clean = df_clean.drop_duplicates(subset=['drug', 'food'])
    
    df_clean['drug'] = df_clean['drug'].astype(str).str.lower().str.strip()
    df_clean['food'] = df_clean['food'].astype(str).str.lower().str.strip()
    
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
df_clean = load_and_clean_foodrugs(df2)

# Display sample data
print("\nSample interactions:")
print(df_clean.head(10))

# Knowledge base
print("\n🧠 Creating knowledge base...")
drug_categories = {
    'anticoagulant': ['warfarin', 'heparin', 'coumadin', 'dabigatran', 'rivaroxaban'],
    'statin': ['simvastatin', 'atorvastatin', 'lovastatin', 'rosuvastatin', 'pravastatin'],
    'antibiotic': ['amoxicillin', 'penicillin', 'tetracycline', 'doxycycline', 'ciprofloxacin', 'azithromycin'],
    'antihypertensive': ['lisinopril', 'amlodipine', 'losartan', 'metoprolol', 'hydrochlorothiazide'],
    'antidepressant': ['sertraline', 'fluoxetine', 'paroxetine', 'citalopram', 'escitalopram'],
    'diabetes': ['metformin', 'glipizide', 'insulin', 'glyburide', 'pioglitazone'],
    'pain_relief': ['aspirin', 'ibuprofen', 'acetaminophen', 'naproxen', 'celecoxib'],
    'heart_rhythm': ['digoxin', 'amiodarone', 'flecainide', 'propafenone'],
    'ppi': ['omeprazole', 'lansoprazole', 'pantoprazole', 'esomeprazole']
}

food_categories = {
    'citrus': ['grapefruit', 'orange', 'lemon', 'lime', 'tangerine'],
    'dairy': ['milk', 'cheese', 'yogurt', 'calcium', 'ice cream'],
    'alcohol': ['alcohol', 'ethanol', 'wine', 'beer', 'spirits'],
    'leafy_greens': ['spinach', 'kale', 'lettuce', 'broccoli', 'brussels sprouts'],
    'high_potassium': ['banana', 'potassium', 'avocado', 'orange juice', 'coconut water'],
    'high_sodium': ['salt', 'sodium', 'processed foods', 'pickles', 'soup'],
    'caffeinated': ['coffee', 'tea', 'caffeine', 'cola', 'energy drinks'],
    'fiber_rich': ['bran', 'oats', 'fiber', 'whole grains', 'beans']
}

high_risk_interactions = {
    ('anticoagulant', 'leafy_greens'): 'vitamin_k_competition',
    ('statin', 'citrus'): 'cyp3a4_inhibition',
    ('antibiotic', 'dairy'): 'calcium_chelation',
    ('antidepressant', 'alcohol'): 'cns_depression',
    ('diabetes', 'alcohol'): 'hypoglycemia_risk',
    ('heart_rhythm', 'high_potassium'): 'arrhythmia_risk',
    ('pain_relief', 'alcohol'): 'gi_bleeding_risk'
}

def categorize_entity(entity, categories):
    """Categorize drug or food based on keywords"""
    entity_lower = str(entity).lower()
    for category, items in categories.items():
        if any(item in entity_lower for item in items):
            return category
    return 'other'

def get_interaction_mechanism(drug_cat, food_cat):
    """Get interaction mechanism based on categories"""
    for (d_cat, f_cat), mechanism in high_risk_interactions.items():
        if d_cat == drug_cat and f_cat == food_cat:
            return mechanism
    return 'unknown'

# Apply categorization
print("Categorizing drugs and foods...")
df_clean['drug_category'] = df_clean['drug'].apply(lambda x: categorize_entity(x, drug_categories))
df_clean['food_category'] = df_clean['food'].apply(lambda x: categorize_entity(x, food_categories))
df_clean['mechanism'] = df_clean.apply(
    lambda x: get_interaction_mechanism(x['drug_category'], x['food_category']), 
    axis=1
)

# Generate negative samples
print("\n⚖️ Generating negative samples...")
unique_drugs = df_clean['drug'].unique()
unique_foods = df_clean['food'].unique()

if len(unique_drugs) > 1000:
    unique_drugs = np.random.choice(unique_drugs, 1000, replace=False)
if len(unique_foods) > 1000:
    unique_foods = np.random.choice(unique_foods, 1000, replace=False)

existing_interactions = set(zip(df_clean['drug'], df_clean['food']))
negative_samples = []
max_negatives = len(df_clean)
attempts = 0
max_attempts = max_negatives * 10

while len(negative_samples) < max_negatives and attempts < max_attempts:
    drug = np.random.choice(unique_drugs)
    food = np.random.choice(unique_foods)
    attempts += 1
    
    if (drug, food) not in existing_interactions:
        drug_cat = categorize_entity(drug, drug_categories)
        food_cat = categorize_entity(food, food_categories)
        mechanism = get_interaction_mechanism(drug_cat, food_cat)
        
        if mechanism == 'unknown' and drug_cat != food_cat:
            negative_samples.append({
                'drug': drug,
                'food': food,
                'interaction': 0,
                'drug_category': drug_cat,
                'food_category': food_cat,
                'mechanism': mechanism
            })

df_negatives = pd.DataFrame(negative_samples)
df_final = pd.concat([df_clean, df_negatives], ignore_index=True)

print(f"Final dataset: {len(df_final)} samples")
print(f"Positive interactions: {len(df_clean)}")
print(f"Negative interactions: {len(df_negatives)}")

# Feature engineering
print("\n🔧 BIOBERT FEATURE ENGINEERING")
print("-" * 40)

df_final['drug_context'] = df_final['drug'].apply(
    lambda x: f"pharmaceutical drug medication {x} therapeutic agent"
)
df_final['food_context'] = df_final['food'].apply(
    lambda x: f"food nutrition dietary item {x} consumable"
)

print("📊 Computing drug embeddings...")
drug_embeddings = biobert_embedder.get_embeddings(df_final['drug_context'].tolist())
print(f"Drug embeddings shape: {drug_embeddings.shape}")

print("📊 Computing food embeddings...")
food_embeddings = biobert_embedder.get_embeddings(df_final['food_context'].tolist())
print(f"Food embeddings shape: {food_embeddings.shape}")

print("📊 Computing similarity features...")
similarity_features = biobert_embedder.get_similarity_features(drug_embeddings, food_embeddings)

def create_risk_score(drug_cat, food_cat, mechanism):
    """Create risk score based on known interactions"""
    if mechanism != 'unknown':
        return 3
    elif drug_cat != 'other' and food_cat != 'other':
        return 2
    else:
        return 1

df_final['risk_score'] = df_final.apply(
    lambda x: create_risk_score(x['drug_category'], x['food_category'], x['mechanism']), 
    axis=1
)

drug_dummies = pd.get_dummies(df_final['drug_category'], prefix='drug').astype(int)
food_dummies = pd.get_dummies(df_final['food_category'], prefix='food').astype(int)
mechanism_dummies = pd.get_dummies(df_final['mechanism'], prefix='mechanism').astype(int)

pca_drug = PCA(n_components=50)
pca_food = PCA(n_components=50)

drug_embeddings_reduced = pca_drug.fit_transform(drug_embeddings)
food_embeddings_reduced = pca_food.fit_transform(food_embeddings)

biobert_drug_reduced_df = pd.DataFrame(drug_embeddings_reduced, columns=[f'drug_pca_{i}' for i in range(50)])
biobert_food_reduced_df = pd.DataFrame(food_embeddings_reduced, columns=[f'food_pca_{i}' for i in range(50)])
similarity_df = pd.DataFrame(similarity_features, columns=['cosine_sim', 'euclidean_dist', 'manhattan_dist', 'dot_product'])

X = pd.concat([
    biobert_drug_reduced_df,
    biobert_food_reduced_df,
    similarity_df,
    drug_dummies, 
    food_dummies, 
    mechanism_dummies, 
    df_final[['risk_score']]
], axis=1)

y = df_final['interaction']
feature_cols = list(X.columns)
X = X.astype(float)

print(f"Final feature matrix shape: {X.shape}")

# Model training
print("\n🤖 MODEL TRAINING")
print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_counts = Counter(y_train)
weight_ratio = train_counts[0] / train_counts[1] if train_counts[1] > 0 else 1

def calculate_metrics(y_true, y_pred, y_pred_proba, model_name):
    """Calculate comprehensive metrics for a model"""
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)
    }
    return metrics

print("\n🚀 Training XGBoost model...")
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.03,
    scale_pos_weight=weight_ratio,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)

xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

print("🌲 Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]

print("🚀 Training Gradient Boosting model...")
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_pred_proba = gb_model.predict_proba(X_test)[:, 1]

xgb_metrics = calculate_metrics(y_test, xgb_pred, xgb_pred_proba, "XGBoost + BioBERT")
rf_metrics = calculate_metrics(y_test, rf_pred, rf_pred_proba, "Random Forest + BioBERT")
gb_metrics = calculate_metrics(y_test, gb_pred, gb_pred_proba, "Gradient Boosting + BioBERT")

metrics_df = pd.DataFrame([xgb_metrics, rf_metrics, gb_metrics])
print("\n📊 MODEL COMPARISON")
print(metrics_df.round(4).to_string(index=False))

best_model_idx = metrics_df['f1_score'].idxmax()
best_model_name = metrics_df.loc[best_model_idx, 'model_name']
best_f1_score = metrics_df.loc[best_model_idx, 'f1_score']

if "XGBoost" in best_model_name:
    best_model, best_pred, best_pred_proba = xgb_model, xgb_pred, xgb_pred_proba
elif "Random Forest" in best_model_name:
    best_model, best_pred, best_pred_proba = rf_model, rf_pred, rf_pred_proba
else:
    best_model, best_pred, best_pred_proba = gb_model, gb_pred, gb_pred_proba

print(f"\n🏆 BEST MODEL: {best_model_name} (F1-Score: {best_f1_score:.4f})")

# Save model
model_package = {
    'model': best_model,
    'feature_columns': feature_cols,
    'pca_drug': pca_drug,
    'pca_food': pca_food,
    'model_name': best_model_name,
    'drug_categories': drug_categories,
    'food_categories': food_categories,
    'high_risk_interactions': high_risk_interactions
}

with open('/Users/sachidhoka/Desktop/biobert_drug_food_interaction_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print("✅ Model saved to '/Users/sachidhoka/Desktop/biobert_drug_food_interaction_model.pkl'")
