import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, accuracy_score, precision_score, recall_score, 
                           f1_score, precision_recall_curve, average_precision_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
import catboost as cb
from collections import Counter
import pickle
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

print("🚀 ENHANCED DRUG-FOOD INTERACTION PREDICTOR")
print("=" * 60)

# ENHANCED DATA PREPROCESSING
print("\n📋 PHASE 1: ENHANCED DATA PREPROCESSING")
print("-" * 40)

def load_and_clean_foodrugs(filepath=None):
    """Load and clean FooDrugs dataset with enhanced preprocessing"""
    print("Loading FooDrugs dataset...")
    
    if filepath is None:
        filepath = '/content/drive/MyDrive/ASEP_2/food-drug interactions.csv'
    
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
        print("⚠️ FooDrugs file not found. Creating enhanced sample data...")
        # Create comprehensive sample data
        drugs = ['warfarin', 'simvastatin', 'tetracycline', 'aspirin', 'metformin', 
                'lisinopril', 'sertraline', 'digoxin', 'amoxicillin', 'atorvastatin',
                'ibuprofen', 'omeprazole', 'losartan', 'metoprolol', 'fluoxetine',
                'amlodipine', 'levothyroxine', 'prednisone', 'gabapentin', 'tramadol'] * 100
        
        foods = ['spinach', 'grapefruit', 'milk', 'alcohol', 'bread', 'banana', 
                'coffee', 'cheese', 'broccoli', 'orange', 'yogurt', 'kale',
                'wine', 'tea', 'avocado', 'salt', 'calcium', 'fiber', 'beans', 'nuts'] * 100
        
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

# Enhanced knowledge base
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

# Enhanced interaction mechanisms
high_risk_interactions = {
    ('anticoagulant', 'leafy_greens'): 'vitamin_k_competition',
    ('statin', 'citrus'): 'cyp3a4_inhibition',
    ('antibiotic', 'dairy'): 'calcium_chelation',
    ('antidepressant', 'alcohol'): 'cns_depression',
    ('diabetes', 'alcohol'): 'hypoglycemia_risk',
    ('heart_rhythm', 'high_potassium'): 'arrhythmia_risk',
    ('pain_relief', 'alcohol'): 'gi_bleeding_risk',
    ('thyroid', 'fiber_rich'): 'absorption_interference',
    ('ppi', 'caffeinated'): 'acid_suppression_interference'
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

def get_interaction_mechanism(drug_cat, food_cat):
    """Get interaction mechanism"""
    for (d_cat, f_cat), mechanism in high_risk_interactions.items():
        if d_cat == drug_cat and f_cat == food_cat:
            return mechanism
    return 'unknown'

# Apply categorization
print("🏷️ Categorizing drugs and foods...")
df_clean['drug_category'] = df_clean['drug'].apply(lambda x: categorize_entity(x, drug_categories))
df_clean['food_category'] = df_clean['food'].apply(lambda x: categorize_entity(x, food_categories))
df_clean['mechanism'] = df_clean.apply(
    lambda x: get_interaction_mechanism(x['drug_category'], x['food_category']), 
    axis=1
)

print("\nDrug category distribution:")
print(df_clean['drug_category'].value_counts().head(10))
print("\nFood category distribution:")
print(df_clean['food_category'].value_counts().head(10))

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
            mechanism = get_interaction_mechanism(drug_cat, food_cat)
            
            # More lenient negative sampling - include some potential interactions as negatives
            # to make the model learn nuanced differences
            probability_negative = 0.7  # 70% chance to include as negative
            
            if mechanism == 'unknown':
                probability_negative = 0.9  # Higher chance for unknown mechanisms
            elif mechanism in ['vitamin_k_competition', 'cyp3a4_inhibition', 'calcium_chelation']:
                probability_negative = 0.3  # Lower chance for known high-risk mechanisms
            
            if np.random.random() < probability_negative:
                negative_samples.append({
                    'drug': drug,
                    'food': food,
                    'interaction': 0,
                    'drug_category': drug_cat,
                    'food_category': food_cat,
                    'mechanism': mechanism
                })
    
    return pd.DataFrame(negative_samples)

# Generate balanced negatives
df_negatives = generate_balanced_negatives(df_clean, target_ratio=0.8)
df_final = pd.concat([df_clean, df_negatives], ignore_index=True)

print(f"Final dataset: {len(df_final)} samples")
print(f"Positive interactions: {len(df_clean)}")
print(f"Negative interactions: {len(df_negatives)}")
print(f"Balance ratio: {len(df_negatives)/len(df_clean):.2f}")

# ENHANCED FEATURE ENGINEERING
print("\n🔧 PHASE 2: ENHANCED FEATURE ENGINEERING")
print("-" * 40)

def create_enhanced_features(df):
    """Create comprehensive feature set"""
    
    # 1. Basic categorical features
    drug_dummies = pd.get_dummies(df['drug_category'], prefix='drug').astype(int)
    food_dummies = pd.get_dummies(df['food_category'], prefix='food').astype(int)
    mechanism_dummies = pd.get_dummies(df['mechanism'], prefix='mechanism').astype(int)
    
    # 2. Risk scoring features
    def create_risk_score(drug_cat, food_cat, mechanism):
        if mechanism != 'unknown':
            return 3  # High risk
        elif drug_cat != 'other' and food_cat != 'other':
            return 2  # Medium risk
        else:
            return 1  # Low risk
    
    df['risk_score'] = df.apply(
        lambda x: create_risk_score(x['drug_category'], x['food_category'], x['mechanism']), 
        axis=1
    )
    
    # 3. Text-based features using TF-IDF
    print("Creating text-based features...")
    
    # Combine drug and food names for text analysis
    df['combined_text'] = df['drug'] + ' ' + df['food']
    
    # TF-IDF for drug names
    drug_tfidf = TfidfVectorizer(max_features=50, ngram_range=(1, 2), stop_words='english')
    drug_tfidf_features = drug_tfidf.fit_transform(df['drug'])
    
    # TF-IDF for food names
    food_tfidf = TfidfVectorizer(max_features=50, ngram_range=(1, 2), stop_words='english')
    food_tfidf_features = food_tfidf.fit_transform(df['food'])
    
    # Reduce dimensionality
    svd_drug = TruncatedSVD(n_components=10, random_state=42)
    svd_food = TruncatedSVD(n_components=10, random_state=42)
    
    drug_features = svd_drug.fit_transform(drug_tfidf_features)
    food_features = svd_food.fit_transform(food_tfidf_features)
    
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
        drug_dummies, food_dummies, mechanism_dummies,
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

# Train models with enhanced parameters
models = {}
results = []

# 1. Enhanced LightGBM
print("🚀 Training Enhanced LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    class_weight='balanced',
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)
lgb_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
models['LightGBM'] = lgb_model
results.append(evaluate_model(y_test, lgb_pred, lgb_pred_proba, 'LightGBM'))

# 2. Enhanced CatBoost
print("🐱 Training Enhanced CatBoost...")
cb_model = cb.CatBoostClassifier(
    iterations=300,
    depth=8,
    learning_rate=0.05,
    class_weights=[1, weight_ratio],
    subsample=0.8,
    l2_leaf_reg=3,
    random_state=42,
    verbose=False
)

cb_model.fit(X_train, y_train)
cb_pred = cb_model.predict(X_test)
cb_pred_proba = cb_model.predict_proba(X_test)[:, 1]
models['CatBoost'] = cb_model
results.append(evaluate_model(y_test, cb_pred, cb_pred_proba, 'CatBoost'))

# 3. Enhanced Extra Trees
print("🌳 Training Enhanced Extra Trees...")
et_model = ExtraTreesClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

et_model.fit(X_train, y_train)
et_pred = et_model.predict(X_test)
et_pred_proba = et_model.predict_proba(X_test)[:, 1]
models['ExtraTrees'] = et_model
results.append(evaluate_model(y_test, et_pred, et_pred_proba, 'ExtraTrees'))

# 4. Enhanced MLP
print("🧠 Training Enhanced MLP...")
mlp_model = MLPClassifier(
    hidden_layer_sizes=(200, 100, 50),
    max_iter=1000,
    learning_rate_init=0.001,
    alpha=0.01,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20
)

mlp_model.fit(X_train_scaled, y_train)
mlp_pred = mlp_model.predict(X_test_scaled)
mlp_pred_proba = mlp_model.predict_proba(X_test_scaled)[:, 1]
models['MLP'] = mlp_model
results.append(evaluate_model(y_test, mlp_pred, mlp_pred_proba, 'MLP'))

# Results comparison
results_df = pd.DataFrame(results)
print("\n📊 ENHANCED MODEL COMPARISON:")
print("=" * 80)
print(results_df.round(4))

# Find best model
best_model_name = results_df.loc[results_df['f1'].idxmax(), 'model']
best_model = models[best_model_name]
best_metrics = results_df[results_df['model'] == best_model_name].iloc[0]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"F1 Score: {best_metrics['f1']:.4f}")
print(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")
print(f"Balance Achievement: Negative/Positive ratio = {len(df_negatives)/len(df_clean):.2f}")

# Enhanced prediction function
def predict_interaction_enhanced(drug, food, model=best_model, model_name=best_model_name):
    """Enhanced prediction with text features"""
    
    # Create a temporary dataframe for feature extraction
    temp_df = pd.DataFrame({
        'drug': [drug.lower().strip()],
        'food': [food.lower().strip()]
    })
    
    # Apply categorization
    temp_df['drug_category'] = temp_df['drug'].apply(lambda x: categorize_entity(x, drug_categories))
    temp_df['food_category'] = temp_df['food'].apply(lambda x: categorize_entity(x, food_categories))
    temp_df['mechanism'] = temp_df.apply(
        lambda x: get_interaction_mechanism(x['drug_category'], x['food_category']), 
        axis=1
    )
    temp_df['interaction'] = 0  # Placeholder
    
    # Generate features using the same process
    X_sample, _ = create_enhanced_features(temp_df)
    
    # Ensure feature compatibility
    for col in feature_info['feature_names']:
        if col not in X_sample.columns:
            X_sample[col] = 0.0
    
    X_sample = X_sample[feature_info['feature_names']]
    X_sample = X_sample.fillna(0).replace([np.inf, -np.inf], 0)
    
    # Predict
    if model_name == 'MLP':
        X_sample_scaled = scaler.transform(X_sample)
        pred_proba = model.predict_proba(X_sample_scaled)[0, 1]
    else:
        pred_proba = model.predict_proba(X_sample)[0, 1]
    
    return pred_proba, temp_df.iloc[0]['drug_category'], temp_df.iloc[0]['food_category'], temp_df.iloc[0]['mechanism']

# Test cases
test_cases = [
    ('warfarin', 'spinach'),
    ('simvastatin', 'grapefruit'),
    ('tetracycline', 'milk'),
    ('aspirin', 'alcohol'),
    ('metformin', 'bread')
]

print("\n🧪 Testing Enhanced Predictions:")
print("-" * 50)
for drug, food in test_cases:
    prob, drug_cat, food_cat, mechanism = predict_interaction_enhanced(drug, food)
    risk_level = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"
    print(f"{drug.title()} + {food.title()}:")
    print(f"  Probability: {prob:.3f}")
    print(f"  Risk Level: {risk_level}")
    print(f"  Categories: {drug_cat} + {food_cat}")
    print(f"  Mechanism: {mechanism}")
    print()

# Save enhanced model
print("💾 Saving enhanced model...")
enhanced_model_package = {
    'model': best_model,
    'scaler': scaler,
    'feature_info': feature_info,
    'drug_categories': drug_categories,
    'food_categories': food_categories,
    'high_risk_interactions': high_risk_interactions,
    'model_name': best_model_name,
    'performance': dict(best_metrics),
    'balance_ratio': len(df_negatives)/len(df_clean)
}

with open('enhanced_drug_food_model.pkl', 'wb') as f:
    pickle.dump(enhanced_model_package, f)

print("✅ Enhanced model saved successfully!")

print("\n🎉 ENHANCEMENT SUMMARY")
print("=" * 50)
print("✅ DATASET BALANCING:")
print(f"   - Original imbalance ratio: 0.23")
print(f"   - New balance ratio: {len(df_negatives)/len(df_clean):.2f}")
print(f"   - Total samples: {len(df_final):,}")

print("\n✅ ENHANCED PREPROCESSING:")
print("   - Advanced text cleaning with regex")
print("   - Improved categorization with partial matching")
print("   - Better negative sampling strategy")

print("\n✅ ADVANCED FEATURE ENGINEERING:")
print(f"   - Total features: {X.shape[1]} (vs original 28)")
print("   - Text features using TF-IDF + SVD")
print("   - Statistical features (length, similarity)")
print("   - Category interaction features")
print("   - Enhanced risk scoring")

print(f"\n🏆 Best Model: {best_model_name} with F1: {best_metrics['f1']:.4f}")
