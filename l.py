import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, accuracy_score, precision_score, recall_score, f1_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from collections import Counter
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# Dataset path
df2 = '/Users/sachidhoka/Desktop/food-drug interactions.csv'

print("🚀 STARTING DRUG-FOOD INTERACTION PREDICTOR")
print("=" * 60)

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

# Feature engineering (without BioBERT)
print("\n🔧 FEATURE ENGINEERING")
print("-" * 40)

def create_text_features(df):
    """Create TF-IDF features for drug and food names"""
    print("Creating TF-IDF features...")
    
    # Combine drug and food texts
    drug_texts = df['drug'].tolist()
    food_texts = df['food'].tolist()
    
    # Create TF-IDF vectorizers
    drug_vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2), lowercase=True)
    food_vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2), lowercase=True)
    
    # Fit and transform
    drug_tfidf = drug_vectorizer.fit_transform(drug_texts).toarray()
    food_tfidf = food_vectorizer.fit_transform(food_texts).toarray()
    
    # Create DataFrames
    drug_tfidf_df = pd.DataFrame(drug_tfidf, columns=[f'drug_tfidf_{i}' for i in range(drug_tfidf.shape[1])])
    food_tfidf_df = pd.DataFrame(food_tfidf, columns=[f'food_tfidf_{i}' for i in range(food_tfidf.shape[1])])
    
    return drug_tfidf_df, food_tfidf_df, drug_vectorizer, food_vectorizer

def create_interaction_features(df):
    """Create interpretable interaction features"""
    print("Creating interaction features...")
    
    features = pd.DataFrame()
    
    # String matching features
    features['drug_food_char_overlap'] = df.apply(
        lambda x: len(set(x['drug']) & set(x['food'])) / max(len(x['drug']), len(x['food'])), axis=1
    )
    
    features['drug_food_length_ratio'] = df.apply(
        lambda x: len(x['drug']) / len(x['food']) if len(x['food']) > 0 else 0, axis=1
    )
    
    # Known interaction patterns
    features['same_first_letter'] = (df['drug'].str[0] == df['food'].str[0]).astype(int)
    
    # Drug-specific features
    features['drug_length'] = df['drug'].str.len()
    features['food_length'] = df['food'].str.len()
    
    # Vowel/consonant ratios (simple linguistic features)
    features['drug_vowel_ratio'] = df['drug'].apply(
        lambda x: sum(1 for c in x.lower() if c in 'aeiou') / len(x) if len(x) > 0 else 0
    )
    features['food_vowel_ratio'] = df['food'].apply(
        lambda x: sum(1 for c in x.lower() if c in 'aeiou') / len(x) if len(x) > 0 else 0
    )
    
    return features

def create_risk_score(drug_cat, food_cat, mechanism):
    """Create risk score based on known interactions"""
    if mechanism != 'unknown':
        return 3
    elif drug_cat != 'other' and food_cat != 'other':
        return 2
    else:
        return 1

# Create all features
df_final['risk_score'] = df_final.apply(
    lambda x: create_risk_score(x['drug_category'], x['food_category'], x['mechanism']), 
    axis=1
)

# Create categorical dummy variables
drug_dummies = pd.get_dummies(df_final['drug_category'], prefix='drug').astype(int)
food_dummies = pd.get_dummies(df_final['food_category'], prefix='food').astype(int)
mechanism_dummies = pd.get_dummies(df_final['mechanism'], prefix='mechanism').astype(int)

# Create text features
drug_tfidf_df, food_tfidf_df, drug_vectorizer, food_vectorizer = create_text_features(df_final)

# Create interaction features
interaction_features = create_interaction_features(df_final)

# Combine all features
X = pd.concat([
    drug_dummies, 
    food_dummies, 
    mechanism_dummies,
    drug_tfidf_df,
    food_tfidf_df,
    interaction_features,
    df_final[['risk_score']]
], axis=1)

y = df_final['interaction']
feature_cols = list(X.columns)
X = X.astype(float)

print(f"Final feature matrix shape: {X.shape}")
print(f"Feature categories:")
print(f"- Drug categories: {len(drug_dummies.columns)}")
print(f"- Food categories: {len(food_dummies.columns)}")
print(f"- Mechanism categories: {len(mechanism_dummies.columns)}")
print(f"- Drug TF-IDF features: {len(drug_tfidf_df.columns)}")
print(f"- Food TF-IDF features: {len(food_tfidf_df.columns)}")
print(f"- Interaction features: {len(interaction_features.columns)}")
print(f"- Risk score: 1")

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

# Calculate metrics
xgb_metrics = calculate_metrics(y_test, xgb_pred, xgb_pred_proba, "XGBoost")
rf_metrics = calculate_metrics(y_test, rf_pred, rf_pred_proba, "Random Forest")
gb_metrics = calculate_metrics(y_test, gb_pred, gb_pred_proba, "Gradient Boosting")

metrics_df = pd.DataFrame([xgb_metrics, rf_metrics, gb_metrics])
print("\n📊 MODEL COMPARISON")
print(metrics_df.round(4).to_string(index=False))

# Select best model
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

# Feature importance analysis for XAI
print("\n🔍 FEATURE IMPORTANCE ANALYSIS (XAI)")
print("-" * 50)

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 20 Most Important Features:")
    print(feature_importance.head(20).to_string(index=False))
    
    # Visualize top features
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top 15 Feature Importances - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# Detailed classification report
print(f"\n📋 DETAILED CLASSIFICATION REPORT - {best_model_name}")
print("-" * 60)
print(classification_report(y_test, best_pred))

# Confusion matrix
print("\n🔢 CONFUSION MATRIX")
print("-" * 30)
cm = confusion_matrix(y_test, best_pred)
print(cm)
print(f"True Negatives: {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives: {cm[1,1]}")

# Save model package
model_package = {
    'model': best_model,
    'feature_columns': feature_cols,
    'drug_vectorizer': drug_vectorizer,
    'food_vectorizer': food_vectorizer,
    'model_name': best_model_name,
    'drug_categories': drug_categories,
    'food_categories': food_categories,
    'high_risk_interactions': high_risk_interactions,
    'feature_importance': feature_importance if 'feature_importance' in locals() else None
}

with open('/Users/sachidhoka/Desktop/drug_food_interaction_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print(f"\n✅ Model saved to '/Users/sachidhoka/Desktop/drug_food_interaction_model.pkl'")

# Prediction function for new interactions
def predict_interaction(drug_name, food_name, model_package):
    """Predict interaction for new drug-food pair"""
    
    # Create temporary dataframe
    temp_df = pd.DataFrame({
        'drug': [drug_name.lower().strip()],
        'food': [food_name.lower().strip()]
    })
    
    # Apply same preprocessing
    temp_df['drug_category'] = temp_df['drug'].apply(lambda x: categorize_entity(x, drug_categories))
    temp_df['food_category'] = temp_df['food'].apply(lambda x: categorize_entity(x, food_categories))
    temp_df['mechanism'] = temp_df.apply(
        lambda x: get_interaction_mechanism(x['drug_category'], x['food_category']), 
        axis=1
    )
    temp_df['risk_score'] = temp_df.apply(
        lambda x: create_risk_score(x['drug_category'], x['food_category'], x['mechanism']), 
        axis=1
    )
    
    # Create features (same as training)
    drug_dummies_temp = pd.get_dummies(temp_df['drug_category'], prefix='drug').reindex(
        columns=[col for col in feature_cols if col.startswith('drug_')], fill_value=0
    )
    food_dummies_temp = pd.get_dummies(temp_df['food_category'], prefix='food').reindex(
        columns=[col for col in feature_cols if col.startswith('food_')], fill_value=0
    )
    mechanism_dummies_temp = pd.get_dummies(temp_df['mechanism'], prefix='mechanism').reindex(
        columns=[col for col in feature_cols if col.startswith('mechanism_')], fill_value=0
    )
    
    # TF-IDF features
    drug_tfidf_temp = model_package['drug_vectorizer'].transform([drug_name]).toarray()
    food_tfidf_temp = model_package['food_vectorizer'].transform([food_name]).toarray()
    
    drug_tfidf_df_temp = pd.DataFrame(drug_tfidf_temp, columns=[f'drug_tfidf_{i}' for i in range(drug_tfidf_temp.shape[1])])
    food_tfidf_df_temp = pd.DataFrame(food_tfidf_temp, columns=[f'food_tfidf_{i}' for i in range(food_tfidf_temp.shape[1])])
    
    # Interaction features
    interaction_features_temp = create_interaction_features(temp_df)
    
    # Combine features
    X_temp = pd.concat([
        drug_dummies_temp, 
        food_dummies_temp, 
        mechanism_dummies_temp,
        drug_tfidf_df_temp,
        food_tfidf_df_temp,
        interaction_features_temp,
        temp_df[['risk_score']]
    ], axis=1)
    
    # Ensure all columns are present
    X_temp = X_temp.reindex(columns=feature_cols, fill_value=0)
    
    # Predict
    prediction = model_package['model'].predict(X_temp)[0]
    probability = model_package['model'].predict_proba(X_temp)[0]
    
    return {
        'interaction_predicted': bool(prediction),
        'interaction_probability': float(probability[1]),
        'drug_category': temp_df['drug_category'].iloc[0],
        'food_category': temp_df['food_category'].iloc[0],
        'mechanism': temp_df['mechanism'].iloc[0],
        'risk_score': temp_df['risk_score'].iloc[0]
    }

# Example predictions
print("\n🧪 EXAMPLE PREDICTIONS")
print("-" * 40)

test_pairs = [
    ('warfarin', 'spinach'),
    ('simvastatin', 'grapefruit'),
    ('aspirin', 'pizza'),
    ('metformin', 'banana')
]

for drug, food in test_pairs:
    result = predict_interaction(drug, food, model_package)
    print(f"\n{drug.capitalize()} + {food.capitalize()}:")
    print(f"  Interaction Risk: {'HIGH' if result['interaction_predicted'] else 'LOW'}")
    print(f"  Probability: {result['interaction_probability']:.3f}")
    print(f"  Drug Category: {result['drug_category']}")
    print(f"  Food Category: {result['food_category']}")
    print(f"  Mechanism: {result['mechanism']}")
    print(f"  Risk Score: {result['risk_score']}")

print(f"\n🎉 ANALYSIS COMPLETE!")
print(f"✅ Model trained and saved successfully")
print(f"🔍 Features are interpretable for XAI")
print(f"📊 Best model: {best_model_name} with F1-Score: {best_f1_score:.4f}")
