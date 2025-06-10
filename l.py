import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from collections import Counter
import pickle
import warnings
warnings.filterwarnings('ignore')

# For M3 MacBook Air MPS support
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
np.random.seed(42)

print("🚀 STARTING 12-HOUR DRUG-FOOD INTERACTION PREDICTOR SPRINT")
print("=" * 60)

# HOUR 1-2: EMERGENCY DATA PREP
print("\n📋 PHASE 1: DATA PREPROCESSING (Hours 1-2)")
print("-" * 40)
df2 = '/Users/sachidhoka/Desktop/food-drug interactions.csv'
try:
    # For newer pandas versions (1.3.0+)
    foodrugs_df = pd.read_csv(df2, on_bad_lines='skip')
except TypeError:
    # For older pandas versions
    foodrugs_df = pd.read_csv(df2, error_bad_lines=False)
def load_and_clean_foodrugs(filepath=foodrugs_df):
    """Load and clean FooDrugs dataset"""
    print("Loading FooDrugs dataset...")
    
    try:
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                print(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise Exception("Could not load file with any encoding")
            
    except FileNotFoundError:
        print("⚠️ FooDrugs file not found. Creating enhanced sample data for demo...")
        # Create more comprehensive sample data for demonstration
        drugs = ['warfarin', 'simvastatin', 'tetracycline', 'aspirin', 'metformin', 
                'lisinopril', 'sertraline', 'digoxin', 'amoxicillin', 'atorvastatin',
                'ibuprofen', 'omeprazole', 'losartan', 'metoprolol', 'fluoxetine'] * 50
        
        foods = ['spinach', 'grapefruit', 'milk', 'alcohol', 'bread', 'banana', 
                'coffee', 'cheese', 'broccoli', 'orange', 'yogurt', 'kale',
                'wine', 'tea', 'avocado'] * 50
        
        # Create random combinations
        np.random.shuffle(drugs)
        np.random.shuffle(foods)
        
        sample_data = {
            'drug': drugs,
            'food': foods
        }
        df = pd.DataFrame(sample_data)
    
    print(f"Original dataset size: {len(df)}")
    
    # Clean the dataset
    df_clean = df.dropna(subset=['drug', 'food'])
    df_clean = df_clean.drop_duplicates(subset=['drug', 'food'])
    
    # Clean text
    df_clean['drug'] = df_clean['drug'].astype(str).str.lower().str.strip()
    df_clean['food'] = df_clean['food'].astype(str).str.lower().str.strip()
    
    # Remove invalid entries
    df_clean = df_clean[
        (df_clean['drug'].str.len() > 2) & 
        (df_clean['food'].str.len() > 2) &
        (~df_clean['drug'].str.contains(r'^\d+$')) &
        (~df_clean['food'].str.contains(r'^\d+$'))
    ]
    
    # Keep only essential columns and add interaction label
    df_clean = df_clean[['drug', 'food']].copy()
    df_clean['interaction'] = 1  # All remaining are positive interactions
    
    print(f"Clean dataset size: {len(df_clean)} interactions")
    print(f"Unique drugs: {df_clean['drug'].nunique()}")
    print(f"Unique foods: {df_clean['food'].nunique()}")
    
    return df_clean

# Load data
df_clean = load_and_clean_foodrugs()

# Display sample data
print("\nSample interactions:")
print(df_clean.head(10))

# HARDCODED KNOWLEDGE BASE (Critical for 12-hour sprint)
print("\n🧠 Creating knowledge base...")

# Drug categories based on mechanism of action
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

# Food categories based on interaction mechanisms
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

# Known high-risk interaction mechanisms
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

# Display categorization results
print("\nDrug category distribution:")
print(df_clean['drug_category'].value_counts().head(10))
print("\nFood category distribution:")
print(df_clean['food_category'].value_counts().head(10))

# GENERATE NEGATIVE SAMPLES (Critical!)
print("\n⚖️ Generating negative samples...")

# Get unique entities (increased for better model)
unique_drugs = df_clean['drug'].unique()
unique_foods = df_clean['food'].unique()

print(f"Working with {len(unique_drugs)} drugs and {len(unique_foods)} foods")

# Create negative samples more efficiently
from itertools import product
existing_interactions = set(zip(df_clean['drug'], df_clean['food']))

negative_samples = []
max_negatives = min(len(df_clean) * 3, 2000)  # Limit for performance

# Sample random combinations for negatives
all_combinations = list(product(unique_drugs, unique_foods))
np.random.shuffle(all_combinations)

for drug, food in all_combinations:
    if (drug, food) not in existing_interactions:
        drug_cat = categorize_entity(drug, drug_categories)
        food_cat = categorize_entity(food, food_categories)
        mechanism = get_interaction_mechanism(drug_cat, food_cat)
        
        negative_samples.append({
            'drug': drug,
            'food': food,
            'interaction': 0,
            'drug_category': drug_cat,
            'food_category': food_cat,
            'mechanism': mechanism
        })
    
    if len(negative_samples) >= max_negatives:
        break

df_negatives = pd.DataFrame(negative_samples)
df_final = pd.concat([df_clean, df_negatives], ignore_index=True)

print(f"Final dataset: {len(df_final)} samples")
print(f"Positive interactions: {len(df_clean)}")
print(f"Negative interactions: {len(df_negatives)}")
print(f"Balance ratio: {len(df_negatives)/len(df_clean):.2f}")

# HOUR 3-4: FEATURE ENGINEERING
print("\n🔧 PHASE 2: FEATURE ENGINEERING (Hours 3-4)")
print("-" * 40)

# Create interaction features
def create_risk_score(drug_cat, food_cat, mechanism):
    """Create risk score based on known interactions"""
    if mechanism != 'unknown':
        return 3  # High risk
    elif drug_cat != 'other' and food_cat != 'other':
        return 2  # Medium risk
    else:
        return 1  # Low risk

df_final['risk_score'] = df_final.apply(
    lambda x: create_risk_score(x['drug_category'], x['food_category'], x['mechanism']), 
    axis=1
)

# Create binary features with proper data types
# Create binary features with proper data types
drug_dummies = pd.get_dummies(df_final['drug_category'], prefix='drug').astype(int)
food_dummies = pd.get_dummies(df_final['food_category'], prefix='food').astype(int)
mechanism_dummies = pd.get_dummies(df_final['mechanism'], prefix='mechanism').astype(int)

# Select feature columns FIRST (before combining)
feature_cols = list(drug_dummies.columns) + list(food_dummies.columns) + list(mechanism_dummies.columns)
feature_cols.append('risk_score')

# Create feature matrix with ONLY the numeric columns
X = pd.concat([drug_dummies, food_dummies, mechanism_dummies, df_final[['risk_score']]], axis=1)
y = df_final['interaction']

# Ensure all features are numeric
X = X.astype(float)

print(f"Feature matrix shape: {X.shape}")
print(f"Number of features: {len(feature_cols)}")
print(f"Feature data types: {X.dtypes.unique()}")

# Handle class imbalance
class_counts = Counter(y)
print(f"Class distribution: {dict(class_counts)}")

# HOUR 5-6: MODEL TRAINING
print("\n🤖 PHASE 3: MODEL TRAINING (Hours 5-6)")
print("-" * 40)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Ensure training data is also properly typed
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# Calculate class weights
train_counts = Counter(y_train)
weight_ratio = train_counts[0] / train_counts[1] if train_counts[1] > 0 else 1

# XGBoost Model (Primary) - Fixed version
print("\n🚀 Training XGBoost model...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=weight_ratio,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',  # Optimized for M3 chip
    enable_categorical=False  # Explicitly disable categorical handling
)

xgb_model.fit(X_train, y_train)

# Predictions
xgb_pred = xgb_model.predict(X_test)
xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# Random Forest Model (Backup)
print("🌲 Training Random Forest backup...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Model comparison
xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
rf_auc = roc_auc_score(y_test, rf_pred_proba)

print(f"\nModel Performance:")
print(f"XGBoost ROC-AUC: {xgb_auc:.3f}")
print(f"Random Forest ROC-AUC: {rf_auc:.3f}")

# Choose best model
if xgb_auc > rf_auc:
    best_model = xgb_model
    best_pred = xgb_pred
    best_pred_proba = xgb_pred_proba
    best_name = "XGBoost"
else:
    best_model = rf_model
    best_pred = rf_pred
    best_pred_proba = rf_pred_proba
    best_name = "Random Forest"

print(f"Best model: {best_name}")

# HOUR 7-8: VALIDATION AND TESTING
print("\n📊 PHASE 4: VALIDATION (Hours 7-8)")
print("-" * 40)

# Detailed evaluation
print(f"\n{best_name} Performance Report:")
print(f"ROC-AUC Score: {roc_auc_score(y_test, best_pred_proba):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, best_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Interaction', 'Interaction'],
            yticklabels=['No Interaction', 'Interaction'])
plt.title(f'{best_name} Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ROC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, best_pred_proba)
plt.plot(fpr, tpr, linewidth=2, label=f'{best_name} (AUC = {roc_auc_score(y_test, best_pred_proba):.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature Importance
importance = best_model.feature_importances_

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': importance
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15))

# Plot feature importance
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title(f'Top 15 Feature Importance - {best_name}')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# PREDICTION FUNCTION
def predict_interaction(drug, food, model=best_model, features=feature_cols):
    """Predict drug-food interaction"""
    # Create feature vector
    sample_features = {col: 0.0 for col in features}
    
    # Categorize input
    drug_cat = categorize_entity(drug, drug_categories)
    food_cat = categorize_entity(food, food_categories)
    mechanism = get_interaction_mechanism(drug_cat, food_cat)
    risk_score = create_risk_score(drug_cat, food_cat, mechanism)
    
    # Set relevant features
    if f'drug_{drug_cat}' in sample_features:
        sample_features[f'drug_{drug_cat}'] = 1.0
    if f'food_{food_cat}' in sample_features:
        sample_features[f'food_{food_cat}'] = 1.0
    if f'mechanism_{mechanism}' in sample_features:
        sample_features[f'mechanism_{mechanism}'] = 1.0
    if 'risk_score' in sample_features:
        sample_features['risk_score'] = float(risk_score)
    
    # Predict
    features_array = np.array(list(sample_features.values())).reshape(1, -1)
    prediction_proba = model.predict_proba(features_array)[0, 1]
    
    return prediction_proba, drug_cat, food_cat, mechanism

# Test known interactions
test_cases = [
    ('warfarin', 'spinach'),
    ('simvastatin', 'grapefruit'),
    ('tetracycline', 'milk'),
    ('aspirin', 'alcohol'),
    ('metformin', 'bread'),
    ('digoxin', 'banana'),
    ('sertraline', 'wine'),
    ('omeprazole', 'coffee'),
    ('lisinopril', 'potassium')
]

print("\n🧪 Testing Known Interactions:")
print("-" * 50)
for drug, food in test_cases:
    prob, drug_cat, food_cat, mechanism = predict_interaction(drug, food)
    risk_level = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"
    print(f"{drug.title()} + {food.title()}:")
    print(f"  Probability: {prob:.3f}")
    print(f"  Risk Level: {risk_level}")
    print(f"  Categories: {drug_cat} + {food_cat}")
    print(f"  Mechanism: {mechanism}")
    print()

# Save model and components
print("💾 Saving model...")
model_package = {
    'model': best_model,
    'feature_cols': feature_cols,
    'drug_categories': drug_categories,
    'food_categories': food_categories,
    'high_risk_interactions': high_risk_interactions,
    'model_name': best_name,
    'performance': {
        'roc_auc': roc_auc_score(y_test, best_pred_proba),
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
}

with open('drug_food_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print("✅ Model saved successfully!")

# FINAL SUMMARY
print("\n🎉 12-HOUR SPRINT SUMMARY")
print("=" * 60)
print(f"✅ Dataset processed: {len(df_final):,} samples")
print(f"✅ Features engineered: {len(feature_cols)}")
print(f"✅ Best model: {best_name}")
print(f"✅ ROC-AUC Score: {roc_auc_score(y_test, best_pred_proba):.3f}")
print(f"✅ Model saved: drug_food_model.pkl")
print(f"✅ Plots saved: confusion_matrix.png, roc_curve.png, feature_importance.png")
print("\n🚀 Ready for API development in Hours 9-10!")

# Additional debugging info
print("\n🔍 DEBUG INFO:")
print(f"Final X shape: {X.shape}")
print(f"Final X dtypes: {X.dtypes.value_counts()}")
print(f"Any NaN values: {X.isnull().sum().sum()}")
print(f"Feature columns: {len(feature_cols)}")
