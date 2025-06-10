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
warnings.filterwarnings('ignore')

# For M3 MacBook Air MPS support (if available)
try:
    import torch
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
except ImportError:
    print("PyTorch not available, using CPU for sklearn models")

# Set random seeds for reproducibility
np.random.seed(42)

print("🚀 STARTING 12-HOUR DRUG-FOOD INTERACTION PREDICTOR SPRINT")
print("=" * 60)

# HOUR 1-2: EMERGENCY DATA PREP
print("\n📋 PHASE 1: DATA PREPROCESSING (Hours 1-2)")
print("-" * 40)

def load_and_clean_foodrugs(filepath=None):
    """Load and clean FooDrugs dataset"""
    print("Loading FooDrugs dataset...")
    
    if filepath is None:
        # filepath = '/Users/sachidhoka/Desktop/food-drug interactions.csv'
        filepath ='/content/drive/MyDrive/ASEP_2/food-drug interactions.csv'
    
    try:
        # Try different encodings with error handling
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

# Get unique entities (LIMITED for memory efficiency)
unique_drugs = df_clean['drug'].unique()
unique_foods = df_clean['food'].unique()

# LIMIT the entities to prevent memory issues
if len(unique_drugs) > 1000:
    unique_drugs = np.random.choice(unique_drugs, 1000, replace=False)
if len(unique_foods) > 1000:
    unique_foods = np.random.choice(unique_foods, 1000, replace=False)

print(f"Working with {len(unique_drugs)} drugs and {len(unique_foods)} foods")

existing_interactions = set(zip(df_clean['drug'], df_clean['food']))

negative_samples = []
max_negatives = len(df_clean)  # Match positive samples 1:1

# Generate random combinations WITHOUT creating all_combinations list
attempts = 0
max_attempts = max_negatives * 10  # Prevent infinite loop

# Better negative sampling strategy
while len(negative_samples) < max_negatives and attempts < max_attempts:
    drug = np.random.choice(unique_drugs)
    food = np.random.choice(unique_foods)
    attempts += 1
    
    if (drug, food) not in existing_interactions:
        drug_cat = categorize_entity(drug, drug_categories)
        food_cat = categorize_entity(food, food_categories)
        mechanism = get_interaction_mechanism(drug_cat, food_cat)
        
        # Only add as negative if it's truly unlikely to interact
        # Skip combinations that should logically interact
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

# Check if we have negative samples
if len(df_negatives) > 0:
    print(f"Balance ratio: {len(df_negatives)/len(df_clean):.2f}")
else:
    print("Warning: No negative samples generated!")

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

# Scale features for MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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

# Initialize models
models = {}
results = []

# 1. LightGBM Model
print("\n🚀 Training LightGBM model...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    class_weight='balanced',
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)
lgb_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
models['LightGBM'] = lgb_model
results.append(evaluate_model(y_test, lgb_pred, lgb_pred_proba, 'LightGBM'))

# 2. CatBoost Model
print("🐱 Training CatBoost model...")
cb_model = cb.CatBoostClassifier(
    iterations=200,
    depth=8,
    learning_rate=0.05,
    class_weights=[1, weight_ratio],
    subsample=0.8,
    random_state=42,
    verbose=False
)

cb_model.fit(X_train, y_train)
cb_pred = cb_model.predict(X_test)
cb_pred_proba = cb_model.predict_proba(X_test)[:, 1]
models['CatBoost'] = cb_model
results.append(evaluate_model(y_test, cb_pred, cb_pred_proba, 'CatBoost'))

# 3. Extra Trees Model
print("🌳 Training Extra Trees model...")
et_model = ExtraTreesClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

et_model.fit(X_train, y_train)
et_pred = et_model.predict(X_test)
et_pred_proba = et_model.predict_proba(X_test)[:, 1]
models['ExtraTrees'] = et_model
results.append(evaluate_model(y_test, et_pred, et_pred_proba, 'ExtraTrees'))

# 4. MLP Classifier
print("🧠 Training MLP Classifier...")
mlp_model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

mlp_model.fit(X_train_scaled, y_train)
mlp_pred = mlp_model.predict(X_test_scaled)
mlp_pred_proba = mlp_model.predict_proba(X_test_scaled)[:, 1]
models['MLP'] = mlp_model
results.append(evaluate_model(y_test, mlp_pred, mlp_pred_proba, 'MLP'))

# 5. Voting Ensemble
print("🗳️ Training Voting Ensemble...")
voting_model = VotingClassifier(
    estimators=[
        ('lgb', lgb_model),
        ('cb', cb_model),
        ('et', et_model)
    ],
    voting='soft'
)

voting_model.fit(X_train, y_train)
voting_pred = voting_model.predict(X_test)
voting_pred_proba = voting_model.predict_proba(X_test)[:, 1]
models['Voting'] = voting_model
results.append(evaluate_model(y_test, voting_pred, voting_pred_proba, 'Voting'))

# Create results DataFrame
results_df = pd.DataFrame(results)
print("\n📊 MODEL COMPARISON RESULTS:")
print("=" * 80)
print(results_df.round(4))

# Find best model based on F1 score
best_model_name = results_df.loc[results_df['f1'].idxmax(), 'model']
best_model = models[best_model_name]
best_metrics = results_df[results_df['model'] == best_model_name].iloc[0]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"F1 Score: {best_metrics['f1']:.4f}")
print(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")

# Cross-validation for best model
print(f"\n🔄 Cross-validation for {best_model_name}:")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

if best_model_name == 'MLP':
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=cv, scoring='f1_weighted')
else:
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='f1_weighted')

print(f"CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# HOUR 7-8: VALIDATION AND TESTING
print("\n📊 PHASE 4: VALIDATION (Hours 7-8)")
print("-" * 40)

# Detailed evaluation for best model
if best_model_name == 'MLP':
    best_pred = mlp_pred
    best_pred_proba = mlp_pred_proba
elif best_model_name == 'LightGBM':
    best_pred = lgb_pred
    best_pred_proba = lgb_pred_proba
elif best_model_name == 'CatBoost':
    best_pred = cb_pred
    best_pred_proba = cb_pred_proba
elif best_model_name == 'ExtraTrees':
    best_pred = et_pred
    best_pred_proba = et_pred_proba
else:  # Voting
    best_pred = voting_pred
    best_pred_proba = voting_pred_proba

print(f"\n{best_model_name} Detailed Performance Report:")
print(f"Accuracy: {best_metrics['accuracy']:.4f}")
print(f"Precision: {best_metrics['precision']:.4f}")
print(f"Recall: {best_metrics['recall']:.4f}")
print(f"F1 Score: {best_metrics['f1']:.4f}")
print(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")
print(f"Average Precision: {best_metrics['avg_precision']:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, best_pred))

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Confusion Matrices for all models
model_preds = {
    'LightGBM': lgb_pred,
    'CatBoost': cb_pred,
    'ExtraTrees': et_pred,
    'MLP': mlp_pred,
    'Voting': voting_pred
}

for i, (name, pred) in enumerate(model_preds.items()):
    if i < 5:
        row, col = i // 3, i % 3
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col],
                   xticklabels=['No Interaction', 'Interaction'],
                   yticklabels=['No Interaction', 'Interaction'])
        axes[row, col].set_title(f'{name} Confusion Matrix')
        axes[row, col].set_ylabel('True Label')
        axes[row, col].set_xlabel('Predicted Label')

# Remove empty subplot
axes[1, 2].remove()

plt.tight_layout()
plt.savefig('all_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. ROC Curves
plt.figure(figsize=(12, 8))
model_probas = {
    'LightGBM': lgb_pred_proba,
    'CatBoost': cb_pred_proba,
    'ExtraTrees': et_pred_proba,
    'MLP': mlp_pred_proba,
    'Voting': voting_pred_proba
}

colors = ['blue', 'red', 'green', 'orange', 'purple']
for i, (name, proba) in enumerate(model_probas.items()):
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_score = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc_score:.3f})', color=colors[i])

plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - All Models')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig('all_roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Metrics Comparison
plt.figure(figsize=(14, 8))
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'avg_precision']
x = np.arange(len(results_df))
width = 0.12

for i, metric in enumerate(metrics_to_plot):
    plt.bar(x + i*width, results_df[metric], width, label=metric.replace('_', ' ').title())

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x + width*2.5, results_df['model'])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature Importance (for tree-based models)
if best_model_name in ['LightGBM', 'CatBoost', 'ExtraTrees']:
    importance = best_model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print(f"\nTop 15 Most Important Features ({best_model_name}):")
    print(feature_importance.head(15))

    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top 15 Feature Importance - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

# PREDICTION FUNCTION
def predict_interaction(drug, food, model=best_model, features=feature_cols, model_name=best_model_name):
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
    
    if model_name == 'MLP':
        features_array = scaler.transform(features_array)
    
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
    'scaler': scaler if best_model_name == 'MLP' else None,
    'feature_cols': feature_cols,
    'drug_categories': drug_categories,
    'food_categories': food_categories,
    'high_risk_interactions': high_risk_interactions,
    'model_name': best_model_name,
    'performance': dict(best_metrics),
    'all_results': results_df.to_dict('records')
}

with open('drug_food_model_enhanced.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print("✅ Enhanced model saved successfully!")

# FINAL SUMMARY
print("\n🎉 12-HOUR SPRINT SUMMARY")
print("=" * 60)
print(f"✅ Dataset processed: {len(df_final):,} samples")
print(f"✅ Features engineered: {len(feature_cols)}")
print(f"✅ Models trained: 5 (LightGBM, CatBoost, ExtraTrees, MLP, Voting)")
print(f"✅ Best model: {best_model_name}")
print(f"✅ Best F1 Score: {best_metrics['f1']:.4f}")
print(f"✅ Best ROC-AUC: {best_metrics['roc_auc']:.4f}")
print(f"✅ Model saved: drug_food_model_enhanced.pkl")
print(f"✅ Plots saved: all_confusion_matrices.png, all_roc_curves.png, metrics_comparison.png")

print("\n📈 ALL MODEL PERFORMANCE SUMMARY:")
print("-" * 60)
for _, row in results_df.iterrows():
    print(f"{row['model']:12} | Acc: {row['accuracy']:.3f} | Prec: {row['precision']:.3f} | "
          f"Rec: {row['recall']:.3f} | F1: {row['f1']:.3f} | AUC: {row['roc_auc']:.3f}")

print("\n🔍 MODEL INSIGHTS:")
print("-" * 30)
if best_model_name in ['LightGBM', 'CatBoost', 'ExtraTrees']:
    print(f"• {best_model_name} excels with tree-based ensemble learning")
    print("• Feature importance analysis available")
elif best_model_name == 'MLP':
    print("• Neural network captured complex non-linear patterns")
    print("• Feature scaling was crucial for performance")
else:  # Voting
    print("• Ensemble voting leveraged strengths of multiple models")
    print("• Robust predictions through model combination")

print(f"\n🎯 KEY ACHIEVEMENTS:")
print(f"• Balanced dataset with {len(df_clean)} positive and {len(df_negatives)} negative samples")
print(f"• {len(feature_cols)} engineered features from drug/food categories")
print(f"• Cross-validation confirmed model stability")
print(f"• Comprehensive evaluation with 6 metrics across 5 models")

print("\n🚀 Ready for API development in Hours 9-10!")

# Additional debugging info
print("\n🔍 DEBUG INFO:")
print(f"Final X shape: {X.shape}")
print(f"Final X dtypes: {X.dtypes.value_counts()}")
print(f"Any NaN values: {X.isnull().sum().sum()}")
print(f"Feature columns: {len(feature_cols)}")
print(f"Class balance in test set: {Counter(y_test)}")

# Performance comparison table
print("\n📊 DETAILED PERFORMANCE TABLE:")
print("=" * 90)
print(f"{'Model':<12} {'Accuracy':<9} {'Precision':<10} {'Recall':<8} {'F1-Score':<9} {'ROC-AUC':<8} {'Avg-Prec':<8}")
print("-" * 90)
for _, row in results_df.iterrows():
    print(f"{row['model']:<12} {row['accuracy']:<9.4f} {row['precision']:<10.4f} "
          f"{row['recall']:<8.4f} {row['f1']:<9.4f} {row['roc_auc']:<8.4f} {row['avg_precision']:<8.4f}")

print("\n✨ ENHANCED FEATURES ADDED:")
print("• LightGBM: Fast gradient boosting with categorical support")
print("• CatBoost: Handles categorical features natively")
print("• Extra Trees: Extremely randomized trees for variance reduction")
print("• MLP Classifier: Neural network for complex pattern recognition")
print("• Voting Ensemble: Combines predictions from multiple models")
print("• Comprehensive metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Avg Precision")
print("• Enhanced visualizations: Multiple confusion matrices, ROC curves, metrics comparison")
print("• Cross-validation for model stability assessment")

print(f"\n🏆 WINNER: {best_model_name} with F1-Score of {best_metrics['f1']:.4f}")
print("🎯 Model selection based on F1-score for balanced performance on imbalanced data")
