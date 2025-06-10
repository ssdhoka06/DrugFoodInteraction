import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, accuracy_score, precision_score, recall_score, f1_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
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

# Function to calculate all metrics for a model
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

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix for a model"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Interaction', 'Interaction'],
                yticklabels=['No Interaction', 'Interaction'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    return cm

# XGBoost Model (Primary)
print("\n🚀 Training XGBoost model...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    scale_pos_weight=weight_ratio,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    enable_categorical=False
)

xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# Random Forest Model
print("🌲 Training Random Forest model...")
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

# Gradient Boosting Model
print("🚀 Training Gradient Boosting model...")
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_pred_proba = gb_model.predict_proba(X_test)[:, 1]

# Calculate metrics for all models
xgb_metrics = calculate_metrics(y_test, xgb_pred, xgb_pred_proba, "XGBoost")
rf_metrics = calculate_metrics(y_test, rf_pred, rf_pred_proba, "Random Forest")
gb_metrics = calculate_metrics(y_test, gb_pred, gb_pred_proba, "Gradient Boosting")

# Create comprehensive comparison table
all_metrics = [xgb_metrics, rf_metrics, gb_metrics]
metrics_df = pd.DataFrame(all_metrics)

print("\n📊 COMPREHENSIVE MODEL COMPARISON")
print("=" * 70)
print(metrics_df.round(4).to_string(index=False))

# Plot confusion matrices for all models
print("\n📈 CONFUSION MATRICES")
print("-" * 40)

# XGBoost Confusion Matrix
xgb_cm = plot_confusion_matrix(y_test, xgb_pred, "XGBoost")

# Random Forest Confusion Matrix
rf_cm = plot_confusion_matrix(y_test, rf_pred, "Random Forest")

# Gradient Boosting Confusion Matrix
gb_cm = plot_confusion_matrix(y_test, gb_pred, "Gradient Boosting")

# Detailed classification reports
print("\n📋 DETAILED CLASSIFICATION REPORTS")
print("=" * 60)

print("\n🚀 XGBoost Classification Report:")
print(classification_report(y_test, xgb_pred))

print("\n🌲 Random Forest Classification Report:")
print(classification_report(y_test, rf_pred))

print("\n⚡ Gradient Boosting Classification Report:")
print(classification_report(y_test, gb_pred))

# Choose best model based on F1-score (good for imbalanced data)
best_model_idx = metrics_df['f1_score'].idxmax()
best_model_name = metrics_df.loc[best_model_idx, 'model_name']
best_f1_score = metrics_df.loc[best_model_idx, 'f1_score']

if best_model_name == "XGBoost":
    best_model, best_pred, best_pred_proba = xgb_model, xgb_pred, xgb_pred_proba
elif best_model_name == "Random Forest":
    best_model, best_pred, best_pred_proba = rf_model, rf_pred, rf_pred_proba
else:
    best_model, best_pred, best_pred_proba = gb_model, gb_pred, gb_pred_proba

print(f"\n🏆 BEST MODEL: {best_model_name} (F1-Score: {best_f1_score:.4f})")

# Cross-validation for best model
print(f"\n🔄 Cross-validation for {best_model_name}:")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores_accuracy = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy')
cv_scores_f1 = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='f1')
cv_scores_roc_auc = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='roc_auc')

print(f"Accuracy: {cv_scores_accuracy.mean():.3f} (+/- {cv_scores_accuracy.std() * 2:.3f})")
print(f"F1-Score: {cv_scores_f1.mean():.3f} (+/- {cv_scores_f1.std() * 2:.3f})")
print(f"ROC-AUC: {cv_scores_roc_auc.mean():.3f} (+/- {cv_scores_roc_auc.std() * 2:.3f})")

# HOUR 7-8: VALIDATION AND TESTING
print("\n📊 PHASE 4: VALIDATION (Hours 7-8)")
print("-" * 40)

# Create comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Metrics Comparison Bar Plot
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
metrics_comparison = metrics_df[['model_name'] + metrics_to_plot].set_index('model_name')

axes[0, 0].bar(range(len(metrics_comparison.columns)), metrics_comparison.iloc[0], 
               alpha=0.7, label='XGBoost', color='blue')
axes[0, 0].bar(range(len(metrics_comparison.columns)), metrics_comparison.iloc[1], 
               alpha=0.7, label='Random Forest', color='green')
axes[0, 0].bar(range(len(metrics_comparison.columns)), metrics_comparison.iloc[2], 
               alpha=0.7, label='Gradient Boosting', color='orange')
axes[0, 0].set_xticks(range(len(metrics_comparison.columns)))
axes[0, 0].set_xticklabels(metrics_comparison.columns, rotation=45)
axes[0, 0].set_ylabel('Score')
axes[0, 0].set_title('Model Performance Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. ROC Curves for all models
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_pred_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_pred_proba)
fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_pred_proba)

axes[0, 1].plot(fpr_xgb, tpr_xgb, linewidth=2, label=f'XGBoost (AUC = {xgb_metrics["roc_auc"]:.3f})')
axes[0, 1].plot(fpr_rf, tpr_rf, linewidth=2, label=f'Random Forest (AUC = {rf_metrics["roc_auc"]:.3f})')
axes[0, 1].plot(fpr_gb, tpr_gb, linewidth=2, label=f'Gradient Boosting (AUC = {gb_metrics["roc_auc"]:.3f})')
axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[0, 1].set_xlim([0.0, 1.0])
axes[0, 1].set_ylim([0.0, 1.05])
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curves Comparison')
axes[0, 1].legend(loc="lower right")
axes[0, 1].grid(True, alpha=0.3)

# 3. Feature Importance for best model
importance = best_model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': importance
}).sort_values('importance', ascending=False)

top_features = feature_importance.head(10)
axes[1, 0].barh(range(len(top_features)), top_features['importance'])
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['feature'])
axes[1, 0].set_xlabel('Feature Importance')
axes[1, 0].set_title(f'Top 10 Feature Importance - {best_model_name}')
axes[1, 0].invert_yaxis()
axes[1, 0].grid(True, alpha=0.3)

# 4. Model Performance Heatmap
performance_matrix = metrics_df.set_index('model_name')[metrics_to_plot].T
im = axes[1, 1].imshow(performance_matrix.values, cmap='YlOrRd', aspect='auto')
axes[1, 1].set_xticks(range(len(performance_matrix.columns)))
axes[1, 1].set_yticks(range(len(performance_matrix.index)))
axes[1, 1].set_xticklabels(performance_matrix.columns)
axes[1, 1].set_yticklabels(performance_matrix.index)
axes[1, 1].set_title('Performance Metrics Heatmap')

# Add text annotations
for i in range(len(performance_matrix.index)):
    for j in range(len(performance_matrix.columns)):
        text = axes[1, 1].text(j, i, f'{performance_matrix.iloc[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im, ax=axes[1, 1])
plt.tight_layout()
plt.savefig('comprehensive_model_analysis.png', dpi=300, bbox_inches='tight')
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
    'model_name': best_model_name,
    'performance': {
        'accuracy': metrics_df.loc[best_model_idx, 'accuracy'],
        'precision': metrics_df.loc[best_model_idx, 'precision'],
        'recall': metrics_df.loc[best_model_idx, 'recall'],
        'f1_score': metrics_df.loc[best_model_idx, 'f1_score'],
        'roc_auc': metrics_df.loc[best_model_idx, 'roc_auc'],
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    },
    'all_model_metrics': metrics_df.to_dict('records')
}

with open('drug_food_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print("✅ Model saved successfully!")

# FINAL SUMMARY
print("\n🎉 12-HOUR SPRINT SUMMARY")
print("=" * 60)
print(f"✅ Dataset processed: {len(df_final):,} samples")
print(f"✅ Features engineered: {len(feature_cols)}")
print(f"✅ Best model: {best_model_name}")  # Changed from best_name to best_model_name
print(f"✅ ROC-AUC Score: {metrics_df.loc[best_model_idx, 'roc_auc']:.3f}")  # Changed from best_auc
print(f"✅ Model saved: drug_food_model.pkl")
print(f"✅ Plots saved: confusion_matrix.png, roc_curve.png, feature_importance.png")
print("\n🚀 Ready for API development in Hours 9-10!")

# Additional debugging info
print("\n🔍 DEBUG INFO:")
print(f"Final X shape: {X.shape}")
print(f"Final X dtypes: {X.dtypes.value_counts()}")
print(f"Any NaN values: {X.isnull().sum().sum()}")
print(f"Feature columns: {len(feature_cols)}")
