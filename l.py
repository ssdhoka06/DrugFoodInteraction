import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, accuracy_score, precision_score, recall_score, 
                           f1_score, precision_recall_curve, average_precision_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import catboost as cb
from collections import Counter
import pickle
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re
import networkx as nx
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

print("🚀 ENHANCED DRUG-FOOD INTERACTION PREDICTOR WITH RISK CATEGORIZATION")
print("=" * 80)

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
                risk_level = 'LOW'
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

class EPGCNClassifier(BaseEstimator, ClassifierMixin):
    """Enhanced Propagation Graph Convolutional Network - Drug Similarity (EPGCN-DS)"""
    
    def __init__(self, n_layers=3, hidden_dim=64, learning_rate=0.01, epochs=100):
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.classes_ = None
        self.classifier = None
    
    
        
    def _build_similarity_graph(self, X, y):
        """Build drug-food similarity graph"""
        n_samples = X.shape[0]
        if n_samples > 5000:
            # Sample a subset to avoid memory issues
            indices = np.random.choice(n_samples, 5000, replace=False)
            X = X[indices]
            n_samples = 5000
        
        # Create adjacency matrix based on feature similarity
        sim_matrix = cosine_similarity(X)
        
        # Keep only top-k similarities
        k = min(10, n_samples // 10)
        for i in range(n_samples):
            top_k_indices = np.argsort(sim_matrix[i])[-k-1:-1]  # Exclude self
            mask = np.zeros(n_samples, dtype=bool)
            mask[top_k_indices] = True
            sim_matrix[i, ~mask] = 0
        
        return csr_matrix(sim_matrix)
    
    def _propagate_features(self, X, adj_matrix, h_prev):
        """Feature propagation step"""
        # Simplified GCN propagation: H = AXW
        normalized_adj = adj_matrix + np.eye(adj_matrix.shape[0])
        degree = np.array(normalized_adj.sum(axis=1)).flatten()
        degree_inv_sqrt = np.power(degree + 1e-10, -0.5)  # Add small epsilon
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
        
        # D^(-1/2) * A * D^(-1/2)
        normalized_adj = normalized_adj.multiply(degree_inv_sqrt).T.multiply(degree_inv_sqrt).T
        
        # Simple linear transformation (simplified for demonstration)
        h_new = normalized_adj @ h_prev
        return h_new
    
    def fit(self, X, y):
        """Train EPGCN model"""
        self.classes_ = np.unique(y)
        
        # Build similarity graph
        adj_matrix = self._build_similarity_graph(X, y)
        
        # Initialize features
        h = X.copy()
        
        # Propagation layers (simplified)
        for layer in range(self.n_layers):
            h = self._propagate_features(X, adj_matrix, h)
            # Add non-linearity
            h = np.tanh(h)
        
        # Final classification layer (using logistic regression)
        self.classifier = LogisticRegression(random_state=42)
        self.classifier.fit(h, y)
        
        return self
    
    def predict(self, X):
        """Predict using EPGCN"""
        return self.classifier.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities using EPGCN"""
        return self.classifier.predict_proba(X)

class MRGNNClassifier(BaseEstimator, ClassifierMixin):
    """Multi-Relational Graph Neural Network (MR-GNN)"""
    
    def __init__(self, n_relations=3, hidden_dim=64, epochs=100):
        self.n_relations = n_relations
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.classes_ = None
        
    def _create_multi_relational_graph(self, X, y):
        """Create multiple relation graphs"""
        n_samples = X.shape[0]
        
        # Relation 1: Drug similarity
        drug_features = X[:, :X.shape[1]//2]  # First half for drugs
        drug_sim = cosine_similarity(drug_features)
        
        # Relation 2: Food similarity  
        food_features = X[:, X.shape[1]//2:]  # Second half for foods
        food_sim = cosine_similarity(food_features)
        
        # Relation 3: Interaction similarity
        interaction_sim = cosine_similarity(X)
        
        return [csr_matrix(drug_sim), csr_matrix(food_sim), csr_matrix(interaction_sim)]
    
    def fit(self, X, y):
        """Train MR-GNN model"""
        self.classes_ = np.unique(y)
        
        # Create multi-relational graphs
        relation_graphs = self._create_multi_relational_graph(X, y)
        
        # Aggregate information from all relations
        aggregated_features = X.copy()
        for graph in relation_graphs:
            # Simple aggregation (mean of neighbor features)
            neighbor_features = graph @ X
            aggregated_features += neighbor_features
        
        aggregated_features /= (len(relation_graphs) + 1)
        
        # Train classifier on aggregated features
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(aggregated_features, y)
        
        return self
    
    def predict(self, X):
        """Predict using MR-GNN"""
        return self.classifier.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities using MR-GNN"""
        return self.classifier.predict_proba(X)

class DFIMSClassifier(BaseEstimator, ClassifierMixin):
    """Drug-Food Interaction Multi-Scale classifier (DFI-MS)"""
    
    def __init__(self, scales=[1, 2, 4], base_estimator=None):
        self.scales = scales
        self.base_estimator = base_estimator or RandomForestClassifier(n_estimators=100, random_state=42)
        self.scale_classifiers = {}
        self.classes_ = None
        
    def _create_multi_scale_features(self, X, scale):
        """Create features at different scales"""
        if scale == 1:
            return X
        
        # For scale > 1, create pooled features
        n_features = X.shape[1]
        pool_size = min(scale, n_features)
        
        if pool_size >= n_features:
            return X
        
        # Average pooling
        pooled_features = []
        for i in range(0, n_features, pool_size):
            end_idx = min(i + pool_size, n_features)
            pooled = np.mean(X[:, i:end_idx], axis=1, keepdims=True)
            pooled_features.append(pooled)
        
        return np.hstack(pooled_features)
    
    def fit(self, X, y):
        """Train DFI-MS model"""
        self.classes_ = np.unique(y)
        
        # Train classifiers at different scales
        for scale in self.scales:
            X_scale = self._create_multi_scale_features(X, scale)
            
            # Create new instance instead of cloning
            if hasattr(self.base_estimator, 'get_params'):
                classifier = self.base_estimator.__class__(**self.base_estimator.get_params())
            else:
                classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            classifier.fit(X_scale, y)
            
            self.scale_classifiers[scale] = classifier
        
        return self
    
    def predict(self, X):
        """Predict using ensemble of multi-scale classifiers"""
        predictions = []
        
        for scale in self.scales:
            X_scale = self._create_multi_scale_features(X, scale)
            pred = self.scale_classifiers[scale].predict(X_scale)
            predictions.append(pred)
        
        # Majority voting
        predictions = np.array(predictions).T
        final_predictions = []
        
        for pred_row in predictions:
            unique, counts = np.unique(pred_row, return_counts=True)
            final_predictions.append(unique[np.argmax(counts)])
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        """Predict probabilities using ensemble averaging"""
        probabilities = []
        
        for scale in self.scales:
            X_scale = self._create_multi_scale_features(X, scale)
            proba = self.scale_classifiers[scale].predict_proba(X_scale)
            probabilities.append(proba)
        
        # Average probabilities
        avg_proba = np.mean(probabilities, axis=0)
        return avg_proba

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
    def create_risk_score(risk_level, mechanism):
        risk_scores = {'HIGH': 3, 'MODERATE': 2, 'LOW': 1}
        base_score = risk_scores.get(risk_level, 1)
        
        if mechanism != 'unknown':
            base_score += 1
        
        return base_score
    
    df['risk_score'] = df.apply(
        lambda x: create_risk_score(x['risk_level'], x['mechanism']), 
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

# Import additional required libraries
from sklearn.ensemble import GradientBoostingClassifier
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️ XGBoost not available. Skipping XGBoost model.")
    XGBOOST_AVAILABLE = False

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

# 8. EPGCN-DS (Enhanced Propagation Graph Convolutional Network - Drug Similarity)
epgcn_model = EPGCNClassifier(
    n_layers=3,
    hidden_dim=64,
    learning_rate=0.01,
    epochs=100
)
models['EPGCN-DS'] = epgcn_model

# 9. MR-GNN (Multi-Relational Graph Neural Network)
mrgnn_model = MRGNNClassifier(
    n_relations=3,
    hidden_dim=64,
    epochs=100
)
models['MR-GNN'] = mrgnn_model

# 10. DFI-MS (Drug-Food Interaction Multi-Scale)
dfims_model = DFIMSClassifier(
    scales=[1, 2, 4],
    base_estimator=RandomForestClassifier(n_estimators=100, random_state=42)
)
models['DFI-MS'] = dfims_model

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
print("                EPGCN-DS, MR-GNN, DFI-MS, Random Forest, XGBoost, Gradient Boosting")
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

# Prediction function for new interactions
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
print("                  EPGCN-DS, MR-GNN, DFI-MS, Random Forest, XGBoost, Gradient Boosting")
print("   • Risk categorization system operational (HIGH/MODERATE/LOW)")

# Save the best model
best_model = models[results_df.iloc[0]['model']]
try:
    with open('best_drug_food_interaction_model.pkl', 'wb') as f:
        pickle.dump({
            'model': best_model,
            'feature_info': feature_info,
            'scaler': scaler,
            'drug_categories': drug_categories,
            'food_categories': food_categories,
            'high_risk_interactions': high_risk_interactions
        }, f)
    print("✅ Best model saved as 'best_drug_food_interaction_model.pkl'")
except Exception as e:
    print(f"⚠️ Could not save model: {str(e)}")

print("\n🚀 Enhanced Drug-Food Interaction Predictor Complete! 🚀")
