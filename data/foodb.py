import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# 1. Setup GPU Acceleration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# 2. Data Loading
print("Loading data...")
df = pd.read_csv('/Users/sachidhoka/foodb_combined_drug_interaction_dataset.csv')
df['potential_drug_interactor'] = df['potential_drug_interactor'].fillna(0)
print(f"Dataset shape: {df.shape}")

# 3. Feature Engineering
print("Starting feature engineering...")

# A. Biochemical Interaction Features
print("Creating biochemical interaction features...")
df['cyp_interactor'] = df['interacting_enzymes'].str.contains('CYP', na=False).astype(int)
df['grapefruit_like'] = df['compound_name'].str.contains('flavonoid|furanocoumarin', regex=True, na=False).astype(int)

interaction_triggers = {
    'CYP3A4': ['bergamottin', 'naringin'],
    'MAOI': ['tyramine'],
    'Warfarin': ['vitamin_K']
}

for enzyme, compounds in interaction_triggers.items():
    df[f'interacts_with_{enzyme.lower()}'] = df['compound_name'].isin(compounds).astype(int)

print("Biochemical features created.")

# B. GPU-Accelerated Pathway Scoring
print("Creating pathway scoring features...")
pathway_scores = torch.tensor([0.8, 0.5, 0.6], device=device)  # P450, UGT, ABC

def calculate_pathway_activation(pathway_text):
    if pd.isna(pathway_text):
        return 0.0
    
    pathway_text = str(pathway_text)
    score = 0.0
    pathway_names = ['P450', 'UGT', 'ABC']
    
    for i, pathway in enumerate(pathway_names):
        if pathway in pathway_text:
            score += pathway_scores[i].item()
    
    return score

df['pathway_activation'] = df['metabolic_pathways'].apply(calculate_pathway_activation)
print("Pathway scoring completed.")

# C. Text Embeddings (with robust fallback)
print("Processing text features...")

def get_text_features(health_effects_series, max_features=50):
    """Get text features with multiple fallback options"""
    
    # First try: Sentence Transformers with CPU fallback
    try:
        print("Attempting sentence transformers...")
        from sentence_transformers import SentenceTransformer
        
        # Force CPU for sentence transformers to avoid MPS issues
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Process in smaller batches to avoid memory issues
        text_data = health_effects_series.fillna('').tolist()
        batch_size = 100
        embeddings_list = []
        
        for i in range(0, len(text_data), batch_size):
            batch = text_data[i:i + batch_size]
            batch_embeddings = model.encode(batch, convert_to_tensor=False, device='cpu')
            embeddings_list.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings_list)
        print(f"Sentence transformers successful: {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        print(f"Sentence transformers failed: {str(e)}")
        print("Falling back to TF-IDF...")
        
        # Fallback: TF-IDF
        try:
            vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(health_effects_series.fillna(''))
            embeddings = tfidf_matrix.toarray()
            print(f"TF-IDF successful: {embeddings.shape}")
            return embeddings
            
        except Exception as e2:
            print(f"TF-IDF also failed: {str(e2)}")
            print("Using simple text length features...")
            
            # Final fallback: Simple text features
            text_lengths = health_effects_series.fillna('').str.len().values.reshape(-1, 1)
            word_counts = health_effects_series.fillna('').str.split().str.len().fillna(0).values.reshape(-1, 1)
            embeddings = np.hstack([text_lengths, word_counts])
            
            # Pad to desired size
            if embeddings.shape[1] < max_features:
                padding = np.zeros((embeddings.shape[0], max_features - embeddings.shape[1]))
                embeddings = np.hstack([embeddings, padding])
            
            print(f"Simple text features: {embeddings.shape}")
            return embeddings

# Get text features
text_features = get_text_features(df['health_effects'])
text_cols = [f'text_feat_{i}' for i in range(text_features.shape[1])]
df[text_cols] = pd.DataFrame(text_features)

print("Text features completed.")

# D. Numerical Features
print("Processing numerical features...")
df['content'] = pd.to_numeric(df['content'], errors='coerce')
content_median = df['content'].median() if not df['content'].isna().all() else 0
df['log_content'] = np.log1p(df['content'].fillna(content_median))
print("Numerical features completed.")

# E. Categorical Encoding
print("Encoding categorical features...")
try:
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df['compound_class_encoded'] = encoder.fit_transform(df[['compound_class']]).ravel()
    print("Categorical encoding completed.")
except Exception as e:
    print(f"Ordinal encoding failed: {e}")
    print("Using label encoding fallback...")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['compound_class_encoded'] = le.fit_transform(df['compound_class'].fillna('Unknown'))

# 4. Final Feature Selection
print("Selecting final features...")

interaction_feature_cols = [f'interacts_with_{e.lower()}' for e in interaction_triggers.keys()]
features = df[[
    'cyp_interactor', 'grapefruit_like',
    *interaction_feature_cols,
    'pathway_activation',
    *text_cols,
    'log_content', 'compound_class_encoded',
    'potential_drug_interactor', 'food_id'
]].copy()

print(f"Final feature matrix shape: {features.shape}")

# 5. Validation Split (Stratified by Interaction)
print("Setting up cross-validation...")

X = features.drop(columns=['potential_drug_interactor', 'food_id'])
y = features['potential_drug_interactor'].astype(int)
groups = features['food_id']

print(f"Features shape: {X.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# Handle potential issues with StratifiedGroupKFold
try:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Running cross-validation splits...")
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        print(f"\nProcessing Fold {fold+1}:")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        print(f"Train set: {X_train.shape}, Target distribution: {y_train.value_counts().to_dict()}")
        print(f"Test set: {X_test.shape}, Target distribution: {y_test.value_counts().to_dict()}")
        
        # Convert to PyTorch tensors with error handling
        try:
            X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
            
            # Move to device
            X_train_tensor = X_train_tensor.to(device)
            y_train_tensor = y_train_tensor.to(device)
            X_test_tensor = X_test_tensor.to(device)
            y_test_tensor = y_test_tensor.to(device)
            
            print(f"Tensors created successfully on {device}")
            print(f"Train tensor shape: {X_train_tensor.shape}")
            print(f"Test tensor shape: {X_test_tensor.shape}")
            
        except Exception as e:
            print(f"Error creating tensors for fold {fold+1}: {e}")
            continue
        
        # Optional: Clear tensors to free memory
        del X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()

except Exception as e:
    print(f"Cross-validation setup failed: {e}")
    print("Proceeding with simple train-test split...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 6. Save Processed Data
print("\nSaving processed data...")
try:
    features.to_csv('processed_drug_interaction_features.csv', index=False)
    print("Successfully saved to 'processed_drug_interaction_features.csv'")
    
    # Also save feature names for reference
    feature_names = {
        'biochemical_features': ['cyp_interactor', 'grapefruit_like'] + interaction_feature_cols,
        'pathway_features': ['pathway_activation'],
        'text_features': text_cols,
        'numerical_features': ['log_content'],
        'categorical_features': ['compound_class_encoded'],
        'target': 'potential_drug_interactor',
        'group': 'food_id'
    }
    
    import json
    with open('feature_mapping.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    print("Feature mapping saved to 'feature_mapping.json'")
    
except Exception as e:
    print(f"Error saving files: {e}")

print(f"\nProcessing complete!")
print(f"Final dataset shape: {features.shape}")
print(f"Features ready for model training on {device}")

# Memory cleanup
if device.type == 'mps':
    torch.mps.empty_cache()
elif device.type == 'cuda':
    torch.cuda.empty_cache()