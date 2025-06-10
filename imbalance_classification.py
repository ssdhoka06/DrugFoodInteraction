import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_and_model_foodrugs(foodrugs_path, merged_data_path, final_features_path=None):
    """
    Comprehensive modeling using both FoodDrugs and merged_drug_food_data datasets
    """
    print("🔄 Loading datasets...")
    
    # Load FoodDrugs data
    foodrugs = pd.read_csv('/Users/sachidhoka/Desktop/processed_drug_food_interactions.csv')
    print(f"✅ Loaded FoodDrugs: {len(foodrugs)} drug-food interactions")
    
    # Load merged_drug_food_data
    merged_data = pd.read_csv('/Users/sachidhoka/Desktop/merged_drug_food_data.csv')
    print(f"✅ Loaded Merged Data: {len(merged_data)} drug-food interactions")
    
    # Combine datasets
    combined_data = combine_datasets(foodrugs, merged_data)
    print(f"🔗 Combined dataset: {len(combined_data)} total interactions")
    
    # Analyze distribution
    print("\n📊 Combined Distribution:")
    severity_dist = combined_data['severity_label'].value_counts()
    print(severity_dist)
    
    # Load features if available
    if final_features_path:
        try:
            final_features = np.load(final_features_path)
            print(f"✅ Loaded features: {final_features.shape}")
        except:
            print("⚠️ Could not load final_features.npy, will use text-based features")
            final_features = None
    else:
        final_features = None
    
    # Create enhanced dataset with known dangerous interactions
    enhanced_data = add_known_interactions(combined_data)
    
    print(f"\n📈 Enhanced dataset: {len(enhanced_data)} interactions")
    print("Enhanced distribution:")
    print(enhanced_data['severity_label'].value_counts())
    
    # Create features
    X = create_enhanced_features(enhanced_data)
    
    # Prepare labels for two-stage approach
    # Stage 1: Binary (Risk vs No-Risk)
    enhanced_data['binary_risk'] = (enhanced_data['severity_encoded'] > 0).astype(int)
    y_binary = enhanced_data['binary_risk'].values
    
    # Stage 2: Severity (Moderate vs High for risk samples only)  
    risk_mask = enhanced_data['severity_encoded'] > 0
    X_risk = X[risk_mask]
    y_severity = enhanced_data.loc[risk_mask, 'severity_encoded'].values
    
    print(f"\n🎯 Training Data Prepared:")
    print(f"Total samples: {len(X)}")
    print(f"Binary labels distribution: {np.bincount(y_binary)}")
    print(f"Risk samples: {len(X_risk)}")
    print(f"Severity distribution: {np.bincount(y_severity)}")
    
    # Train Stage 1: Binary classifier
    stage1_model, stage1_results = train_binary_classifier(X, y_binary)
    
    # Train Stage 2: Severity classifier (if enough samples)
    if len(np.unique(y_severity)) > 1 and len(y_severity) > 10:
        stage2_model, stage2_results = train_severity_classifier(X_risk, y_severity)
    else:
        print("⚠️ Insufficient samples for Stage 2 training")
        stage2_model, stage2_results = None, None
    
    # Test the complete pipeline
    test_complete_pipeline(enhanced_data, stage1_model, stage2_model)
    
    return stage1_model, stage2_model, enhanced_data

def combine_datasets(foodrugs_df, merged_df):
    """
    Intelligently combine FoodDrugs and merged_drug_food_data datasets
    """
    print("\n🔗 Combining datasets...")
    
    # Standardize column names for FoodDrugs
    foodrugs_standardized = foodrugs_df.copy()
    
    # Ensure consistent column naming
    if 'drug_name' not in foodrugs_standardized.columns and 'Drug' in foodrugs_standardized.columns:
        foodrugs_standardized = foodrugs_standardized.rename(columns={'Drug': 'drug_name'})
    if 'food_name' not in foodrugs_standardized.columns and 'Food' in foodrugs_standardized.columns:
        foodrugs_standardized = foodrugs_standardized.rename(columns={'Food': 'food_name'})
    if 'interaction_description' not in foodrugs_standardized.columns and 'Description' in foodrugs_standardized.columns:
        foodrugs_standardized = foodrugs_standardized.rename(columns={'Description': 'interaction_description'})
    
    # Standardize column names for merged data
    merged_standardized = merged_df.copy()
    
    # Check and rename common column variations
    column_mapping = {
        'Drug': 'drug_name',
        'drug': 'drug_name',
        'Drug_Name': 'drug_name',
        'Food': 'food_name', 
        'food': 'food_name',
        'Food_Name': 'food_name',
        'Description': 'interaction_description',
        'description': 'interaction_description',
        'Interaction_Description': 'interaction_description',
        'severity': 'severity_label',
        'Severity': 'severity_label',
        'severity_level': 'severity_label',
        'risk_level': 'severity_label'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in merged_standardized.columns:
            merged_standardized = merged_standardized.rename(columns={old_name: new_name})
    
    print(f"FoodDrugs columns: {list(foodrugs_standardized.columns)}")
    print(f"Merged data columns: {list(merged_standardized.columns)}")
    
    # Ensure severity encoding exists
    foodrugs_standardized = ensure_severity_encoding(foodrugs_standardized)
    merged_standardized = ensure_severity_encoding(merged_standardized)
    
    # Add source identifier
    foodrugs_standardized['data_source'] = 'foodrugs'
    merged_standardized['data_source'] = 'merged_data'
    
    # Combine datasets
    # First, get common columns
    common_cols = ['drug_name', 'food_name', 'interaction_description', 'severity_label', 'severity_encoded', 'data_source']
    
    # Ensure both datasets have these columns
    for col in common_cols:
        if col not in foodrugs_standardized.columns:
            if col == 'interaction_description':
                foodrugs_standardized[col] = "Unknown interaction"
            elif col == 'severity_label':
                foodrugs_standardized[col] = "no-risk"
            elif col == 'severity_encoded':
                foodrugs_standardized[col] = 0
            else:
                foodrugs_standardized[col] = ""
                
        if col not in merged_standardized.columns:
            if col == 'interaction_description':
                merged_standardized[col] = "Unknown interaction"
            elif col == 'severity_label':
                merged_standardized[col] = "no-risk"
            elif col == 'severity_encoded':
                merged_standardized[col] = 0
            else:
                merged_standardized[col] = ""
    
    # Select common columns and combine
    foodrugs_subset = foodrugs_standardized[common_cols]
    merged_subset = merged_standardized[common_cols]
    
    combined_df = pd.concat([foodrugs_subset, merged_subset], ignore_index=True)
    
    # Remove duplicates based on drug-food pairs
    print(f"Before deduplication: {len(combined_df)} interactions")
    combined_df = combined_df.drop_duplicates(subset=['drug_name', 'food_name'], keep='first')
    print(f"After deduplication: {len(combined_df)} interactions")
    
    # Clean data
    combined_df = clean_combined_data(combined_df)
    
    return combined_df

def ensure_severity_encoding(df):
    """
    Ensure severity_label and severity_encoded columns exist
    """
    # Create severity_label if missing
    if 'severity_label' not in df.columns:
        # Try to infer from other columns
        if 'severity' in df.columns:
            df['severity_label'] = df['severity']
        elif 'risk_level' in df.columns:
            df['severity_label'] = df['risk_level']
        else:
            # Default to no-risk
            df['severity_label'] = 'no-risk'
    
    # Standardize severity labels
    severity_mapping = {
        'high': 'high-risk',
        'high-risk': 'high-risk',
        'high_risk': 'high-risk',
        'severe': 'high-risk',
        'major': 'high-risk',
        'moderate': 'moderate-risk',
        'moderate-risk': 'moderate-risk',
        'moderate_risk': 'moderate-risk',
        'medium': 'moderate-risk',
        'minor': 'moderate-risk',
        'low': 'no-risk',
        'low-risk': 'no-risk',
        'low_risk': 'no-risk',
        'none': 'no-risk',
        'no-risk': 'no-risk',
        'no_risk': 'no-risk',
        'safe': 'no-risk'
    }
    
    df['severity_label'] = df['severity_label'].astype(str).str.lower().map(severity_mapping).fillna('no-risk')
    
    # Create severity_encoded
    encoding_map = {
        'no-risk': 0,
        'moderate-risk': 1,
        'high-risk': 2
    }
    
    df['severity_encoded'] = df['severity_label'].map(encoding_map).fillna(0)
    
    return df

def clean_combined_data(df):
    """
    Clean and standardize the combined dataset
    """
    print("\n🧹 Cleaning combined data...")
    
    # Clean drug names
    df['drug_name'] = df['drug_name'].astype(str).str.lower().str.strip()
    df['drug_name'] = df['drug_name'].str.replace(r'[^\w\s-]', '', regex=True)
    
    # Clean food names
    df['food_name'] = df['food_name'].astype(str).str.lower().str.strip()
    df['food_name'] = df['food_name'].str.replace(r'[^\w\s-]', '', regex=True)
    
    # Clean descriptions
    df['interaction_description'] = df['interaction_description'].astype(str).str.strip()
    df['interaction_description'] = df['interaction_description'].fillna("Unknown interaction")
    
    # Remove invalid entries
    initial_len = len(df)
    df = df[df['drug_name'].str.len() > 0]
    df = df[df['food_name'].str.len() > 0]
    df = df[df['drug_name'] != 'nan']
    df = df[df['food_name'] != 'nan']
    
    print(f"Removed {initial_len - len(df)} invalid entries")
    
    return df

def add_known_interactions(combined_df):
    """Add curated high-risk and moderate-risk interactions"""
    
    # Create additional high-risk interactions
    high_risk_interactions = [
        # Warfarin + Vitamin K foods
        {"drug_name": "warfarin", "food_name": "spinach", "interaction_description": "Vitamin K interference with anticoagulation", "severity_label": "high-risk", "severity_encoded": 2},
        {"drug_name": "warfarin", "food_name": "kale", "interaction_description": "High vitamin K reduces warfarin effect", "severity_label": "high-risk", "severity_encoded": 2},
        {"drug_name": "warfarin", "food_name": "broccoli", "interaction_description": "Vitamin K antagonizes warfarin", "severity_label": "high-risk", "severity_encoded": 2},
        {"drug_name": "warfarin", "food_name": "brussels_sprouts", "interaction_description": "Vitamin K reduces anticoagulation", "severity_label": "high-risk", "severity_encoded": 2},
        
        # Grapefruit + CYP3A4 substrates
        {"drug_name": "atorvastatin", "food_name": "grapefruit", "interaction_description": "CYP3A4 inhibition increases statin levels", "severity_label": "high-risk", "severity_encoded": 2},
        {"drug_name": "simvastatin", "food_name": "grapefruit", "interaction_description": "Muscle toxicity risk via CYP3A4", "severity_label": "high-risk", "severity_encoded": 2},
        {"drug_name": "felodipine", "food_name": "grapefruit", "interaction_description": "Dramatic increase in drug levels", "severity_label": "high-risk", "severity_encoded": 2},
        {"drug_name": "cyclosporine", "food_name": "grapefruit", "interaction_description": "Increased immunosuppressant toxicity", "severity_label": "high-risk", "severity_encoded": 2},
        
        # MAOI + Tyramine
        {"drug_name": "phenelzine", "food_name": "aged_cheese", "interaction_description": "Hypertensive crisis from tyramine", "severity_label": "high-risk", "severity_encoded": 2},
        {"drug_name": "tranylcypromine", "food_name": "fermented_foods", "interaction_description": "Dangerous blood pressure elevation", "severity_label": "high-risk", "severity_encoded": 2},
        {"drug_name": "selegiline", "food_name": "red_wine", "interaction_description": "Tyramine-induced hypertensive crisis", "severity_label": "high-risk", "severity_encoded": 2},
    ]
    
    # Moderate-risk interactions
    moderate_risk_interactions = [
        # Antibiotics + Dairy
        {"drug_name": "ciprofloxacin", "food_name": "milk", "interaction_description": "Calcium reduces antibiotic absorption", "severity_label": "moderate-risk", "severity_encoded": 1},
        {"drug_name": "doxycycline", "food_name": "dairy", "interaction_description": "Chelation reduces drug efficacy", "severity_label": "moderate-risk", "severity_encoded": 1},
        {"drug_name": "tetracycline", "food_name": "yogurt", "interaction_description": "Calcium binding reduces absorption", "severity_label": "moderate-risk", "severity_encoded": 1},
        
        # Iron + Tannins
        {"drug_name": "iron_supplement", "food_name": "coffee", "interaction_description": "Tannins reduce iron absorption", "severity_label": "moderate-risk", "severity_encoded": 1},
        {"drug_name": "iron_supplement", "food_name": "tea", "interaction_description": "Polyphenols inhibit iron uptake", "severity_label": "moderate-risk", "severity_encoded": 1},
        
        # Other moderate interactions
        {"drug_name": "amlodipine", "food_name": "grapefruit", "interaction_description": "Moderate increase in drug levels", "severity_label": "moderate-risk", "severity_encoded": 1},
        {"drug_name": "levothyroxine", "food_name": "soy", "interaction_description": "Reduced thyroid hormone absorption", "severity_label": "moderate-risk", "severity_encoded": 1},
    ]
    
    # Convert to DataFrames and combine
    high_risk_df = pd.DataFrame(high_risk_interactions)
    moderate_risk_df = pd.DataFrame(moderate_risk_interactions)
    additional_interactions = pd.concat([high_risk_df, moderate_risk_df], ignore_index=True)
    
    # Add data source
    additional_interactions['data_source'] = 'curated'
    
    # Combine with original combined data
    enhanced_df = pd.concat([combined_df, additional_interactions], ignore_index=True)
    
    # Remove duplicates based on drug-food pairs, keeping first occurrence
    enhanced_df = enhanced_df.drop_duplicates(subset=['drug_name', 'food_name'], keep='first')
    
    return enhanced_df

def analyze_dataset_overlap(enhanced_data):
    """
    Analyze overlap and contribution of different data sources
    """
    print("\n📊 Dataset Analysis:")
    
    # Source distribution
    source_dist = enhanced_data['data_source'].value_counts()
    print(f"Data source distribution:")
    for source, count in source_dist.items():
        print(f"  {source}: {count} interactions")
    
    # Severity by source
    print(f"\nSeverity distribution by source:")
    severity_by_source = pd.crosstab(enhanced_data['data_source'], enhanced_data['severity_label'])
    print(severity_by_source)
    
    # Check for overlapping drug-food pairs between sources
    foodrugs_pairs = set()
    merged_pairs = set()
    
    for _, row in enhanced_data.iterrows():
        pair = (row['drug_name'], row['food_name'])
        if row['data_source'] == 'foodrugs':
            foodrugs_pairs.add(pair)
        elif row['data_source'] == 'merged_data':
            merged_pairs.add(pair)
    
    overlap = foodrugs_pairs.intersection(merged_pairs)
    print(f"\nOverlap analysis:")
    print(f"  FoodDrugs unique pairs: {len(foodrugs_pairs)}")
    print(f"  Merged data unique pairs: {len(merged_pairs)}")
    print(f"  Overlapping pairs: {len(overlap)}")
    print(f"  Total unique pairs: {len(foodrugs_pairs.union(merged_pairs))}")
    
    return {
        'source_distribution': source_dist,
        'severity_by_source': severity_by_source,
        'overlap_count': len(overlap),
        'unique_pairs': len(foodrugs_pairs.union(merged_pairs))
    }

def create_enhanced_features(df):
    """Create comprehensive features from drug-food interaction data"""
    
    features = []
    
    for _, row in df.iterrows():
        drug = str(row['drug_name']).lower()
        food = str(row['food_name']).lower()
        desc = str(row.get('interaction_description', '')).lower()
        
        # Drug category features
        drug_features = [
            int('warfarin' in drug),
            int(any(statin in drug for statin in ['statin', 'atorva', 'simva', 'rosuva'])),
            int(any(maoi in drug for maoi in ['phenelzine', 'tranylcypromine', 'selegiline'])),
            int(any(antibiotic in drug for antibiotic in ['cipro', 'doxy', 'tetra', 'antibiotic'])),
            int('iron' in drug or 'supplement' in drug),
            int('levothyroxine' in drug or 'thyroid' in drug),
            int(any(bp_med in drug for bp_med in ['amlodipine', 'felodipine', 'nifedipine'])),
            int('cyclosporine' in drug or 'immunosuppress' in drug),
        ]
        
        # Food category features  
        food_features = [
            int('grapefruit' in food),
            int(any(vit_k in food for vit_k in ['spinach', 'kale', 'broccoli', 'brussels_sprouts'])),
            int(any(dairy in food for dairy in ['milk', 'dairy', 'yogurt', 'cheese'])),
            int(any(fermented in food for fermented in ['aged_cheese', 'fermented', 'wine'])),
            int(any(caffeine in food for caffeine in ['coffee', 'tea'])),
            int('soy' in food),
            int(any(citrus in food for citrus in ['citrus', 'orange', 'lemon'])),
            int(any(leafy in food for leafy in ['lettuce', 'arugula', 'greens'])),
        ]
        
        # Mechanism-based features
        mechanism_features = [
            int(any(cyp in desc for cyp in ['cyp3a4', 'cyp', 'enzyme', 'metabolism'])),
            int(any(vit in desc for vit in ['vitamin k', 'vitamin', 'k interference'])),
            int(any(abs in desc for abs in ['absorption', 'chelation', 'binding'])),
            int(any(tyr in desc for tyr in ['tyramine', 'hypertensive', 'crisis'])),
            int(any(calc in desc for calc in ['calcium', 'mineral', 'chelat'])),
            int(any(level in desc for level in ['increase', 'decrease', 'level', 'concentration'])),
            int(any(toxic in desc for toxic in ['toxic', 'toxicity', 'adverse'])),
            int(any(red in desc for red in ['reduce', 'inhibit', 'block'])),
        ]
        
        # Severity indicators from text
        severity_features = [
            int(any(high in desc for high in ['dangerous', 'severe', 'crisis', 'dramatic'])),
            int(any(mod in desc for mod in ['moderate', 'mild', 'slight'])),
            int(len(desc.split())),  # Description length as complexity indicator
            int(any(clin in desc for clin in ['clinical', 'significant', 'important'])),
        ]
        
        # Data source features (new)
        source_features = [
            int(row.get('data_source', '') == 'foodrugs'),
            int(row.get('data_source', '') == 'merged_data'),
            int(row.get('data_source', '') == 'curated'),
        ]
        
        # Combine all features
        row_features = drug_features + food_features + mechanism_features + severity_features + source_features
        features.append(row_features)
    
    return np.array(features)

def train_binary_classifier(X, y):
    """Train binary classifier for risk detection"""
    print("\n🎯 Training Stage 1: Binary Risk Classifier")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Apply SMOTE for balancing
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"Original training: {np.bincount(y_train)}")
    print(f"After SMOTE: {np.bincount(y_train_balanced)}")
    
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    rf.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    
    print("\n📊 Stage 1 Results:")
    print(classification_report(y_test, y_pred, target_names=['No Risk', 'Risk']))
    
    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score: {f1:.3f}")
    
    # Feature importance
    feature_names = create_feature_names()
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 Top 10 Important Features:")
    print(importance_df.head(10))
    
    return rf, {
        'f1_score': f1,
        'classification_report': classification_report(y_test, y_pred, target_names=['No Risk', 'Risk']),
        'feature_importance': importance_df
    }

def train_severity_classifier(X, y):
    """Train severity classifier for risk stratification"""
    print("\n🎯 Training Stage 2: Severity Classifier")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Apply SMOTE if needed
    if len(np.unique(y_train)) > 1:
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"Original training: {np.bincount(y_train)}")
        print(f"After SMOTE: {np.bincount(y_train_balanced)}")
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    rf.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    
    severity_labels = ['Moderate Risk', 'High Risk']
    print("\n📊 Stage 2 Results:")
    print(classification_report(y_test, y_pred, target_names=severity_labels))
    
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Weighted F1 Score: {f1:.3f}")
    
    return rf, {
        'f1_score': f1,
        'classification_report': classification_report(y_test, y_pred, target_names=severity_labels)
    }

def create_feature_names():
    """Create feature names corresponding to the features"""
    drug_names = [
        'is_warfarin', 'is_statin', 'is_maoi', 'is_antibiotic', 
        'is_iron_supplement', 'is_thyroid_med', 'is_bp_med', 'is_immunosuppressant'
    ]
    
    food_names = [
        'is_grapefruit', 'is_vitamin_k_food', 'is_dairy', 'is_fermented',
        'is_caffeine', 'is_soy', 'is_citrus', 'is_leafy_green'
    ]
    
    mechanism_names = [
        'involves_cyp_enzyme', 'involves_vitamin_k', 'involves_absorption', 
        'involves_tyramine', 'involves_calcium', 'involves_level_change',
        'involves_toxicity', 'involves_reduction'
    ]
    
    severity_names = [
        'has_severe_keywords', 'has_moderate_keywords', 'description_length', 'has_clinical_keywords'
    ]
    
    source_names = [
        'from_foodrugs', 'from_merged_data', 'from_curated'
    ]
    
    return drug_names + food_names + mechanism_names + severity_names + source_names

def test_complete_pipeline(df, stage1_model, stage2_model=None):
    """Test the complete two-stage pipeline"""
    print("\n🧪 Testing Complete Pipeline")
    
    # Create test samples
    test_samples = [
        {"drug_name": "warfarin", "food_name": "spinach", "interaction_description": "Vitamin K reduces anticoagulation effect"},
        {"drug_name": "atorvastatin", "food_name": "grapefruit", "interaction_description": "CYP3A4 inhibition increases statin levels"},
        {"drug_name": "ciprofloxacin", "food_name": "milk", "interaction_description": "Calcium reduces absorption"},
        {"drug_name": "aspirin", "food_name": "apple", "interaction_description": "No known interaction"},
    ]
    
    for sample in test_samples:
        # Create feature vector
        temp_df = pd.DataFrame([sample])
        X_sample = create_enhanced_features(temp_df)
        
        # Stage 1: Risk detection
        risk_prob = stage1_model.predict_proba(X_sample)[0, 1]
        is_risk = stage1_model.predict(X_sample)[0]
        
        if is_risk and stage2_model is not None:
            # Stage 2: Severity classification
            severity = stage2_model.predict(X_sample)[0]
            severity_label = "High Risk" if severity == 2 else "Moderate Risk"
            print(f"🔍 {sample['drug_name']} + {sample['food_name']}: {severity_label} (Risk Prob: {risk_prob:.3f})")
        else:
            risk_label = "Risk Detected" if is_risk else "No Risk"
            print(f"🔍 {sample['drug_name']} + {sample['food_name']}: {risk_label} (Risk Prob: {risk_prob:.3f})")



def predict_interaction(drug_name, food_name, stage1_model, stage2_model=None, description=""):
    """Predict interaction for a new drug-food pair"""
    
    # Create sample
    sample = {
        "drug_name": drug_name,
        "food_name": food_name,
        "interaction_description": description
    }
    
    # Create features
    temp_df = pd.DataFrame([sample])
    X_sample = create_enhanced_features(temp_df)
    
    # Stage 1: Risk detection
    risk_prob = stage1_model.predict_proba(X_sample)[0, 1]
    is_risk = stage1_model.predict(X_sample)[0]
    
    result = {
        "drug": drug_name,
        "food": food_name,
        "risk_probability": risk_prob,
        "has_risk": bool(is_risk)
    }
    
    if is_risk and stage2_model is not None:
        # Stage 2: Severity classification
        severity = stage2_model.predict(X_sample)[0]
        severity_prob = stage2_model.predict_proba(X_sample)[0]
        result.update({
            "severity_level": int(severity),
            "severity_label": "High Risk" if severity == 2 else "Moderate Risk",
            "severity_probabilities": severity_prob.tolist()
        })
    
    return result

# Example usage function
def main():
    """Main function to run the analysis with both datasets"""
    
    # File paths - adjust as needed
    foodrugs_path = "/Users/sachidhoka/Desktop/processed_drug_food_interactions.csv"  # Your FoodDrugs dataset
    merged_data_path = "/Users/sachidhoka/Desktop/merged_drug_food_data.csv"  # Your merged dataset
    final_features_path = None  # Optional pre-computed features
    
    try:
        # Run the complete analysis with both datasets
        stage1_model, stage2_model, enhanced_data = analyze_and_model_foodrugs(
            foodrugs_path, merged_data_path, final_features_path
        )
        
        # Additional analysis of combined data
        print("\n📈 Combined Dataset Statistics:")
        print(f"Total interactions: {len(enhanced_data)}")
        print(f"Unique drugs: {enhanced_data['drug_name'].nunique()}")
        print(f"Unique foods: {enhanced_data['food_name'].nunique()}")
        
        # Show sample of high-risk interactions from both datasets
        high_risk_samples = enhanced_data[enhanced_data['severity_label'] == 'high-risk'].sample(
            min(10, len(enhanced_data[enhanced_data['severity_label'] == 'high-risk']))
        )
        print(f"\n🚨 Sample High-Risk Interactions:")
        for _, row in high_risk_samples.iterrows():
            print(f"  {row['drug_name']} + {row['food_name']} ({row['data_source']}): {row['interaction_description'][:100]}...")
        
        # Test predictions
        print("\n🔮 Example Predictions:")
        
        test_cases = [
            ("warfarin", "kale", "Patient on anticoagulant therapy"),
            ("simvastatin", "grapefruit_juice", "Statin medication with citrus"),
            ("amoxicillin", "yogurt", "Antibiotic with dairy product"),
            ("metformin", "broccoli", "Diabetes medication with vegetable"),
            ("phenelzine", "aged_cheese", "MAOI with tyramine-rich food"),
            ("ciprofloxacin", "milk", "Fluoroquinolone with calcium-rich food")
        ]
        
        for drug, food, desc in test_cases:
            result = predict_interaction(drug, food, stage1_model, stage2_model, desc)
            print(f"\n{drug} + {food}:")
            print(f"  Risk: {'Yes' if result['has_risk'] else 'No'} (P={result['risk_probability']:.3f})")
            if 'severity_label' in result:
                print(f"  Severity: {result['severity_label']}")
        
        return stage1_model, stage2_model, enhanced_data
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find required CSV file: {str(e)}")
        print("Please ensure both 'foodrugs.csv' and 'merged_drug_food_data.csv' exist in the current directory.")
        return None, None, None
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    main()