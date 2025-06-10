import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
import re
import warnings
warnings.filterwarnings('ignore')

class DrugFoodInteractionPipeline:
    def __init__(self):
        self.drug_rf_model = None
        self.food_autoencoder = None
        self.food_pca = None
        self.interaction_model = None
        self.drug_scaler = StandardScaler()
        self.food_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def verify_drugbank_features(self, drugbank_df):
        """Verify and engineer key pharmacodynamic features from DrugBank data"""
        print("=== DRUGBANK FEATURE VERIFICATION ===")
        
        # Check for existing CYP features
        existing_cyp_cols = [col for col in drugbank_df.columns if 'cyp' in col.lower()]
        
        # Engineer missing CYP features from metabolism text
        if 'metabolism' in drugbank_df.columns and 'cyp3a4_substrate' not in drugbank_df.columns:
            drugbank_df['cyp3a4_substrate'] = drugbank_df['metabolism'].str.contains(
                'CYP3A4|3A4|cytochrome P450 3A4', case=False, na=False
            ).astype(int)
            drugbank_df['cyp2d6_substrate'] = drugbank_df['metabolism'].str.contains(
                'CYP2D6|2D6|cytochrome P450 2D6', case=False, na=False
            ).astype(int)
            drugbank_df['cyp1a2_substrate'] = drugbank_df['metabolism'].str.contains(
                'CYP1A2|1A2|cytochrome P450 1A2', case=False, na=False
            ).astype(int)
        
        # Engineer P-glycoprotein feature
        if 'p_glycoprotein_substrate' not in drugbank_df.columns and 'mechanism-of-action' in drugbank_df.columns:
            drugbank_df['p_glycoprotein_substrate'] = drugbank_df['mechanism-of-action'].str.contains(
                'P-glycoprotein|P-gp|ABCB1|MDR1', case=False, na=False
            ).astype(int)
        
        # Verify key features
        key_features = {
            'logp': 'logp' in drugbank_df.columns,
            'bioavailability': 'bioavailability' in drugbank_df.columns,
            'cyp3a4_substrate': 'cyp3a4_substrate' in drugbank_df.columns,
            'cyp2d6_substrate': 'cyp2d6_substrate' in drugbank_df.columns,
            'cyp1a2_substrate': 'cyp1a2_substrate' in drugbank_df.columns,
            'p_glycoprotein_substrate': 'p_glycoprotein_substrate' in drugbank_df.columns
        }
        
        print("Key Features Status:")
        for feature, exists in key_features.items():
            print(f"  {feature}: {'✓' if exists else '✗'}")
        
        return drugbank_df
    
    def ensure_consistent_drug_ids(self, drugbank_df, foodrugs_df):
        """Match drug identifiers between DrugBank and FooDrugs"""
        print("\n=== DRUG ID CONSISTENCY CHECK ===")
        
        # Create comprehensive drug name mapping
        drug_map = {}
        for _, row in drugbank_df.iterrows():
            names = {
                row['name'].lower().strip(),
                row['primary_id'].lower(),
                *[x.strip().lower() for x in str(row.get('secondary_ids', '')).split(';') if x.strip()],
                *[x.strip().lower() for x in str(row.get('categories', '')).split(';') if x.strip()]
            }
            for name in names:
                if name and len(name) > 2:  # Exclude very short names
                    drug_map[name] = row['primary_id']
        
        # Map FooDrugs drug names to DrugBank IDs
        foodrugs_df['drug_name_clean'] = foodrugs_df['drug_name'].str.lower().str.strip()
        foodrugs_df['drugbank_id'] = foodrugs_df['drug_name_clean'].map(drug_map)
        
        # Calculate mapping success
        mapped = foodrugs_df['drugbank_id'].notna().sum()
        total = len(foodrugs_df)
        
        print(f"Drug Mapping Results:")
        print(f"  Total interactions: {total}")
        print(f"  Successfully mapped: {mapped} ({mapped/total:.1%})")
        print(f"  Unmapped interactions: {total - mapped}")
        
        # Show sample unmapped drugs
        unmapped = foodrugs_df[foodrugs_df['drugbank_id'].isna()]['drug_name'].unique()
        print(f"\nSample unmapped drugs: {unmapped[:10].tolist()}")
        
        return foodrugs_df
    
    def create_food_embeddings(self, foodb_df, embedding_dim=32):
        """Create food embeddings from FooDB features"""
        print("\n=== FOOD EMBEDDING CREATION ===")
        
        # Select relevant features
        feature_cols = []
        
        # CYP interaction features
        cyp_cols = [col for col in foodb_df.columns if 'cyp' in col.lower() or 'interact' in col.lower()]
        feature_cols.extend(cyp_cols)
        
        # Other important features
        important_cols = ['grapefruit_like', 'interacts_with_maoi', 'interacts_with_warfarin']
        feature_cols.extend([col for col in important_cols if col in foodb_df.columns])
        
        # Text features (if available)
        text_cols = [col for col in foodb_df.columns if 'text_feat' in col]
        if text_cols:
            feature_cols.extend(text_cols[:100])  # Use first 100 text features
        
        # Ensure we have features
        if not feature_cols:
            raise ValueError("No suitable features found in FooDB data")
        
        # Prepare feature matrix
        X_food = foodb_df[feature_cols].fillna(0).values
        X_food = self.food_scaler.fit_transform(X_food)
        
        # Dimensionality reduction
        self.food_pca = PCA(n_components=min(embedding_dim, X_food.shape[1]))
        pca_features = self.food_pca.fit_transform(X_food)
        
        # Train simple autoencoder
        self.food_autoencoder = MLPRegressor(
            hidden_layer_sizes=(64, embedding_dim, 64),
            max_iter=200,
            random_state=42
        )
        self.food_autoencoder.fit(pca_features, pca_features)
        
        # Generate final embeddings
        food_embeddings = self.food_autoencoder.predict(pca_features)
        
        print(f"Created food embeddings: {food_embeddings.shape}")
        print(f"Explained variance: {self.food_pca.explained_variance_ratio_.sum():.2f}")
        
        return food_embeddings
    
    def create_drug_embeddings(self, drugbank_df, embedding_dim=32):
        """Create drug embeddings using pharmacologically relevant features"""
        print("\n=== DRUG EMBEDDING CREATION ===")
        
        # Select important drug features
        feature_cols = [
            'logp', 'bioavailability', 'molecular_weight',
            'h_bond_donors', 'h_bond_acceptors', 'polar_surface_area',
            'cyp3a4_substrate', 'cyp2d6_substrate', 'cyp1a2_substrate',
            'p_glycoprotein_substrate'
        ]
        feature_cols = [col for col in feature_cols if col in drugbank_df.columns]
        
        # Prepare feature matrix
        X_drug = drugbank_df[feature_cols].fillna(0)
        X_drug = self.drug_scaler.fit_transform(X_drug)
        
        # Train RF model to get feature importance
        self.drug_rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Create synthetic target based on interaction potential
        y = (drugbank_df.filter(regex='cyp|p_glyco').sum(axis=1) > 0).astype(int)
        self.drug_rf_model.fit(X_drug, y)
        
        # Create embeddings using feature importance and predictions
        importances = self.drug_rf_model.feature_importances_
        probas = self.drug_rf_model.predict_proba(X_drug)
        
        # Combine into final embeddings
        drug_embeddings = np.column_stack([
            X_drug * importances,  # Weight features by importance
            probas,                # Interaction probabilities
            np.zeros((X_drug.shape[0], max(0, embedding_dim - X_drug.shape[1] - probas.shape[1])))
        ])[:, :embedding_dim]
        
        print(f"Created drug embeddings: {drug_embeddings.shape}")
        print("Feature importances:")
        for feat, imp in zip(feature_cols, importances):
            print(f"  {feat}: {imp:.4f}")
        
        return drug_embeddings
    
    def prepare_interaction_data(self, foodrugs_df, drug_embeddings, food_embeddings, drugbank_df, foodb_df):
        """Prepare final interaction dataset"""
        print("\n=== INTERACTION DATASET PREPARATION ===")
        
        # Create mappings
        drug_id_to_idx = {drug_id: idx for idx, drug_id in enumerate(drugbank_df['primary_id'])}
        food_name_to_idx = {name: idx for idx, name in enumerate(foodb_df['food_name'])}
        
        # Prepare samples
        X = []
        y = []
        
        for _, row in foodrugs_df.iterrows():
            if pd.isna(row['drugbank_id']):
                continue
                
            drug_idx = drug_id_to_idx.get(row['drugbank_id'])
            food_idx = food_name_to_idx.get(row['food_name'])
            
            if drug_idx is not None and food_idx is not None:
                combined = np.concatenate([
                    drug_embeddings[drug_idx],
                    food_embeddings[food_idx]
                ])
                X.append(combined)
                y.append(row['severity_encoded'])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Prepared {len(X)} interaction samples")
        print(f"Feature dimension: {X.shape[1]}")
        print("Class distribution:", dict(zip(*np.unique(y, return_counts=True))))
        
        return X, y
    
    def train_interaction_model(self, X, y):
        """Train final interaction classifier"""
        print("\n=== INTERACTION MODEL TRAINING ===")
        
        if len(X) == 0:
            print("Error: No valid interaction data available!")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.interaction_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
        self.interaction_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.interaction_model.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        return self.interaction_model
    
    def run_pipeline(self, drugbank_path, foodb_path, foodrugs_path):
        """Complete pipeline execution"""
        # Load data
        drugbank_df = pd.read_csv(drugbank_path)
        foodb_df = pd.read_csv(foodb_path)
        foodrugs_df = pd.read_csv(foodrugs_path)
        
        # 1. Verify DrugBank features
        drugbank_df = self.verify_drugbank_features(drugbank_df)
        
        # 2. Ensure consistent drug IDs
        foodrugs_df = self.ensure_consistent_drug_ids(drugbank_df, foodrugs_df)
        
        # 3. Create embeddings
        food_embeddings = self.create_food_embeddings(foodb_df)
        drug_embeddings = self.create_drug_embeddings(drugbank_df)
        
        # 4. Prepare interaction data
        X, y = self.prepare_interaction_data(
            foodrugs_df, drug_embeddings, food_embeddings, drugbank_df, foodb_df
        )
        
        # 5. Train model
        model = self.train_interaction_model(X, y)
        
        return {
            'drug_embeddings': drug_embeddings,
            'food_embeddings': food_embeddings,
            'interaction_model': model,
            'mapped_interactions': foodrugs_df
        }

# Example usage
if __name__ == "__main__":
    pipeline = DrugFoodInteractionPipeline()
    results = pipeline.run_pipeline(
        drugbank_path="/Users/sachidhoka/Desktop/processed_drugbank_data.csv",
        foodb_path="/Users/sachidhoka/Desktop/processed_drug_interaction_features.csv",
        foodrugs_path="/Users/sachidhoka/Desktop/processed_drug_food_interactions.csv"
    )