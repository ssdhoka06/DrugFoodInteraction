

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from fuzzywuzzy import fuzz, process
import warnings
import subprocess
import sys
import os
import json
from typing import Dict, Tuple, List, Optional
import gc
warnings.filterwarnings('ignore')

class EnhancedDrugFoodFeatureFusion:
    def __init__(self, sample_size: Optional[int] = None):
        """
        Initialize the enhanced feature fusion pipeline
        
        Args:
            sample_size: If provided, will sample this many rows for memory efficiency
        """
        self.drug_tokenizer = None
        self.drug_model = None
        self.food_model = None
        self.scaler = StandardScaler()
        self.device = None
        self.sample_size = sample_size
        
        # Memory management settings
        self.batch_size = 16  # Conservative batch size
        self.max_length = 256  # Reduced for memory efficiency
        
    def setup_device(self):
        """Enhanced device setup with better compatibility"""
        print("🔧 Setting up compute device...")
        
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device('mps')
            print("✅ Using Apple Silicon MPS GPU acceleration!")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("✅ Using CUDA GPU acceleration!")
        else:
            self.device = torch.device('cpu')
            print("⚠️ Using CPU (consider GPU for faster processing)")
        
        return self.device
    
    def install_requirements(self):
        """Install required packages with better error handling"""
        required_packages = [
            'transformers==4.35.0',
            'sentence-transformers', 
            'fuzzywuzzy',
            'python-levenshtein',
            'scikit-learn',
            'torch',
            'numpy',
            'pandas'
        ]
        
        print("📦 Checking required packages...")
        for package in required_packages:
            try:
                package_name = package.split('==')[0]
                __import__(package_name.replace('-', '_'))
                print(f"✅ {package_name} already installed")
            except ImportError:
                try:
                    print(f"🔄 Installing {package}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    print(f"✅ {package} installed successfully")
                except Exception as e:
                    print(f"❌ Failed to install {package}: {e}")
                    return False
        return True
    
    def load_models(self):
        """Enhanced model loading with fallback options"""
        print("🧠 Loading transformer models...")
        
        if not self.install_requirements():
            raise RuntimeError("Failed to install required packages")
        
        self.setup_device()
        
        try:
            from transformers import AutoTokenizer, AutoModel
            from sentence_transformers import SentenceTransformer
            
            # Drug model loading with fallback hierarchy
            drug_models = [
                ('dmis-lab/biobert-base-cased-v1.1', 'BioBERT'),
                ('allenai/scibert_scivocab_uncased', 'SciBERT'),
                ('bert-base-uncased', 'BERT-base')
            ]
            
            for model_name, model_type in drug_models:
                try:
                    print(f"🔄 Loading {model_type}...")
                    self.drug_tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.drug_model = AutoModel.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        device_map=None
                    )
                    self.drug_model.eval()
                    self.drug_model.to(self.device)
                    print(f"✅ {model_type} loaded successfully")
                    break
                except Exception as e:
                    print(f"❌ {model_type} failed: {e}")
                    continue
            else:
                raise RuntimeError("All drug models failed to load")
            
            # Food model loading
            print("🔄 Loading Sentence-BERT for food embeddings...")
            self.food_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Sentence-BERT loaded successfully")
            
        except Exception as e:
            print(f"❌ Critical error loading models: {e}")
            raise
    
    def create_enhanced_food_mapping(self, foodb_df: pd.DataFrame) -> Dict[str, int]:
        """
        Create comprehensive food name mapping from FooDB
        """
        print("🔍 Creating enhanced food name mapping...")
        
        food_mapping = {}
        
        # Strategy 1: Look for direct name columns
        name_candidates = ['name', 'food_name', 'public_id', 'common_name']
        name_col = None
        
        for col in name_candidates:
            if col in foodb_df.columns:
                name_col = col
                print(f"Found name column: {col}")
                break
        
        if name_col:
            valid_rows = foodb_df[foodb_df[name_col].notna() & foodb_df['food_id'].notna()]
            for _, row in valid_rows.iterrows():
                food_name = str(row[name_col]).lower().strip()
                if food_name and food_name != 'nan':
                    food_mapping[food_name] = row['food_id']
        
        # Strategy 2: Create synthetic mapping from food_id if no names found
        if not food_mapping:
            print("⚠️ No name columns found, creating synthetic mapping...")
            for _, row in foodb_df.iterrows():
                if pd.notna(row['food_id']):
                    food_mapping[f"food_{row['food_id']}"] = row['food_id']
        
        print(f"✅ Created mapping for {len(food_mapping)} foods")
        return food_mapping
    
    def enhanced_fuzzy_matching(self, mapped_drugs: pd.DataFrame, food_mapping: Dict[str, int]) -> Dict[str, int]:
        """
        Enhanced fuzzy matching with multiple strategies
        """
        print("🎯 Performing enhanced fuzzy matching...")
        
        if not food_mapping:
            return {}
        
        unique_foods = mapped_drugs['food_name'].dropna().unique()
        fuzzy_matches = {}
        food_keys = list(food_mapping.keys())
        
        print(f"Processing {len(unique_foods)} unique food names...")
        
        for i, food_name in enumerate(unique_foods):
            if i % 1000 == 0:
                print(f"  Progress: {i}/{len(unique_foods)}")
            
            food_clean = str(food_name).lower().strip()
            
            # Strategy 1: Exact match
            if food_clean in food_mapping:
                fuzzy_matches[food_name] = food_mapping[food_clean]
                continue
            
            # Strategy 2: Fuzzy matching with multiple scorers
            try:
                scorers = [fuzz.ratio, fuzz.partial_ratio, fuzz.token_sort_ratio]
                best_match = None
                best_score = 0
                
                for scorer in scorers:
                    match = process.extractOne(food_clean, food_keys, scorer=scorer)
                    if match and match[1] > best_score:
                        best_match = match
                        best_score = match[1]
                
                if best_match and best_score >= 70:  # Threshold
                    fuzzy_matches[food_name] = food_mapping[best_match[0]]
                    
            except Exception as e:
                continue
        
        print(f"✅ Successfully matched {len(fuzzy_matches)}/{len(unique_foods)} foods")
        return fuzzy_matches
    
    def merge_datasets_enhanced(self, mapped_drugs: pd.DataFrame, drugbank_df: pd.DataFrame, foodb_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced dataset merging with better diagnostics
        """
        print("🔗 Enhanced dataset merging...")
        
        # Sample data if requested for memory management
        if self.sample_size and len(mapped_drugs) > self.sample_size:
            print(f"📊 Sampling {self.sample_size} rows for memory efficiency...")
            mapped_drugs = mapped_drugs.sample(n=self.sample_size, random_state=42)
        
        merged_df = mapped_drugs.copy()
        
        # DrugBank merge
        print("🔄 Merging with DrugBank...")
        if 'drugbank_id' in merged_df.columns and 'primary_id' in drugbank_df.columns:
            before_shape = merged_df.shape
            merged_df = merged_df.merge(
                drugbank_df, 
                left_on='drugbank_id', 
                right_on='primary_id', 
                how='left'
            )
            drugbank_matches = (~merged_df['primary_id'].isna()).sum()
            print(f"  ✅ DrugBank merge: {before_shape} → {merged_df.shape}")
            print(f"  📊 Matches: {drugbank_matches}/{len(merged_df)} ({100*drugbank_matches/len(merged_df):.1f}%)")
        
        # Enhanced FooDB merge
        print("🔄 Merging with FooDB...")
        food_mapping = self.create_enhanced_food_mapping(foodb_df)
        
        if food_mapping and 'food_name' in merged_df.columns:
            fuzzy_matches = self.enhanced_fuzzy_matching(merged_df, food_mapping)
            
            if fuzzy_matches:
                merged_df['matched_food_id'] = merged_df['food_name'].map(fuzzy_matches)
                
                before_shape = merged_df.shape
                merged_df = merged_df.merge(
                    foodb_df,
                    left_on='matched_food_id',
                    right_on='food_id',
                    how='left'
                )
                
                foodb_matches = (~merged_df['food_id'].isna()).sum()
                print(f"  ✅ FooDB merge: {before_shape} → {merged_df.shape}")
                print(f"  📊 Matches: {foodb_matches}/{len(merged_df)} ({100*foodb_matches/len(merged_df):.1f}%)")
        
        # Add missing interaction columns with defaults
        interaction_cols = {
            'cyp_interactor': 0,
            'grapefruit_like': 0,
            'interacts_with_cyp3a4': 0,
            'interacts_with_maoi': 0,
            'interacts_with_warfarin': 0,
            'pathway_activation': 0,
            'potential_drug_interactor': 0
        }
        
        for col, default_val in interaction_cols.items():
            if col not in merged_df.columns:
                merged_df[col] = default_val
        
        print(f"📊 Final merged dataset: {merged_df.shape}")
        return merged_df
    
    def create_drug_embeddings_optimized(self, df: pd.DataFrame) -> np.ndarray:
        """
        Memory-optimized drug embedding creation
        """
        print("🧬 Creating optimized drug embeddings...")
        
        # Prepare drug texts
        drug_texts = []
        for _, row in df.iterrows():
            text_parts = []
            
            # Collect available text fields
            text_fields = ['mechanism-of-action', 'pharmacodynamics', 'description', 'indication']
            for field in text_fields:
                if field in df.columns and pd.notna(row.get(field)):
                    text_parts.append(str(row[field]))
            
            # Fallback to drug name if no detailed text
            if not text_parts:
                text_parts.append(f"Drug: {row.get('drug_name', 'Unknown')}")
            
            combined_text = ' '.join(text_parts)[:1000]  # Limit length
            drug_texts.append(combined_text)
        
        # Process in batches
        embeddings = []
        batch_size = self.batch_size
        
        print(f"Processing {len(drug_texts)} texts in batches of {batch_size}...")
        
        for i in range(0, len(drug_texts), batch_size):
            batch_texts = drug_texts[i:i+batch_size]
            
            try:
                inputs = self.drug_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.drug_model(**inputs)
                    # Use mean pooling over all tokens
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                    embeddings.append(batch_embeddings.cpu().numpy())
                
            except Exception as e:
                print(f"⚠️ Batch {i//batch_size + 1} failed: {e}")
                # Create zero embeddings for failed batch
                batch_embeddings = np.zeros((len(batch_texts), 768))
                embeddings.append(batch_embeddings)
            
            # Memory cleanup
            if i % (batch_size * 5) == 0:
                self._clear_gpu_cache()
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Progress: {i//batch_size + 1}/{(len(drug_texts) + batch_size - 1)//batch_size} batches")
        
        drug_embeddings = np.vstack(embeddings)
        print(f"✅ Drug embeddings shape: {drug_embeddings.shape}")
        return drug_embeddings
    
    def create_food_embeddings_optimized(self, df: pd.DataFrame) -> np.ndarray:
        """
        Optimized food embedding creation
        """
        print("🍎 Creating optimized food embeddings...")
        
        # Prepare food texts
        food_texts = []
        for _, row in df.iterrows():
            text_parts = []
            
            # Collect available text fields
            text_fields = ['health_effects', 'potential_drug_interactor', 'interaction_description']
            for field in text_fields:
                if field in df.columns and pd.notna(row.get(field)):
                    text_parts.append(str(row[field]))
            
            # Add interaction flags as text
            flag_fields = ['cyp_interactor', 'grapefruit_like', 'interacts_with_cyp3a4']
            for field in flag_fields:
                if field in df.columns and row.get(field, 0) == 1:
                    text_parts.append(f"interacts with {field.replace('_', ' ')}")
            
            # Fallback to food name
            if not text_parts:
                text_parts.append(f"Food: {row.get('food_name', 'Unknown')}")
            
            combined_text = ' '.join(text_parts)[:500]  # Limit length
            food_texts.append(combined_text)
        
        # Create embeddings
        print(f"Processing {len(food_texts)} food texts...")
        food_embeddings = self.food_model.encode(
            food_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"✅ Food embeddings shape: {food_embeddings.shape}")
        return food_embeddings
    
    def extract_enhanced_numerical_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Extract comprehensive numerical features
        """
        print("🔢 Extracting enhanced numerical features...")
        
        # DrugBank numerical features
        drugbank_features = [
            'molecular_weight', 'logp', 'h_bond_donors', 'h_bond_acceptors',
            'polar_surface_area', 'rotatable_bonds', 'bioavailability',
            'half-life_numeric', 'protein-binding_numeric'
        ]
        
        # FooDB interaction flags
        interaction_features = [
            'cyp_interactor', 'grapefruit_like', 'interacts_with_cyp3a4',
            'interacts_with_maoi', 'interacts_with_warfarin',
            'pathway_activation', 'potential_drug_interactor'
        ]
        
        # Severity encoding
        severity_features = ['severity_encoded'] if 'severity_encoded' in df.columns else []
        
        # Collect available features
        all_features = drugbank_features + interaction_features + severity_features
        available_features = [f for f in all_features if f in df.columns]
        
        print(f"Available features: {len(available_features)}/{len(all_features)}")
        print(f"Features: {available_features}")
        
        if available_features:
            feature_df = df[available_features].copy()
            
            # Handle missing values
            for col in feature_df.columns:
                if feature_df[col].dtype in ['float64', 'int64']:
                    feature_df[col] = feature_df[col].fillna(feature_df[col].median())
                else:
                    feature_df[col] = feature_df[col].fillna(0)
            
            numerical_features = self.scaler.fit_transform(feature_df.values)
        else:
            # Create dummy features if none available
            print("⚠️ No numerical features found, creating dummy features")
            numerical_features = np.ones((len(df), 1))
            available_features = ['dummy_feature']
        
        print(f"✅ Numerical features shape: {numerical_features.shape}")
        return numerical_features, available_features
    
    def _clear_gpu_cache(self):
        """Clear GPU cache safely"""
        try:
            if self.device.type == 'mps':
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
            elif self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        except:
            pass
    
    def create_final_features(self, drug_embeddings: np.ndarray, food_embeddings: np.ndarray, 
                            numerical_features: np.ndarray) -> np.ndarray:
        """
        Combine all features with proper alignment
        """
        print("🎯 Creating final feature matrix...")
        
        # Ensure same number of samples
        n_samples = min(len(drug_embeddings), len(food_embeddings), len(numerical_features))
        print(f"Aligning to {n_samples} samples")
        
        # Trim to same size
        drug_embeddings = drug_embeddings[:n_samples]
        food_embeddings = food_embeddings[:n_samples]
        numerical_features = numerical_features[:n_samples]
        
        # Combine features
        final_features = np.hstack([
            drug_embeddings,
            food_embeddings,
            numerical_features
        ])
        
        print(f"✅ Final features shape: {final_features.shape}")
        print(f"   • Drug embeddings: {drug_embeddings.shape[1]} dims")
        print(f"   • Food embeddings: {food_embeddings.shape[1]} dims")
        print(f"   • Numerical features: {numerical_features.shape[1]} dims")
        
        return final_features
    
    def run_complete_pipeline(self, mapped_drugs: pd.DataFrame, drugbank_df: pd.DataFrame, 
                             foodb_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame, Dict]:
        """
        Run the complete feature fusion pipeline
        """
        print("🚀 ENHANCED DRUG-FOOD FEATURE FUSION PIPELINE")
        print("=" * 60)
        
        try:
            # Step 1: Load models
            print("\n📦 Step 1: Loading Models")
            self.load_models()
            
            # Step 2: Merge datasets
            print("\n🔗 Step 2: Enhanced Dataset Merging")
            merged_df = self.merge_datasets_enhanced(mapped_drugs, drugbank_df, foodb_df)
            
            # Step 3: Create embeddings
            print("\n🧬 Step 3: Drug Embeddings")
            drug_embeddings = self.create_drug_embeddings_optimized(merged_df)
            
            print("\n🍎 Step 4: Food Embeddings")
            food_embeddings = self.create_food_embeddings_optimized(merged_df)
            
            # Step 5: Numerical features
            print("\n🔢 Step 5: Numerical Features")
            numerical_features, feature_names = self.extract_enhanced_numerical_features(merged_df)
            
            # Step 6: Final features
            print("\n🎯 Step 6: Final Feature Matrix")
            final_features = self.create_final_features(drug_embeddings, food_embeddings, numerical_features)
            
            # Create metadata
            feature_info = {
                'total_samples': final_features.shape[0],
                'total_features': final_features.shape[1],
                'drug_embedding_dims': drug_embeddings.shape[1],
                'food_embedding_dims': food_embeddings.shape[1],
                'numerical_feature_names': feature_names,
                'device_used': str(self.device),
                'model_info': {
                    'drug_model': 'BioBERT/SciBERT/BERT',
                    'food_model': 'Sentence-BERT (all-MiniLM-L6-v2)'
                },
                'dataset_info': {
                    'original_mapped_drugs': len(mapped_drugs),
                    'final_merged': len(merged_df),
                    'sample_size_used': self.sample_size
                }
            }
            
            print("\n" + "=" * 60)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"📊 Final dataset: {final_features.shape[0]} samples × {final_features.shape[1]} features")
            print(f"🔧 Device: {self.device}")
            print(f"💾 Memory efficient: {'Yes' if self.sample_size else 'No'}")
            
            return final_features, merged_df, feature_info
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """
    Main execution function with comprehensive error handling
    """
    print("🧬 Enhanced Drug-Food Interaction Feature Fusion")
    print("=" * 50)
    
    # Configuration
    file_paths = {
        'mapped_drugs': '/Users/sachidhoka/Desktop/mapped_interactions.csv',
        'drugbank_df': '/Users/sachidhoka/Desktop/processed_drugbank_data.csv',
        'foodb_df': '/Users/sachidhoka/Desktop/processed_foodb.csv'
    }
    
    # Check files
    missing_files = [path for path in file_paths.values() if not os.path.exists(path)]
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"   • {file}")
        return
    
    try:
        # Initialize pipeline (with sampling for memory efficiency)
        pipeline = EnhancedDrugFoodFeatureFusion(sample_size=50000)  # Adjust as needed
        
        # Load data
        print("📂 Loading datasets...")
        mapped_drugs = pd.read_csv(file_paths['mapped_drugs'])
        drugbank_df = pd.read_csv(file_paths['drugbank_df'])
        foodb_df = pd.read_csv(file_paths['foodb_df'])
        
        print(f"✅ Datasets loaded:")
        print(f"   • Mapped drugs: {mapped_drugs.shape}")
        print(f"   • DrugBank: {drugbank_df.shape}")
        print(f"   • FooDB: {foodb_df.shape}")
        
        # Run pipeline
        final_features, merged_df, feature_info = pipeline.run_complete_pipeline(
            mapped_drugs, drugbank_df, foodb_df
        )
        
        # Save results
        print("\n💾 Saving results...")
        output_files = {
            'final_features.npy': final_features,
            'merged_drug_food_data.csv': merged_df,
            'feature_info.json': feature_info
        }
        
        # Save numpy array
        np.save('final_features.npy', final_features)
        
        # Save merged dataframe
        merged_df.to_csv('merged_drug_food_data.csv', index=False)
        
        # Save feature info
        with open('feature_info.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print("✅ Files saved:")
        for filename in output_files.keys():
            print(f"   • {filename}")
        
        print("\n🎉 Enhanced Feature Fusion Pipeline Completed Successfully!")
        print(f"📊 Ready for ML modeling with {final_features.shape[0]} samples and {final_features.shape[1]} features")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Ensure all CSV files exist and are readable")
        print("2. Check available memory (reduce sample_size if needed)")
        print("3. Update packages: pip install --upgrade torch transformers sentence-transformers")
        print("4. Restart Python kernel if memory issues persist")


if __name__ == "__main__":
    main()
