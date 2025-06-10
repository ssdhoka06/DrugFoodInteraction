# import pandas as pd
# import numpy as np
# import torch
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.feature_extraction.text import TfidfVectorizer
# import warnings
# import subprocess
# import sys
# import os
# warnings.filterwarnings('ignore')

# def install_package(package):
#     """Install a package if it's not already installed"""
#     try:
#         __import__(package.replace('-', '_'))
#         return True
#     except ImportError:
#         try:
#             print(f"Installing {package}...")
#             subprocess.check_call([sys.executable, "-m", "pip", "install", package])
#             return True
#         except Exception as e:
#             print(f"Failed to install {package}: {e}")
#             return False

# class DrugFoodFeatureFusion:
#     def __init__(self):
#         """
#         Initialize the feature fusion pipeline for drug-food interactions
#         """
#         self.drug_tokenizer = None
#         self.drug_model = None
#         self.food_model = None
#         self.scaler = StandardScaler()
#         self.pca = PCA(n_components=50)
#         self.device = None
#         self.use_fallback_embeddings = True  # Start with fallback by default
        
#     def load_models(self):
#         """
#         Load BioBERT for drug embeddings and Sentence-BERT for food embeddings
#         """
#         print("Attempting to load advanced models...")
        
#         # Try to install and load transformers
#         if install_package('transformers') and install_package('sentence-transformers'):
#             try:
#                 from transformers import AutoTokenizer, AutoModel
#                 from sentence_transformers import SentenceTransformer
                
#                 print("Loading BioBERT for drug embeddings...")
#                 try:
#                     self.drug_tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
#                     self.drug_model = AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
#                 except Exception as e:
#                     print(f"BioBERT not available ({e}), using standard BERT...")
#                     self.drug_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
#                     self.drug_model = AutoModel.from_pretrained('bert-base-uncased')
                
#                 print("Loading Sentence-BERT for food embeddings...")
#                 self.food_model = SentenceTransformer('all-MiniLM-L6-v2')
                
#                 # Set models to evaluation mode
#                 self.drug_model.eval()
                
#                 # Set device compatibility
#                 if torch.backends.mps.is_available():
#                     device = torch.device('mps')
#                     print("Using Apple Silicon MPS GPU acceleration!")
#                 elif torch.cuda.is_available():
#                     device = torch.device('cuda')
#                     print("Using CUDA GPU acceleration!")
#                 else:
#                     device = torch.device('cpu')
#                     print("Using CPU")
                
#                 self.drug_model.to(device)
#                 self.device = device
#                 self.use_fallback_embeddings = False
                
#                 print(f"Advanced models loaded successfully on device: {device}")
                
#             except Exception as e:
#                 print(f"Error loading advanced models: {e}")
#                 print("Falling back to TF-IDF embeddings...")
#                 self.use_fallback_embeddings = True
#         else:
#             print("Required packages not available, using TF-IDF embeddings...")
#             self.use_fallback_embeddings = True
        
#     def merge_datasets(self, mapped_drugs, drugbank_df, foodb_df):
#         """
#         Merge the three key datasets based on drugbank_id and food_name
#         """
#         print("Merging datasets...")
#         print(f"Mapped drugs shape: {mapped_drugs.shape}")
#         print(f"DrugBank shape: {drugbank_df.shape}")
#         print(f"FooDB shape: {foodb_df.shape}")
        
#         # Check available columns
#         print("\nAvailable columns:")
#         print(f"Mapped drugs: {list(mapped_drugs.columns)}")
#         print(f"DrugBank: {list(drugbank_df.columns)}")
#         print(f"FooDB: {list(foodb_df.columns)}")
        
#         # Start with mapped_drugs as base
#         merged_df = mapped_drugs.copy()
        
#         # Merge with drugbank features
#         if 'drugbank_id' in mapped_drugs.columns and 'primary_id' in drugbank_df.columns:
#             merged_df = merged_df.merge(
#                 drugbank_df, 
#                 left_on='drugbank_id', 
#                 right_on='primary_id', 
#                 how='left'
#             )
#             print(f"After drugbank merge: {merged_df.shape}")
#             drugbank_matches = (~merged_df['primary_id'].isna()).sum()
#             print(f"DrugBank matches: {drugbank_matches}/{len(merged_df)}")
#         else:
#             print("Warning: Cannot merge with DrugBank - missing required columns")
        
#         # Enhanced food name matching for FooDB
#         food_name_cols = ['food_name', 'name', 'food', 'food_item', 'common_name', 'ingredient_name']
#         foodb_food_col = None
        
#         # Find the food name column in FooDB
#         for col in food_name_cols:
#             if col in foodb_df.columns:
#                 foodb_food_col = col
#                 print(f"Found FooDB food column: {foodb_food_col}")
#                 break
        
#         # If no direct match, try to find columns with 'food' or 'name' in them
#         if foodb_food_col is None:
#             possible_cols = [col for col in foodb_df.columns if 'food' in col.lower() or 'name' in col.lower()]
#             if possible_cols:
#                 foodb_food_col = possible_cols[0]
#                 print(f"Using best guess FooDB food column: {foodb_food_col}")
        
#         # Attempt merge with FooDB
#         if foodb_food_col and 'food_name' in mapped_drugs.columns:
#             print(f"Merging with FooDB using column: {foodb_food_col}")
            
#             # Clean food names for better matching
#             merged_df_clean = merged_df.copy()
#             foodb_clean = foodb_df.copy()
            
#             # Normalize food names (lowercase, strip whitespace)
#             merged_df_clean['food_name_clean'] = merged_df_clean['food_name'].str.lower().str.strip()
#             foodb_clean[f'{foodb_food_col}_clean'] = foodb_clean[foodb_food_col].str.lower().str.strip()
            
#             # Try exact match first
#             final_df = merged_df_clean.merge(
#                 foodb_clean, 
#                 left_on='food_name_clean',
#                 right_on=f'{foodb_food_col}_clean',
#                 how='left'
#             )
            
#             # Check merge success
#             successful_matches = (~final_df[foodb_food_col].isna()).sum()
#             print(f"Successful FooDB matches: {successful_matches}/{len(final_df)}")
            
#             # If low match rate, try partial matching
#             if successful_matches < len(final_df) * 0.3:  # Less than 30% match rate
#                 print("Low match rate, attempting fuzzy matching...")
#                 final_df = self._fuzzy_food_matching(merged_df, foodb_df, foodb_food_col)
            
#             print(f"After foodb merge: {final_df.shape}")
#         else:
#             print("Warning: Cannot merge with FooDB - missing food name columns")
#             print(f"Available FooDB columns: {list(foodb_df.columns)}")
#             final_df = merged_df.copy()
        
#         print(f"\nFinal merged dataset shape: {final_df.shape}")
#         return final_df
    
#     def _fuzzy_food_matching(self, mapped_drugs, foodb_df, foodb_food_col):
#         """
#         Attempt fuzzy matching for food names when exact matching fails
#         """
#         print("Performing fuzzy food name matching...")
        
#         # Try to install fuzzywuzzy
#         if install_package('fuzzywuzzy') and install_package('python-Levenshtein'):
#             try:
#                 from fuzzywuzzy import fuzz, process
                
#                 # Get unique food names
#                 mapped_foods = mapped_drugs['food_name'].unique()
#                 foodb_foods = foodb_df[foodb_food_col].dropna().unique()
                
#                 # Create mapping dictionary
#                 food_mapping = {}
#                 for food in mapped_foods:
#                     if pd.notna(food):
#                         # Find best match
#                         match = process.extractOne(food, foodb_foods, scorer=fuzz.ratio)
#                         if match and match[1] > 70:  # 70% similarity threshold
#                             food_mapping[food] = match[0]
#                             print(f"Fuzzy match: '{food}' -> '{match[0]}' (score: {match[1]})")
                
#                 # Apply mapping
#                 mapped_drugs_fuzzy = mapped_drugs.copy()
#                 mapped_drugs_fuzzy['food_name_mapped'] = mapped_drugs_fuzzy['food_name'].map(food_mapping).fillna(mapped_drugs_fuzzy['food_name'])
                
#                 # Merge with mapped names
#                 final_df = mapped_drugs_fuzzy.merge(
#                     foodb_df,
#                     left_on='food_name_mapped',
#                     right_on=foodb_food_col,
#                     how='left'
#                 )
                
#                 successful_fuzzy_matches = (~final_df[foodb_food_col].isna()).sum()
#                 print(f"Fuzzy matching results: {successful_fuzzy_matches}/{len(final_df)} matches")
                
#                 return final_df
                
#             except Exception as e:
#                 print(f"Fuzzy matching failed: {e}")
#                 return mapped_drugs
#         else:
#             print("Fuzzy matching not available, skipping...")
#             return mapped_drugs
    
#     def create_drug_embeddings(self, df):
#         """
#         Create drug embeddings using BioBERT or TF-IDF fallback
#         """
#         print("Creating drug embeddings...")
        
#         # Combine mechanism-of-action and pharmacodynamics
#         drug_texts = []
#         for _, row in df.iterrows():
#             moa = str(row.get('mechanism-of-action', '')) if pd.notna(row.get('mechanism-of-action')) else ''
#             pharma = str(row.get('pharmacodynamics', '')) if pd.notna(row.get('pharmacodynamics')) else ''
#             desc = str(row.get('description', '')) if pd.notna(row.get('description')) else ''
#             combined_text = f"{moa} {pharma} {desc}".strip()
#             drug_texts.append(combined_text if combined_text else f"Drug: {row.get('drug_name', 'Unknown')}")
        
#         if not self.use_fallback_embeddings and self.drug_model is not None:
#             try:
#                 print("Using BioBERT embeddings for drugs...")
#                 embeddings = []
#                 batch_size = 32
                
#                 for i in range(0, len(drug_texts), batch_size):
#                     batch_texts = drug_texts[i:i+batch_size]
#                     inputs = self.drug_tokenizer(
#                         batch_texts, 
#                         padding=True, 
#                         truncation=True, 
#                         max_length=512,
#                         return_tensors='pt'
#                     ).to(self.device)
                    
#                     with torch.no_grad():
#                         outputs = self.drug_model(**inputs)
#                         batch_embeddings = outputs.last_hidden_state.mean(dim=1)
#                         embeddings.append(batch_embeddings.cpu().numpy())
                
#                 drug_embeddings = np.vstack(embeddings)
#                 print(f"BioBERT drug embeddings shape: {drug_embeddings.shape}")
#                 return drug_embeddings
                
#             except Exception as e:
#                 print(f"BioBERT embedding failed: {e}, falling back to TF-IDF...")
        
#         # TF-IDF fallback
#         print("Using TF-IDF embeddings for drugs...")
#         vectorizer = TfidfVectorizer(
#             max_features=300, 
#             stop_words='english',
#             ngram_range=(1, 2),
#             min_df=1  # Changed from 2 to 1 to handle small datasets
#         )
#         drug_embeddings = vectorizer.fit_transform(drug_texts).toarray()
        
#         print(f"TF-IDF drug embeddings shape: {drug_embeddings.shape}")
#         return drug_embeddings
    
#     def create_food_embeddings(self, df):
#         """
#         Create food embeddings using Sentence-BERT or TF-IDF fallback
#         """
#         print("Creating food embeddings...")
        
#         # Combine health_effects and potential_drug_interactor
#         food_texts = []
#         for _, row in df.iterrows():
#             health_effects = str(row.get('health_effects', '')) if pd.notna(row.get('health_effects')) else ''
#             drug_interactor = str(row.get('potential_drug_interactor', '')) if pd.notna(row.get('potential_drug_interactor')) else ''
#             compound_class = str(row.get('compound_class', '')) if pd.notna(row.get('compound_class')) else ''
#             metabolic = str(row.get('metabolic_pathways', '')) if pd.notna(row.get('metabolic_pathways')) else ''
            
#             combined_text = f"{health_effects} {drug_interactor} {compound_class} {metabolic}".strip()
#             food_texts.append(combined_text if combined_text else f"Food: {row.get('food_name', 'Unknown')}")
        
#         if not self.use_fallback_embeddings and self.food_model is not None:
#             try:
#                 print("Using Sentence-BERT embeddings for foods...")
#                 food_embeddings = self.food_model.encode(food_texts, batch_size=32, show_progress_bar=True)
#                 print(f"Sentence-BERT food embeddings shape: {food_embeddings.shape}")
#                 return food_embeddings
                
#             except Exception as e:
#                 print(f"Sentence-BERT embedding failed: {e}, falling back to TF-IDF...")
        
#         # TF-IDF fallback
#         print("Using TF-IDF embeddings for foods...")
#         vectorizer = TfidfVectorizer(
#             max_features=300, 
#             stop_words='english',
#             ngram_range=(1, 2),
#             min_df=1  # Changed from 2 to 1 to handle small datasets
#         )
#         food_embeddings = vectorizer.fit_transform(food_texts).toarray()
        
#         print(f"TF-IDF food embeddings shape: {food_embeddings.shape}")
#         return food_embeddings
    
#     def extract_numerical_features(self, df):
#         """
#         Extract and process key numerical features
#         """
#         print("Extracting numerical features...")
        
#         # Key numerical features from drugbank
#         drug_numerical = [
#             'molecular_weight', 'logp', 'h_bond_donors', 'h_bond_acceptors',
#             'polar_surface_area', 'rotatable_bonds', 'bioavailability',
#             'half-life_numeric', 'protein-binding_numeric', 
#             'volume-of-distribution_numeric', 'clearance_numeric'
#         ]
        
#         # Extract available numerical features
#         available_features = [col for col in drug_numerical if col in df.columns]
#         print(f"Available numerical features: {available_features}")
        
#         if available_features:
#             numerical_df = df[available_features].copy()
            
#             # Handle missing values
#             numerical_df = numerical_df.fillna(numerical_df.median())
            
#             # Create CYP interaction flags
#             cyp_flags = pd.DataFrame()
#             if 'interacts_with_cyp' in df.columns:
#                 cyp_flags['cyp_interaction'] = df['interacts_with_cyp'].fillna(False).astype(int)
            
#             # Combine all numerical features
#             if not cyp_flags.empty:
#                 print("Including CYP interaction flags")
#                 numerical_features = np.hstack([
#                     numerical_df.values,
#                     cyp_flags.values
#                 ])
#                 available_features.extend(['cyp_interaction'])
#             else:
#                 numerical_features = numerical_df.values
            
#             # Standardize features
#             numerical_features = self.scaler.fit_transform(numerical_features)
            
#         else:
#             print("No numerical features found, creating dummy features")
#             numerical_features = np.zeros((len(df), 1))
#             available_features = ['dummy_feature']
        
#         print(f"Numerical features shape: {numerical_features.shape}")
        
#         return numerical_features, available_features
    
#     def create_final_feature_set(self, drug_embeddings, food_embeddings, numerical_features):
#         """
#         Combine all features into final feature set
#         """
#         print("Creating final feature set...")
        
#         # Ensure all arrays have the same number of rows
#         n_samples = min(len(drug_embeddings), len(food_embeddings), len(numerical_features))
#         print(f"Aligning to {n_samples} samples")
        
#         drug_embeddings = drug_embeddings[:n_samples]
#         food_embeddings = food_embeddings[:n_samples]
#         numerical_features = numerical_features[:n_samples]
        
#         # Combine all features
#         final_features = np.hstack([
#             drug_embeddings,
#             food_embeddings,
#             numerical_features
#         ])
        
#         print(f"Final feature set shape: {final_features.shape}")
        
#         return final_features
    
#     def process_pipeline(self, mapped_drugs, drugbank_df, foodb_df):
#         """
#         Complete pipeline for feature fusion
#         """
#         print("Starting Drug-Food Interaction Feature Fusion Pipeline...")
#         print("="*60)
        
#         # Step 1: Load models
#         self.load_models()
        
#         # Step 2: Merge datasets
#         merged_df = self.merge_datasets(mapped_drugs, drugbank_df, foodb_df)
        
#         # Step 3: Create embeddings
#         drug_embeddings = self.create_drug_embeddings(merged_df)
#         food_embeddings = self.create_food_embeddings(merged_df)
        
#         # Step 4: Extract numerical features
#         numerical_features, feature_names = self.extract_numerical_features(merged_df)
        
#         # Step 5: Create final feature set
#         final_features = self.create_final_feature_set(
#             drug_embeddings, food_embeddings, numerical_features
#         )
        
#         # Create feature metadata
#         feature_info = {
#             'drug_embedding_dims': drug_embeddings.shape[1],
#             'food_embedding_dims': food_embeddings.shape[1],
#             'numerical_feature_names': feature_names,
#             'total_features': final_features.shape[1],
#             'using_advanced_models': not self.use_fallback_embeddings
#         }
        
#         print("="*60)
#         print("Pipeline completed successfully!")
#         print(f"Final dataset shape: {final_features.shape}")
#         print(f"Feature breakdown:")
#         print(f"  - Drug embeddings: {feature_info['drug_embedding_dims']} features")
#         print(f"  - Food embeddings: {feature_info['food_embedding_dims']} features")
#         print(f"  - Numerical features: {len(feature_info['numerical_feature_names'])} features")
#         print(f"  - Using advanced models: {feature_info['using_advanced_models']}")
        
#         return final_features, merged_df, feature_info


# def main():
#     """
#     Example usage of the DrugFoodFeatureFusion pipeline
#     """
#     # Initialize the pipeline
#     fusion_pipeline = DrugFoodFeatureFusion()
    
#     # Check if files exist
#     file_paths = {
#         'mapped_drugs': '/Users/sachidhoka/Desktop/mapped_interactions.csv',
#         'drugbank_df': '/Users/sachidhoka/Desktop/processed_drugbank_data.csv',
#         'foodb_df': '/Users/sachidhoka/Desktop/processed_foodb.csv'
#     }
    
#     # Check file existence
#     missing_files = []
#     for name, path in file_paths.items():
#         if not os.path.exists(path):
#             missing_files.append(path)
    
#     if missing_files:
#         print("Missing files:")
#         for file in missing_files:
#             print(f"  - {file}")
#         print("\nPlease ensure all files exist before running the pipeline.")
#         return
    
#     try:
#         # Load your datasets
#         print("Loading datasets...")
#         mapped_drugs = pd.read_csv(file_paths['mapped_drugs'])
#         drugbank_df = pd.read_csv(file_paths['drugbank_df'])
#         foodb_df = pd.read_csv(file_paths['foodb_df'])
        
#         print(f"Loaded datasets:")
#         print(f"  - Mapped drugs: {mapped_drugs.shape}")
#         print(f"  - DrugBank: {drugbank_df.shape}")
#         print(f"  - FooDB: {foodb_df.shape}")
        
#         # Process the pipeline
#         final_features, merged_df, feature_info = fusion_pipeline.process_pipeline(
#             mapped_drugs, drugbank_df, foodb_df
#         )
        
#         # Save results
#         print("\nSaving results...")
#         np.save('final_features.npy', final_features)
#         merged_df.to_csv('merged_drug_food_data.csv', index=False)
        
#         # Save feature info
#         import json
#         with open('feature_info.json', 'w') as f:
#             json.dump(feature_info, f, indent=2)
        
#         print("Results saved:")
#         print("  - final_features.npy")
#         print("  - merged_drug_food_data.csv")
#         print("  - feature_info.json")
#         print("\nFeature fusion pipeline completed successfully!")
        
#     except Exception as e:
#         print(f"Error in pipeline execution: {e}")
#         import traceback
#         traceback.print_exc()


# if __name__ == "__main__":
#     main()

# import pandas as pd
# import numpy as np
# import torch
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# import warnings
# import subprocess
# import sys
# import os
# warnings.filterwarnings('ignore')

# def check_pytorch_compatibility():
#     """Check PyTorch and transformers compatibility"""
#     try:
#         import torch
#         print(f"PyTorch version: {torch.__version__}")
        
#         # Check if MPS is available
#         if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#             print("✅ MPS backend available")
#         else:
#             print("❌ MPS backend not available")
        
#         # Check transformers version
#         try:
#             import transformers
#             print(f"Transformers version: {transformers.__version__}")
#         except ImportError:
#             print("❌ Transformers not installed")
#             return False
        
#         return True
        
#     except ImportError:
#         print("❌ PyTorch not properly installed")
#         return False

# def install_package(package):
#     """Install a package if it's not already installed"""
#     try:
#         __import__(package.replace('-', '_'))
#         return True
#     except ImportError:
#         try:
#             print(f"Installing {package}...")
#             subprocess.check_call([sys.executable, "-m", "pip", "install", package])
#             return True
#         except Exception as e:
#             print(f"Failed to install {package}: {e}")
#             return False

# def upgrade_transformers():
#     """Upgrade transformers to a compatible version"""
#     try:
#         print("Upgrading transformers to compatible version...")
#         subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "transformers==4.35.0"])
#         print("✅ Transformers upgraded successfully")
#         return True
#     except Exception as e:
#         print(f"❌ Failed to upgrade transformers: {e}")
#         return False

# class DrugFoodFeatureFusion:
#     def __init__(self):
#         """
#         Initialize the feature fusion pipeline for drug-food interactions
#         BioBERT/Sentence-BERT only - no TF-IDF fallback
#         """
#         self.drug_tokenizer = None
#         self.drug_model = None
#         self.food_model = None
#         self.scaler = StandardScaler()
#         self.pca = PCA(n_components=50)
#         self.device = None
        
#     def load_models(self):
#         """
#         Load BioBERT for drug embeddings and Sentence-BERT for food embeddings
#         Optimized for MacBook Air M3 with MPS GPU - Fixed for PyTorch compatibility
#         """
#         print("Loading advanced models for MacBook Air M3...")
        
#         # Check PyTorch version first
#         print(f"PyTorch version: {torch.__version__}")
        
#         # Check MPS availability first
#         if torch.backends.mps.is_available():
#             if torch.backends.mps.is_built():
#                 self.device = torch.device('mps')
#                 print("✅ Using Apple Silicon MPS GPU acceleration!")
#             else:
#                 print("❌ MPS not built, falling back to CPU")
#                 self.device = torch.device('cpu')
#         else:
#             print("❌ MPS not available, falling back to CPU")
#             self.device = torch.device('cpu')
        
#         # Install required packages
#         required_packages = ['transformers', 'sentence-transformers']
#         for package in required_packages:
#             if not install_package(package):
#                 raise RuntimeError(f"Failed to install required package: {package}")
        
#         try:
#             from transformers import AutoTokenizer, AutoModel
#             from sentence_transformers import SentenceTransformer
            
#             print("Loading BioBERT for drug embeddings...")
#             try:
#                 # Try BioBERT first with explicit device mapping to avoid version issues
#                 print("Attempting BioBERT with device compatibility fixes...")
#                 self.drug_tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
                
#                 # Load model with explicit device mapping to avoid get_default_device() issue
#                 self.drug_model = AutoModel.from_pretrained(
#                     'dmis-lab/biobert-base-cased-v1.1',
#                     device_map=None,  # Disable automatic device mapping
#                     torch_dtype=torch.float32  # Explicit dtype
#                 )
#                 print("✅ BioBERT loaded successfully")
#             except Exception as e:
#                 print(f"❌ BioBERT failed ({e})")
#                 # Try SciBERT as alternative
#                 try:
#                     print("Trying SciBERT with compatibility fixes...")
#                     self.drug_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
#                     self.drug_model = AutoModel.from_pretrained(
#                         'allenai/scibert_scivocab_uncased',
#                         device_map=None,  # Disable automatic device mapping
#                         torch_dtype=torch.float32  # Explicit dtype
#                     )
#                     print("✅ SciBERT loaded successfully")
#                 except Exception as e2:
#                     print(f"❌ SciBERT also failed ({e2})")
#                     # Try basic BERT as final fallback
#                     try:
#                         print("Trying basic BERT as final fallback...")
#                         self.drug_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
#                         self.drug_model = AutoModel.from_pretrained(
#                             'bert-base-uncased',
#                             device_map=None,
#                             torch_dtype=torch.float32
#                         )
#                         print("✅ Basic BERT loaded successfully")
#                     except Exception as e3:
#                         print(f"❌ All BERT models failed ({e3})")
#                         raise RuntimeError("Failed to load any transformer model")
            
#             print("Loading Sentence-BERT for food embeddings...")
#             # Use a smaller, faster model optimized for MPS
#             self.food_model = SentenceTransformer('all-MiniLM-L6-v2')
#             print("✅ Sentence-BERT loaded successfully")
            
#             # Set models to evaluation mode and move to device manually
#             self.drug_model.eval()
            
#             # Move to device with error handling
#             try:
#                 self.drug_model = self.drug_model.to(self.device)
#                 print(f"✅ Model moved to {self.device}")
#             except Exception as e:
#                 print(f"⚠️ Could not move model to {self.device}, using CPU: {e}")
#                 self.device = torch.device('cpu')
#                 self.drug_model = self.drug_model.to(self.device)
            
#             print(f"✅ All models loaded successfully on device: {self.device}")
            
#         except Exception as e:
#             print(f"❌ Critical error loading models: {e}")
#             import traceback
#             traceback.print_exc()
#             raise RuntimeError(f"Failed to load required models: {e}")
    
#     def debug_column_matching(self, mapped_drugs, foodb_df):
#         """
#         Debug function to find potential column matches between datasets
#         """
#         print("\n🔍 DEBUG: Analyzing column matching...")
#         print(f"Mapped drugs columns: {list(mapped_drugs.columns)}")
#         print(f"FooDB columns: {list(foodb_df.columns)}")
        
#         # Look for food-related columns in both datasets
#         mapped_food_cols = [col for col in mapped_drugs.columns if 'food' in col.lower() or 'name' in col.lower()]
#         foodb_food_cols = [col for col in foodb_df.columns if 'food' in col.lower() or 'name' in col.lower()]
        
#         print(f"Food-related columns in mapped_drugs: {mapped_food_cols}")
#         print(f"Food-related columns in FooDB: {foodb_food_cols}")
        
#         # Sample some values to see the format
#         if 'food_name' in mapped_drugs.columns:
#             print(f"Sample food_name values from mapped_drugs:")
#             print(mapped_drugs['food_name'].dropna().head(10).tolist())
        
#         for col in foodb_food_cols:
#             print(f"Sample {col} values from FooDB:")
#             print(foodb_df[col].dropna().head(10).tolist())
            
#         return mapped_food_cols, foodb_food_cols
        
#     def merge_datasets(self, mapped_drugs, drugbank_df, foodb_df):
#         """
#         Merge the three key datasets based on drugbank_id and food_name
#         Enhanced with better debugging and fuzzy matching
#         """
#         print("Merging datasets...")
#         print(f"Mapped drugs shape: {mapped_drugs.shape}")
#         print(f"DrugBank shape: {drugbank_df.shape}")
#         print(f"FooDB shape: {foodb_df.shape}")
        
#         # Debug column matching
#         mapped_food_cols, foodb_food_cols = self.debug_column_matching(mapped_drugs, foodb_df)
        
#         # Start with mapped_drugs as base
#         merged_df = mapped_drugs.copy()
        
#         # Merge with drugbank features
#         if 'drugbank_id' in mapped_drugs.columns and 'primary_id' in drugbank_df.columns:
#             merged_df = merged_df.merge(
#                 drugbank_df, 
#                 left_on='drugbank_id', 
#                 right_on='primary_id', 
#                 how='left'
#             )
#             print(f"After drugbank merge: {merged_df.shape}")
#             drugbank_matches = (~merged_df['primary_id'].isna()).sum()
#             print(f"DrugBank matches: {drugbank_matches}/{len(merged_df)}")
#         else:
#             print("Warning: Cannot merge with DrugBank - missing required columns")
        
#         # Enhanced food name matching for FooDB
#         final_df = merged_df.copy()
        
#         if 'food_name' in mapped_drugs.columns and len(foodb_food_cols) > 0:
#             print("🔄 Attempting enhanced FooDB merge...")
            
#             # Try different column combinations
#             merge_attempts = []
            
#             # Direct matches
#             for foodb_col in foodb_food_cols:
#                 if foodb_col in foodb_df.columns:
#                     merge_attempts.append(('food_name', foodb_col, 'direct'))
            
#             # Try with cleaned/normalized names
#             if 'name' in foodb_df.columns:
#                 merge_attempts.append(('food_name', 'name', 'cleaned'))
#             if 'public_id' in foodb_df.columns:
#                 merge_attempts.append(('food_name', 'public_id', 'cleaned'))
            
#             best_match_count = 0
#             best_merge = None
            
#             for left_col, right_col, method in merge_attempts:
#                 try:
#                     print(f"Trying merge: {left_col} -> {right_col} ({method})")
                    
#                     if method == 'direct':
#                         test_merge = merged_df.merge(
#                             foodb_df,
#                             left_on=left_col,
#                             right_on=right_col,
#                             how='left'
#                         )
#                     else:  # cleaned
#                         # Create cleaned versions for better matching
#                         merged_df_clean = merged_df.copy()
#                         foodb_df_clean = foodb_df.copy()
                        
#                         merged_df_clean['food_name_clean'] = merged_df_clean[left_col].str.lower().str.strip()
#                         foodb_df_clean['food_name_clean'] = foodb_df_clean[right_col].str.lower().str.strip()
                        
#                         test_merge = merged_df_clean.merge(
#                             foodb_df_clean,
#                             on='food_name_clean',
#                             how='left'
#                         )
                    
#                     # Count successful matches
#                     match_count = (~test_merge[right_col].isna()).sum()
#                     print(f"  Matches found: {match_count}/{len(test_merge)}")
                    
#                     if match_count > best_match_count:
#                         best_match_count = match_count
#                         best_merge = test_merge
#                         print(f"  ✅ New best match!")
                    
#                 except Exception as e:
#                     print(f"  ❌ Merge failed: {e}")
#                     continue
            
#             if best_merge is not None and best_match_count > 0:
#                 final_df = best_merge
#                 print(f"✅ FooDB merge successful: {best_match_count} matches")
#             else:
#                 print("❌ No successful FooDB merges found")
                
#                 # Fallback: Add empty FooDB columns for consistency
#                 print("Adding empty FooDB columns for consistency...")
#                 key_foodb_cols = ['cyp_interactor', 'grapefruit_like', 'interacts_with_cyp3a4', 
#                                 'interacts_with_maoi', 'interacts_with_warfarin', 
#                                 'pathway_activation', 'potential_drug_interactor']
                
#                 for col in key_foodb_cols:
#                     if col not in final_df.columns:
#                         final_df[col] = 0
#         else:
#             print("❌ Cannot attempt FooDB merge - missing food_name or no food columns in FooDB")
        
#         print(f"\nFinal merged dataset shape: {final_df.shape}")
#         return final_df
    
#     def clear_mps_cache_safe(self):
#         """
#         Safely clear MPS cache - handles different PyTorch versions
#         """
#         if self.device.type == 'mps':
#             try:
#                 # Try different cache clearing methods based on PyTorch version
#                 if hasattr(torch.backends.mps, 'empty_cache'):
#                     torch.backends.mps.empty_cache()
#                 elif hasattr(torch.mps, 'empty_cache'):
#                     torch.mps.empty_cache()
#                 else:
#                     # Force garbage collection as fallback
#                     import gc
#                     gc.collect()
#             except Exception as e:
#                 print(f"⚠️ MPS cache clear failed (non-critical): {e}")
    
#     def create_drug_embeddings(self, df):
#         """
#         Create drug embeddings using BioBERT with MPS optimization
#         Fixed MPS cache handling
#         """
#         print("Creating drug embeddings with BioBERT...")
        
#         if self.drug_model is None or self.drug_tokenizer is None:
#             raise RuntimeError("Drug models not loaded properly")
        
#         # Combine mechanism-of-action and pharmacodynamics
#         drug_texts = []
#         for _, row in df.iterrows():
#             moa = str(row.get('mechanism-of-action', '')) if pd.notna(row.get('mechanism-of-action')) else ''
#             pharma = str(row.get('pharmacodynamics', '')) if pd.notna(row.get('pharmacodynamics')) else ''
#             desc = str(row.get('description', '')) if pd.notna(row.get('description')) else ''
#             combined_text = f"{moa} {pharma} {desc}".strip()
            
#             # Ensure we have some text
#             if not combined_text:
#                 combined_text = f"Drug: {row.get('drug_name', 'Unknown')}"
            
#             drug_texts.append(combined_text)
        
#         print(f"Processing {len(drug_texts)} drug texts...")
        
#         # Process in batches optimized for MPS
#         embeddings = []
#         batch_size = 16 if self.device.type == 'mps' else 32  # Smaller batch for MPS
        
#         for i in range(0, len(drug_texts), batch_size):
#             batch_texts = drug_texts[i:i+batch_size]
            
#             # Tokenize with appropriate max length for MPS
#             inputs = self.drug_tokenizer(
#                 batch_texts, 
#                 padding=True, 
#                 truncation=True, 
#                 max_length=256,  # Reduced for MPS efficiency
#                 return_tensors='pt'
#             ).to(self.device)
            
#             with torch.no_grad():
#                 try:
#                     outputs = self.drug_model(**inputs)
#                     # Use [CLS] token embedding or mean pooling
#                     batch_embeddings = outputs.last_hidden_state.mean(dim=1)
#                     embeddings.append(batch_embeddings.cpu().numpy())
#                 except Exception as e:
#                     print(f"Error processing batch {i//batch_size + 1}: {e}")
#                     # Create zero embeddings for failed batch
#                     batch_embeddings = torch.zeros(len(batch_texts), 768)
#                     embeddings.append(batch_embeddings.numpy())
            
#             # Clear MPS cache periodically using safe method
#             if i % (batch_size * 10) == 0:
#                 self.clear_mps_cache_safe()
            
#             if (i // batch_size + 1) % 10 == 0:
#                 print(f"Processed {i // batch_size + 1}/{(len(drug_texts) + batch_size - 1) // batch_size} batches")
        
#         drug_embeddings = np.vstack(embeddings)
#         print(f"✅ BioBERT drug embeddings shape: {drug_embeddings.shape}")
        
#         # Final MPS cache clear
#         self.clear_mps_cache_safe()
        
#         return drug_embeddings
    
#     def create_food_embeddings(self, df):
#         """
#         Create food embeddings using Sentence-BERT
#         """
#         print("Creating food embeddings with Sentence-BERT...")
        
#         if self.food_model is None:
#             raise RuntimeError("Food model not loaded properly")
        
#         # Combine available food-related text features
#         food_texts = []
#         for _, row in df.iterrows():
#             food_features = []
            
#             # Add interaction description if available
#             if 'interaction_description' in df.columns:
#                 interaction_desc = str(row.get('interaction_description', '')) if pd.notna(row.get('interaction_description')) else ''
#                 if interaction_desc:
#                     food_features.append(interaction_desc)
            
#             # Add health effects if available
#             if 'health_effects' in df.columns:
#                 health_effects = str(row.get('health_effects', '')) if pd.notna(row.get('health_effects')) else ''
#                 if health_effects:
#                     food_features.append(health_effects)
            
#             # Add potential drug interactor info
#             if 'potential_drug_interactor' in df.columns:
#                 drug_interactor = str(row.get('potential_drug_interactor', '')) if pd.notna(row.get('potential_drug_interactor')) else ''
#                 if drug_interactor:
#                     food_features.append(drug_interactor)
            
#             # Combine all available text
#             combined_text = ' '.join(food_features).strip()
            
#             # Ensure we have some text
#             if not combined_text:
#                 combined_text = f"Food: {row.get('food_name', 'Unknown')}"
            
#             food_texts.append(combined_text)
        
#         print(f"Processing {len(food_texts)} food texts...")
        
#         # Create embeddings with Sentence-BERT
#         food_embeddings = self.food_model.encode(
#             food_texts, 
#             batch_size=32,
#             show_progress_bar=True,
#             convert_to_numpy=True
#         )
        
#         print(f"✅ Sentence-BERT food embeddings shape: {food_embeddings.shape}")
#         return food_embeddings
    
#     def extract_numerical_features(self, df):
#         """
#         Extract and process key numerical features
#         """
#         print("Extracting numerical features...")
        
#         # Key numerical features from drugbank
#         drug_numerical = [
#             'molecular_weight', 'logp', 'h_bond_donors', 'h_bond_acceptors',
#             'polar_surface_area', 'rotatable_bonds', 'bioavailability',
#             'half-life_numeric', 'protein-binding_numeric', 
#             'volume-of-distribution_numeric', 'clearance_numeric'
#         ]
        
#         # Extract available numerical features
#         available_features = [col for col in drug_numerical if col in df.columns]
#         print(f"Available numerical features: {available_features}")
        
#         if not available_features:
#             print("❌ No standard numerical features found, using basic features...")
#             # Create basic numerical features from available data
#             basic_features = []
#             if 'drug_name' in df.columns:
#                 basic_features.append('drug_name_length')
#                 df['drug_name_length'] = df['drug_name'].str.len().fillna(0)
#             if 'food_name' in df.columns:
#                 basic_features.append('food_name_length')
#                 df['food_name_length'] = df['food_name'].str.len().fillna(0)
            
#             available_features = basic_features
#             if not available_features:
#                 raise RuntimeError("No numerical features could be created")
        
#         numerical_df = df[available_features].copy()
        
#         # Handle missing values with median imputation
#         numerical_df = numerical_df.fillna(numerical_df.median())
        
#         # Create interaction flags from available FooDB features
#         interaction_flags = pd.DataFrame()
#         foodb_flag_cols = [
#             'cyp_interactor', 'grapefruit_like', 'interacts_with_cyp3a4', 
#             'interacts_with_maoi', 'interacts_with_warfarin', 
#             'pathway_activation', 'potential_drug_interactor'
#         ]
        
#         for col in foodb_flag_cols:
#             if col in df.columns:
#                 interaction_flags[col] = df[col].fillna(0).astype(int)
        
#         # Add severity encoding if available
#         if 'severity_encoded' in df.columns:
#             interaction_flags['severity_encoded'] = df['severity_encoded'].fillna(0)
        
#         # Combine all numerical features
#         if not interaction_flags.empty:
#             print(f"Including interaction flags: {list(interaction_flags.columns)}")
#             numerical_features = np.hstack([
#                 numerical_df.values,
#                 interaction_flags.values
#             ])
#             available_features.extend(list(interaction_flags.columns))
#         else:
#             numerical_features = numerical_df.values
        
#         # Standardize features
#         numerical_features = self.scaler.fit_transform(numerical_features)
        
#         print(f"✅ Numerical features shape: {numerical_features.shape}")
#         return numerical_features, available_features
    
#     def create_final_feature_set(self, drug_embeddings, food_embeddings, numerical_features):
#         """
#         Combine all features into final feature set with alignment
#         """
#         print("Creating final feature set...")
        
#         # Ensure all arrays have the same number of rows
#         n_samples = min(len(drug_embeddings), len(food_embeddings), len(numerical_features))
#         print(f"Aligning to {n_samples} samples")
        
#         drug_embeddings = drug_embeddings[:n_samples]
#         food_embeddings = food_embeddings[:n_samples]
#         numerical_features = numerical_features[:n_samples]
        
#         # Combine all features
#         final_features = np.hstack([
#             drug_embeddings,
#             food_embeddings,
#             numerical_features
#         ])
        
#         print(f"✅ Final feature set shape: {final_features.shape}")
#         return final_features
    
#     def process_pipeline(self, mapped_drugs, drugbank_df, foodb_df):
#         """
#         Complete pipeline for feature fusion - BioBERT only
#         """
#         print("🚀 Starting Drug-Food Interaction Feature Fusion Pipeline...")
#         print("📱 MacBook Air M3 Optimized - BioBERT/Sentence-BERT Only")
#         print("="*70)
        
#         # Step 1: Load models
#         print("\n📦 Step 1: Loading Models...")
#         self.load_models()
        
#         # Step 2: Merge datasets
#         print("\n🔗 Step 2: Merging Datasets...")
#         merged_df = self.merge_datasets(mapped_drugs, drugbank_df, foodb_df)
        
#         # Step 3: Create embeddings
#         print("\n🧠 Step 3: Creating Drug Embeddings...")
#         drug_embeddings = self.create_drug_embeddings(merged_df)
        
#         print("\n🍎 Step 4: Creating Food Embeddings...")
#         food_embeddings = self.create_food_embeddings(merged_df)
        
#         # Step 5: Extract numerical features
#         print("\n🔢 Step 5: Extracting Numerical Features...")
#         numerical_features, feature_names = self.extract_numerical_features(merged_df)
        
#         # Step 6: Create final feature set
#         print("\n🎯 Step 6: Creating Final Feature Set...")
#         final_features = self.create_final_feature_set(
#             drug_embeddings, food_embeddings, numerical_features
#         )
        
#         # Create feature metadata
#         feature_info = {
#             'drug_embedding_dims': drug_embeddings.shape[1],
#             'food_embedding_dims': food_embeddings.shape[1],
#             'numerical_feature_names': feature_names,
#             'total_features': final_features.shape[1],
#             'device_used': str(self.device),
#             'mps_enabled': self.device.type == 'mps',
#             'model_type': 'BioBERT + Sentence-BERT'
#         }
        
#         print("\n" + "="*70)
#         print("✅ Pipeline completed successfully!")
#         print(f"📊 Final dataset shape: {final_features.shape}")
#         print(f"🔧 Device used: {self.device}")
#         print(f"⚡ MPS acceleration: {'✅' if self.device.type == 'mps' else '❌'}")
#         print(f"📈 Feature breakdown:")
#         print(f"   • Drug embeddings: {feature_info['drug_embedding_dims']} features")
#         print(f"   • Food embeddings: {feature_info['food_embedding_dims']} features")
#         print(f"   • Numerical features: {len(feature_info['numerical_feature_names'])} features")
        
#         return final_features, merged_df, feature_info


# def main():
#     """
#     Example usage optimized for MacBook Air M3
#     """
#     print("🍎 Drug-Food Feature Fusion - MacBook Air M3 Edition")
#     print("="*50)
    
#     # Check PyTorch compatibility first
#     print("🔍 Checking PyTorch compatibility...")
#     if not check_pytorch_compatibility():
#         print("❌ PyTorch compatibility issues detected")
#         print("💡 Try running: pip install --upgrade torch transformers")
#         return
    
#     # Check if we need to upgrade transformers
#     try:
#         import transformers
#         from packaging import version
#         if version.parse(transformers.__version__) < version.parse("4.30.0"):
#             print("⚠️ Transformers version too old, upgrading...")
#             if not upgrade_transformers():
#                 print("❌ Failed to upgrade transformers")
#                 return
#     except Exception as e:
#         print(f"⚠️ Could not check transformers version: {e}")
    
#     # Initialize the pipeline
#     fusion_pipeline = DrugFoodFeatureFusion()
    
#     # File paths
#     file_paths = {
#         'mapped_drugs': '/Users/sachidhoka/Desktop/mapped_interactions.csv',
#         'drugbank_df': '/Users/sachidhoka/Desktop/processed_drugbank_data.csv',
#         'foodb_df': '/Users/sachidhoka/Desktop/processed_foodb.csv'
#     }
    
#     # Check file existence
#     missing_files = []
#     for name, path in file_paths.items():
#         if not os.path.exists(path):
#             missing_files.append(path)
    
#     if missing_files:
#         print("❌ Missing files:")
#         for file in missing_files:
#             print(f"   • {file}")
#         print("\n🔧 Please ensure all files exist before running the pipeline.")
#         return
    
#     try:
#         # Load datasets
#         print("📂 Loading datasets...")
#         mapped_drugs = pd.read_csv(file_paths['mapped_drugs'])
#         drugbank_df = pd.read_csv(file_paths['drugbank_df'])
#         foodb_df = pd.read_csv(file_paths['foodb_df'])
        
#         print(f"✅ Loaded datasets:")
#         print(f"   • Mapped drugs: {mapped_drugs.shape}")
#         print(f"   • DrugBank: {drugbank_df.shape}")
#         print(f"   • FooDB: {foodb_df.shape}")
        
#         # Process the pipeline
#         final_features, merged_df, feature_info = fusion_pipeline.process_pipeline(
#             mapped_drugs, drugbank_df, foodb_df
#         )
        
#         # Save results
#         print("\n💾 Saving results...")
#         np.save('final_features.npy', final_features)
#         merged_df.to_csv('merged_drug_food_data.csv', index=False)
        
#         # Save feature info
#         import json
#         with open('feature_info.json', 'w') as f:
#             json.dump(feature_info, f, indent=2)
        
#         print("✅ Results saved:")
#         print("   • final_features.npy")
#         print("   • merged_drug_food_data.csv")
#         print("   • feature_info.json")
#         print("\n🎉 Feature fusion pipeline completed successfully!")
        
#     except Exception as e:
#         print(f"❌ Error in pipeline execution: {e}")
#         import traceback
#         traceback.print_exc()
        
#         # Provide troubleshooting suggestions
#         print("\n🔧 Troubleshooting suggestions:")
#         print("1. pip install --upgrade torch torchvision torchaudio")
#         print("2. pip install --upgrade transformers==4.35.0")
#         print("3. pip install --upgrade sentence-transformers")
#         print("4. Restart your Python kernel/environment")


# if __name__ == "__main__":
#     main()

# import pandas as pd
# import numpy as np
# import torch
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from fuzzywuzzy import fuzz, process
# import warnings
# import subprocess
# import sys
# import os
# warnings.filterwarnings('ignore')

# def check_pytorch_compatibility():
#     """Check PyTorch and transformers compatibility"""
#     try:
#         import torch
#         print(f"PyTorch version: {torch.__version__}")
        
#         # Check if MPS is available
#         if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#             print("✅ MPS backend available")
#         else:
#             print("❌ MPS backend not available")
        
#         # Check transformers version
#         try:
#             import transformers
#             print(f"Transformers version: {transformers.__version__}")
#         except ImportError:
#             print("❌ Transformers not installed")
#             return False
        
#         return True
        
#     except ImportError:
#         print("❌ PyTorch not properly installed")
#         return False

# def install_package(package):
#     """Install a package if it's not already installed"""
#     try:
#         __import__(package.replace('-', '_'))
#         return True
#     except ImportError:
#         try:
#             print(f"Installing {package}...")
#             subprocess.check_call([sys.executable, "-m", "pip", "install", package])
#             return True
#         except Exception as e:
#             print(f"Failed to install {package}: {e}")
#             return False

# class DrugFoodFeatureFusion:
#     def __init__(self):
#         """
#         Initialize the feature fusion pipeline for drug-food interactions
#         BioBERT/Sentence-BERT only - no TF-IDF fallback
#         """
#         self.drug_tokenizer = None
#         self.drug_model = None
#         self.food_model = None
#         self.scaler = StandardScaler()
#         self.pca = PCA(n_components=50)
#         self.device = None
        
#     def load_models(self):
#         """
#         Load BioBERT for drug embeddings and Sentence-BERT for food embeddings
#         Optimized for MacBook Air M3 with MPS GPU - Fixed for PyTorch compatibility
#         """
#         print("Loading advanced models for MacBook Air M3...")
        
#         # Check PyTorch version first
#         print(f"PyTorch version: {torch.__version__}")
        
#         # Check MPS availability first
#         if torch.backends.mps.is_available():
#             if torch.backends.mps.is_built():
#                 self.device = torch.device('mps')
#                 print("✅ Using Apple Silicon MPS GPU acceleration!")
#             else:
#                 print("❌ MPS not built, falling back to CPU")
#                 self.device = torch.device('cpu')
#         else:
#             print("❌ MPS not available, falling back to CPU")
#             self.device = torch.device('cpu')
        
#         # Install required packages
#         required_packages = ['transformers', 'sentence-transformers', 'fuzzywuzzy', 'python-levenshtein']
#         for package in required_packages:
#             if not install_package(package):
#                 print(f"⚠️ Failed to install {package}, continuing...")
        
#         try:
#             from transformers import AutoTokenizer, AutoModel
#             from sentence_transformers import SentenceTransformer
            
#             print("Loading BioBERT for drug embeddings...")
#             try:
#                 # Try BioBERT first with explicit device mapping to avoid version issues
#                 print("Attempting BioBERT with device compatibility fixes...")
#                 self.drug_tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
                
#                 # Load model with explicit device mapping to avoid get_default_device() issue
#                 self.drug_model = AutoModel.from_pretrained(
#                     'dmis-lab/biobert-base-cased-v1.1',
#                     device_map=None,  # Disable automatic device mapping
#                     torch_dtype=torch.float32  # Explicit dtype
#                 )
#                 print("✅ BioBERT loaded successfully")
#             except Exception as e:
#                 print(f"❌ BioBERT failed ({e})")
#                 # Try SciBERT as alternative
#                 try:
#                     print("Trying SciBERT with compatibility fixes...")
#                     self.drug_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
#                     self.drug_model = AutoModel.from_pretrained(
#                         'allenai/scibert_scivocab_uncased',
#                         device_map=None,  # Disable automatic device mapping
#                         torch_dtype=torch.float32  # Explicit dtype
#                     )
#                     print("✅ SciBERT loaded successfully")
#                 except Exception as e2:
#                     print(f"❌ SciBERT also failed ({e2})")
#                     # Try basic BERT as final fallback
#                     try:
#                         print("Trying basic BERT as final fallback...")
#                         self.drug_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
#                         self.drug_model = AutoModel.from_pretrained(
#                             'bert-base-uncased',
#                             device_map=None,
#                             torch_dtype=torch.float32
#                         )
#                         print("✅ Basic BERT loaded successfully")
#                     except Exception as e3:
#                         print(f"❌ All BERT models failed ({e3})")
#                         raise RuntimeError("Failed to load any transformer model")
            
#             print("Loading Sentence-BERT for food embeddings...")
#             # Use a smaller, faster model optimized for MPS
#             self.food_model = SentenceTransformer('all-MiniLM-L6-v2')
#             print("✅ Sentence-BERT loaded successfully")
            
#             # Set models to evaluation mode and move to device manually
#             self.drug_model.eval()
            
#             # Move to device with error handling
#             try:
#                 self.drug_model = self.drug_model.to(self.device)
#                 print(f"✅ Model moved to {self.device}")
#             except Exception as e:
#                 print(f"⚠️ Could not move model to {self.device}, using CPU: {e}")
#                 self.device = torch.device('cpu')
#                 self.drug_model = self.drug_model.to(self.device)
            
#             print(f"✅ All models loaded successfully on device: {self.device}")
            
#         except Exception as e:
#             print(f"❌ Critical error loading models: {e}")
#             import traceback
#             traceback.print_exc()
#             raise RuntimeError(f"Failed to load required models: {e}")
    
#     def create_food_name_mapping(self, foodb_df):
#         """
#         Create a mapping dictionary from food names to food_ids in FooDB
#         This handles the string to integer column mismatch
#         """
#         print("🔄 Creating food name mapping from FooDB...")
        
#         # Check if we have a name column in FooDB
#         name_columns = [col for col in foodb_df.columns if 'name' in col.lower()]
#         print(f"Available name columns in FooDB: {name_columns}")
        
#         food_mapping = {}
        
#         # Try different approaches to create the mapping
#         if 'name' in foodb_df.columns:
#             # Direct name mapping
#             for idx, row in foodb_df.iterrows():
#                 if pd.notna(row['name']) and pd.notna(row['food_id']):
#                     food_name = str(row['name']).lower().strip()
#                     food_mapping[food_name] = row['food_id']
        
#         elif 'public_id' in foodb_df.columns:
#             # Use public_id as name
#             for idx, row in foodb_df.iterrows():
#                 if pd.notna(row['public_id']) and pd.notna(row['food_id']):
#                     food_name = str(row['public_id']).lower().strip()
#                     food_mapping[food_name] = row['food_id']
        
#         else:
#             # Create synthetic mapping based on food_id
#             print("⚠️ No name column found, creating synthetic mapping...")
#             for idx, row in foodb_df.iterrows():
#                 if pd.notna(row['food_id']):
#                     food_name = f"food_{row['food_id']}"
#                     food_mapping[food_name] = row['food_id']
        
#         print(f"✅ Created mapping for {len(food_mapping)} foods")
#         if len(food_mapping) > 0:
#             # Show sample mappings
#             sample_items = list(food_mapping.items())[:5]
#             print(f"Sample mappings: {sample_items}")
        
#         return food_mapping
    
#     def fuzzy_match_foods(self, mapped_drugs, food_mapping, threshold=80):
#         """
#         Use fuzzy matching to link food names between datasets
#         """
#         print(f"🔍 Performing fuzzy matching with threshold {threshold}...")
        
#         if not food_mapping:
#             print("❌ No food mapping available for fuzzy matching")
#             return {}
        
#         # Get unique food names from mapped_drugs
#         unique_foods = mapped_drugs['food_name'].dropna().unique()
#         print(f"Matching {len(unique_foods)} unique food names...")
        
#         fuzzy_matches = {}
#         food_keys = list(food_mapping.keys())
        
#         for food_name in unique_foods:
#             food_name_clean = str(food_name).lower().strip()
            
#             # First try exact match
#             if food_name_clean in food_mapping:
#                 fuzzy_matches[food_name] = food_mapping[food_name_clean]
#                 continue
            
#             # Then try fuzzy matching
#             try:
#                 best_match = process.extractOne(
#                     food_name_clean, 
#                     food_keys, 
#                     scorer=fuzz.ratio
#                 )
                
#                 if best_match and best_match[1] >= threshold:
#                     matched_key = best_match[0]
#                     fuzzy_matches[food_name] = food_mapping[matched_key]
                    
#             except Exception as e:
#                 print(f"⚠️ Fuzzy matching failed for '{food_name}': {e}")
#                 continue
        
#         print(f"✅ Successfully matched {len(fuzzy_matches)} food names")
#         return fuzzy_matches
    
#     def debug_column_matching(self, mapped_drugs, foodb_df):
#         """
#         Debug function to find potential column matches between datasets
#         """
#         print("\n🔍 DEBUG: Analyzing column matching...")
#         print(f"Mapped drugs columns: {list(mapped_drugs.columns)}")
#         print(f"FooDB columns: {list(foodb_df.columns)}")
        
#         # Look for food-related columns in both datasets
#         mapped_food_cols = [col for col in mapped_drugs.columns if 'food' in col.lower() or 'name' in col.lower()]
#         foodb_food_cols = [col for col in foodb_df.columns if 'food' in col.lower() or 'name' in col.lower()]
        
#         print(f"Food-related columns in mapped_drugs: {mapped_food_cols}")
#         print(f"Food-related columns in FooDB: {foodb_food_cols}")
        
#         # Sample some values to see the format
#         if 'food_name' in mapped_drugs.columns:
#             print(f"Sample food_name values from mapped_drugs:")
#             print(mapped_drugs['food_name'].dropna().head(10).tolist())
        
#         for col in foodb_food_cols[:3]:  # Limit to first 3 columns
#             if col in foodb_df.columns:
#                 print(f"Sample {col} values from FooDB:")
#                 print(foodb_df[col].dropna().head(10).tolist())
                
#         return mapped_food_cols, foodb_food_cols
        
#     def merge_datasets(self, mapped_drugs, drugbank_df, foodb_df):
#         """
#         Merge the three key datasets based on drugbank_id and food_name
#         Enhanced with better debugging and fuzzy matching
#         """
#         print("Merging datasets...")
#         print(f"Mapped drugs shape: {mapped_drugs.shape}")
#         print(f"DrugBank shape: {drugbank_df.shape}")
#         print(f"FooDB shape: {foodb_df.shape}")
        
#         # Debug column matching
#         mapped_food_cols, foodb_food_cols = self.debug_column_matching(mapped_drugs, foodb_df)
        
#         # Start with mapped_drugs as base
#         merged_df = mapped_drugs.copy()
        
#         # Merge with drugbank features
#         if 'drugbank_id' in mapped_drugs.columns and 'primary_id' in drugbank_df.columns:
#             merged_df = merged_df.merge(
#                 drugbank_df, 
#                 left_on='drugbank_id', 
#                 right_on='primary_id', 
#                 how='left'
#             )
#             print(f"After drugbank merge: {merged_df.shape}")
#             drugbank_matches = (~merged_df['primary_id'].isna()).sum()
#             print(f"DrugBank matches: {drugbank_matches}/{len(merged_df)}")
#         else:
#             print("Warning: Cannot merge with DrugBank - missing required columns")
        
#         # Enhanced food name matching for FooDB
#         print("🔄 Attempting enhanced FooDB merge...")
        
#         # Step 1: Create food name mapping
#         food_mapping = self.create_food_name_mapping(foodb_df)
        
#         if food_mapping and 'food_name' in mapped_drugs.columns:
#             # Step 2: Apply fuzzy matching
#             fuzzy_matches = self.fuzzy_match_foods(merged_df, food_mapping, threshold=70)
            
#             if fuzzy_matches:
#                 # Step 3: Create the merge
#                 print("🔗 Creating FooDB merge using fuzzy matches...")
                
#                 # Add matched food_ids to merged_df
#                 merged_df['matched_food_id'] = merged_df['food_name'].map(fuzzy_matches)
                
#                 # Merge with FooDB using the matched food_ids
#                 foodb_merge = merged_df.merge(
#                     foodb_df,
#                     left_on='matched_food_id',
#                     right_on='food_id',
#                     how='left'
#                 )
                
#                 # Count successful matches
#                 foodb_matches = (~foodb_merge['food_id'].isna()).sum()
#                 print(f"✅ FooDB merge successful: {foodb_matches}/{len(merged_df)} matches")
                
#                 if foodb_matches > 0:
#                     merged_df = foodb_merge
#                 else:
#                     print("⚠️ No actual FooDB matches found, keeping original dataset")
#             else:
#                 print("❌ No fuzzy matches found")
#         else:
#             print("❌ Cannot create food mapping - missing required columns")
        
#         # Add empty FooDB columns for consistency if merge failed
#         key_foodb_cols = [
#             'cyp_interactor', 'grapefruit_like', 'interacts_with_cyp3a4', 
#             'interacts_with_maoi', 'interacts_with_warfarin', 
#             'pathway_activation', 'potential_drug_interactor'
#         ]
        
#         for col in key_foodb_cols:
#             if col not in merged_df.columns:
#                 merged_df[col] = 0
                
#         print(f"\nFinal merged dataset shape: {merged_df.shape}")
#         return merged_df
    
#     def clear_mps_cache_safe(self):
#         """
#         Safely clear MPS cache - handles different PyTorch versions
#         """
#         if self.device and self.device.type == 'mps':
#             try:
#                 # Try different cache clearing methods based on PyTorch version
#                 if hasattr(torch.backends.mps, 'empty_cache'):
#                     torch.backends.mps.empty_cache()
#                 elif hasattr(torch.mps, 'empty_cache'):
#                     torch.mps.empty_cache()
#                 else:
#                     # Force garbage collection as fallback
#                     import gc
#                     gc.collect()
#             except Exception as e:
#                 print(f"⚠️ MPS cache clear failed (non-critical): {e}")
    
#     def create_drug_embeddings(self, df):
#         """
#         Create drug embeddings using BioBERT with MPS optimization
#         Fixed MPS cache handling
#         """
#         print("Creating drug embeddings with BioBERT...")
        
#         if self.drug_model is None or self.drug_tokenizer is None:
#             raise RuntimeError("Drug models not loaded properly")
        
#         # Combine mechanism-of-action and pharmacodynamics
#         drug_texts = []
#         for _, row in df.iterrows():
#             moa = str(row.get('mechanism-of-action', '')) if pd.notna(row.get('mechanism-of-action')) else ''
#             pharma = str(row.get('pharmacodynamics', '')) if pd.notna(row.get('pharmacodynamics')) else ''
#             desc = str(row.get('description', '')) if pd.notna(row.get('description')) else ''
#             combined_text = f"{moa} {pharma} {desc}".strip()
            
#             # Ensure we have some text
#             if not combined_text:
#                 combined_text = f"Drug: {row.get('drug_name', 'Unknown')}"
            
#             drug_texts.append(combined_text)
        
#         print(f"Processing {len(drug_texts)} drug texts...")
        
#         # Process in batches optimized for MPS
#         embeddings = []
#         batch_size = 16 if self.device.type == 'mps' else 32  # Smaller batch for MPS
        
#         for i in range(0, len(drug_texts), batch_size):
#             batch_texts = drug_texts[i:i+batch_size]
            
#             # Tokenize with appropriate max length for MPS
#             inputs = self.drug_tokenizer(
#                 batch_texts, 
#                 padding=True, 
#                 truncation=True, 
#                 max_length=256,  # Reduced for MPS efficiency
#                 return_tensors='pt'
#             ).to(self.device)
            
#             with torch.no_grad():
#                 try:
#                     outputs = self.drug_model(**inputs)
#                     # Use [CLS] token embedding or mean pooling
#                     batch_embeddings = outputs.last_hidden_state.mean(dim=1)
#                     embeddings.append(batch_embeddings.cpu().numpy())
#                 except Exception as e:
#                     print(f"Error processing batch {i//batch_size + 1}: {e}")
#                     # Create zero embeddings for failed batch
#                     batch_embeddings = torch.zeros(len(batch_texts), 768)
#                     embeddings.append(batch_embeddings.numpy())
            
#             # Clear MPS cache periodically using safe method
#             if i % (batch_size * 10) == 0:
#                 self.clear_mps_cache_safe()
            
#             if (i // batch_size + 1) % 10 == 0:
#                 print(f"Processed {i // batch_size + 1}/{(len(drug_texts) + batch_size - 1) // batch_size} batches")
        
#         drug_embeddings = np.vstack(embeddings)
#         print(f"✅ BioBERT drug embeddings shape: {drug_embeddings.shape}")
        
#         # Final MPS cache clear
#         self.clear_mps_cache_safe()
        
#         return drug_embeddings
    
#     def create_food_embeddings(self, df):
#         """
#         Create food embeddings using Sentence-BERT on health_effects + potential_drug_interactor
#         """
#         print("Creating food embeddings with Sentence-BERT...")
        
#         if self.food_model is None:
#             raise RuntimeError("Food model not loaded properly")
        
#         # Combine available food-related text features
#         food_texts = []
#         for _, row in df.iterrows():
#             food_features = []
            
#             # Add health effects if available
#             if 'health_effects' in df.columns:
#                 health_effects = str(row.get('health_effects', '')) if pd.notna(row.get('health_effects')) else ''
#                 if health_effects and health_effects != 'nan':
#                     food_features.append(health_effects)
            
#             # Add potential drug interactor info
#             if 'potential_drug_interactor' in df.columns:
#                 drug_interactor = str(row.get('potential_drug_interactor', '')) if pd.notna(row.get('potential_drug_interactor')) else ''
#                 if drug_interactor and drug_interactor != 'nan':
#                     food_features.append(drug_interactor)
            
#             # Add interaction description if available
#             if 'interaction_description' in df.columns:
#                 interaction_desc = str(row.get('interaction_description', '')) if pd.notna(row.get('interaction_description')) else ''
#                 if interaction_desc and interaction_desc != 'nan':
#                     food_features.append(interaction_desc)
            
#             # Combine all available text
#             combined_text = ' '.join(food_features).strip()
            
#             # Ensure we have some text
#             if not combined_text:
#                 combined_text = f"Food: {row.get('food_name', 'Unknown')}"
            
#             food_texts.append(combined_text)
        
#         print(f"Processing {len(food_texts)} food texts...")
        
#         # Create embeddings with Sentence-BERT
#         food_embeddings = self.food_model.encode(
#             food_texts, 
#             batch_size=32,
#             show_progress_bar=True,
#             convert_to_numpy=True
#         )
        
#         print(f"✅ Sentence-BERT food embeddings shape: {food_embeddings.shape}")
#         return food_embeddings
    
#     def extract_numerical_features(self, df):
#         """
#         Extract key numerical features including logP, polar surface area, CYP interaction flags
#         """
#         print("Extracting numerical features...")
        
#         # Key numerical features from drugbank
#         drug_numerical = [
#             'molecular_weight', 'logp', 'h_bond_donors', 'h_bond_acceptors',
#             'polar_surface_area', 'rotatable_bonds', 'bioavailability',
#             'half-life_numeric', 'protein-binding_numeric', 
#             'volume-of-distribution_numeric', 'clearance_numeric'
#         ]
        
#         # Extract available numerical features
#         available_features = [col for col in drug_numerical if col in df.columns]
#         print(f"Available drug features: {available_features}")
        
#         # Create basic numerical features if standard ones aren't available
#         if not available_features:
#             print("❌ No standard numerical features found, creating basic features...")
#             basic_features = []
#             if 'drug_name' in df.columns:
#                 basic_features.append('drug_name_length')
#                 df['drug_name_length'] = df['drug_name'].str.len().fillna(0)
#             if 'food_name' in df.columns:
#                 basic_features.append('food_name_length')
#                 df['food_name_length'] = df['food_name'].str.len().fillna(0)
            
#             available_features = basic_features
        
#         # Extract numerical data
#         if available_features:
#             numerical_df = df[available_features].copy()
#             # Handle missing values with median imputation
#             numerical_df = numerical_df.fillna(numerical_df.median())
#         else:
#             # Create dummy numerical features
#             numerical_df = pd.DataFrame({'dummy_feature': [1] * len(df)})
#             available_features = ['dummy_feature']
        
#         # Create CYP interaction flags from available FooDB features
#         interaction_flags = pd.DataFrame()
#         foodb_flag_cols = [
#             'cyp_interactor', 'grapefruit_like', 'interacts_with_cyp3a4', 
#             'interacts_with_maoi', 'interacts_with_warfarin', 
#             'pathway_activation', 'potential_drug_interactor'
#         ]
        
#         print("Creating interaction flags...")
#         for col in foodb_flag_cols:
#             if col in df.columns:
#                 interaction_flags[col] = df[col].fillna(0).astype(int)
#                 print(f"  ✅ Added {col}")
        
#         # Add severity encoding if available
#         if 'severity_encoded' in df.columns:
#             interaction_flags['severity_encoded'] = df['severity_encoded'].fillna(0)
#             print("  ✅ Added severity_encoded")
        
#         # Combine all numerical features
#         if not interaction_flags.empty:
#             print(f"Including interaction flags: {list(interaction_flags.columns)}")
#             numerical_features = np.hstack([
#                 numerical_df.values,
#                 interaction_flags.values
#             ])
#             available_features.extend(list(interaction_flags.columns))
#         else:
#             numerical_features = numerical_df.values
        
#         # Standardize features
#         numerical_features = self.scaler.fit_transform(numerical_features)
        
#         print(f"✅ Numerical features shape: {numerical_features.shape}")
#         print(f"Final feature names: {available_features}")
#         return numerical_features, available_features
    
#     def create_final_feature_set(self, drug_embeddings, food_embeddings, numerical_features):
#         """
#         Combine all features into final feature set with alignment
#         """
#         print("Creating final feature set...")
        
#         # Ensure all arrays have the same number of rows
#         n_samples = min(len(drug_embeddings), len(food_embeddings), len(numerical_features))
#         print(f"Aligning to {n_samples} samples")
        
#         drug_embeddings = drug_embeddings[:n_samples]
#         food_embeddings = food_embeddings[:n_samples]
#         numerical_features = numerical_features[:n_samples]
        
#         # Combine all features
#         final_features = np.hstack([
#             drug_embeddings,
#             food_embeddings,
#             numerical_features
#         ])
        
#         print(f"✅ Final feature set shape: {final_features.shape}")
#         return final_features
    
#     def process_pipeline(self, mapped_drugs, drugbank_df, foodb_df):
#         """
#         Complete pipeline for feature fusion - BioBERT only
#         """
#         print("🚀 Starting Drug-Food Interaction Feature Fusion Pipeline...")
#         print("📱 MacBook Air M3 Optimized - BioBERT/Sentence-BERT Only")
#         print("="*70)
        
#         # Step 1: Load models
#         print("\n📦 Step 1: Loading Models...")
#         self.load_models()
        
#         # Step 2: Merge datasets
#         print("\n🔗 Step 2: Merging Datasets...")
#         merged_df = self.merge_datasets(mapped_drugs, drugbank_df, foodb_df)
        
#         # Step 3: Create embeddings
#         print("\n🧠 Step 3: Creating Drug Embeddings...")
#         drug_embeddings = self.create_drug_embeddings(merged_df)
        
#         print("\n🍎 Step 4: Creating Food Embeddings...")
#         food_embeddings = self.create_food_embeddings(merged_df)
        
#         # Step 5: Extract numerical features
#         print("\n🔢 Step 5: Extracting Numerical Features...")
#         numerical_features, feature_names = self.extract_numerical_features(merged_df)
        
#         # Step 6: Create final feature set
#         print("\n🎯 Step 6: Creating Final Feature Set...")
#         final_features = self.create_final_feature_set(
#             drug_embeddings, food_embeddings, numerical_features
#         )
        
#         # Create feature metadata
#         feature_info = {
#             'drug_embedding_dims': drug_embeddings.shape[1],
#             'food_embedding_dims': food_embeddings.shape[1],
#             'numerical_feature_names': feature_names,
#             'total_features': final_features.shape[1],
#             'device_used': str(self.device),
#             'mps_enabled': self.device.type == 'mps',
#             'model_type': 'BioBERT + Sentence-BERT',
#             'dataset_shape': merged_df.shape
#         }
        
#         print("\n" + "="*70)
#         print("✅ Pipeline completed successfully!")
#         print(f"📊 Final dataset shape: {final_features.shape}")
#         print(f"🔧 Device used: {self.device}")
#         print(f"⚡ MPS acceleration: {'✅' if self.device.type == 'mps' else '❌'}")
#         print(f"📈 Feature breakdown:")
#         print(f"   • Drug embeddings: {feature_info['drug_embedding_dims']} features")
#         print(f"   • Food embeddings: {feature_info['food_embedding_dims']} features")
#         print(f"   • Numerical features: {len(feature_info['numerical_feature_names'])} features")
        
#         return final_features, merged_df, feature_info


# def main():
#     """
#     Example usage optimized for MacBook Air M3
#     """
#     print("🍎 Drug-Food Feature Fusion - MacBook Air M3 Edition")
#     print("="*50)
    
#     # Check PyTorch compatibility first
#     print("🔍 Checking PyTorch compatibility...")
#     if not check_pytorch_compatibility():
#         print("❌ PyTorch compatibility issues detected")
#         print("💡 Try running: pip install --upgrade torch transformers")
#         return
    
#     # Initialize the pipeline
#     fusion_pipeline = DrugFoodFeatureFusion()
    
#     # File paths
#     file_paths = {
#         'mapped_drugs': '/Users/sachidhoka/Desktop/mapped_interactions.csv',
#         'drugbank_df': '/Users/sachidhoka/Desktop/processed_drugbank_data.csv',
#         'foodb_df': '/Users/sachidhoka/Desktop/processed_foodb.csv'
#     }
    
#     # Check file existence
#     missing_files = []
#     for name, path in file_paths.items():
#         if not os.path.exists(path):
#             missing_files.append(path)
    
#     if missing_files:
#         print("❌ Missing files:")
#         for file in missing_files:
#             print(f"   • {file}")
#         print("\n🔧 Please ensure all files exist before running the pipeline.")
#         return
    

    
#     try:
#         # Load datasets
#         print("📂 Loading datasets...")
#         mapped_drugs = pd.read_csv(file_paths['mapped_drugs'])
#         drugbank_df = pd.read_csv(file_paths['drugbank_df'])
#         foodb_df = pd.read_csv(file_paths['foodb_df'])
        
#         print(f"✅ Loaded datasets:")
#         print(f"   • Mapped drugs: {mapped_drugs.shape}")
#         print(f"   • DrugBank: {drugbank_df.shape}")
#         print(f"   • FooDB: {foodb_df.shape}")
        
#         # Process the pipeline
#         final_features, merged_df, feature_info = fusion_pipeline.process_pipeline(
#             mapped_drugs, drugbank_df, foodb_df
#         )
        
#         # Save results
#         print("\n💾 Saving results...")
#         np.save('final_features.npy', final_features)
#         merged_df.to_csv('merged_drug_food_data.csv', index=False)
        
#         # Save feature info
#         import json
#         with open('feature_info.json', 'w') as f:
#             json.dump(feature_info, f, indent=2)
        
#         print("✅ Results saved:")
#         print("   • final_features.npy")
#         print("   • merged_drug_food_data.csv")
#         print("   • feature_info.json")
#         print("\n🎉 Feature fusion pipeline completed successfully!")
        
#     except Exception as e:
#         print(f"❌ Error in pipeline execution: {e}")
#         import traceback
#         traceback.print_exc()
        
#         # Provide troubleshooting suggestions
#         print("\n🔧 Troubleshooting suggestions:")
#         print("1. pip install --upgrade torch torchvision torchaudio")
#         print("2. pip install --upgrade transformers==4.35.0")
#         print("3. pip install --upgrade sentence-transformers")
#         print("4. Restart your Python kernel/environment")


# if __name__ == "__main__":
#     main()

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