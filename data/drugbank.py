import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
import re
import warnings
warnings.filterwarnings('ignore')

class DrugBankPreprocessor:
    def __init__(self, data_path=None, df=None):
        """
        Initialize the preprocessor with either a file path or DataFrame
        """
        if df is not None:
            self.df = df.copy()
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Either data_path or df must be provided")
        
        self.scaler = None
        self.imputer = None
        
    def extract_physicochemical_properties(self):
        """
        Extract physicochemical properties from available text fields
        """
        # Patterns to extract properties from text fields
        patterns = {
            'molecular_weight': [
                r'molecular weight[:\s]*(\d+\.?\d*)\s*(?:g/mol|da|daltons?)',
                r'mw[:\s]*(\d+\.?\d*)',
                r'(\d+\.?\d*)\s*(?:g/mol|da|daltons?)',
            ],
            'logp': [
                r'log\s*p[:\s]*(-?\d+\.?\d*)',
                r'logp[:\s]*(-?\d+\.?\d*)',
                r'partition coefficient[:\s]*(-?\d+\.?\d*)',
            ],
            'h_bond_donors': [
                r'hydrogen bond donor[s]?[:\s]*(\d+)',
                r'h[-\s]?bond donor[s]?[:\s]*(\d+)',
                r'hbd[:\s]*(\d+)',
            ],
            'h_bond_acceptors': [
                r'hydrogen bond acceptor[s]?[:\s]*(\d+)',
                r'h[-\s]?bond acceptor[s]?[:\s]*(\d+)',
                r'hba[:\s]*(\d+)',
            ],
            'polar_surface_area': [
                r'polar surface area[:\s]*(\d+\.?\d*)',
                r'psa[:\s]*(\d+\.?\d*)',
                r'tpsa[:\s]*(\d+\.?\d*)',
            ],
            'rotatable_bonds': [
                r'rotatable bond[s]?[:\s]*(\d+)',
                r'rotatable[:\s]*(\d+)',
            ],
            'bioavailability': [
                r'bioavailability[:\s]*(\d+\.?\d*)%?',
                r'oral bioavailability[:\s]*(\d+\.?\d*)%?',
            ],
            'pka': [
                r'pka[:\s]*(\d+\.?\d*)',
                r'pk[:\s]*(\d+\.?\d*)',
            ]
        }
        
        # Text fields to search for properties
        text_fields = ['description', 'pharmacodynamics', 'mechanism-of-action', 
                      'absorption', 'metabolism', 'toxicity']
        
        # Initialize columns for physicochemical properties with NaN values
        for prop in patterns.keys():
            if prop not in self.df.columns:
                self.df[prop] = np.nan
        
        # Extract properties for each row
        for idx, row in self.df.iterrows():
            # Combine all text fields for searching
            combined_text = ""
            for field in text_fields:
                if field in row and pd.notna(row[field]):
                    combined_text += str(row[field]).lower() + " "
            
            # Extract properties using regex patterns
            for prop, pattern_list in patterns.items():
                found_value = None
                for pattern in pattern_list:
                    matches = re.findall(pattern, combined_text, re.IGNORECASE)
                    if matches:
                        try:
                            found_value = float(matches[0])
                            break
                        except (ValueError, IndexError):
                            continue
                
                # Set the found value or keep NaN
                if found_value is not None:
                    self.df.at[idx, prop] = found_value
        
        return self.df
    
    def extract_numeric_properties(self):
        """
        Extract numeric values from existing columns that contain numeric data
        """
        numeric_columns = ['half-life', 'protein-binding', 'volume-of-distribution', 'clearance']
        
        for col in numeric_columns:
            if col in self.df.columns:
                # Extract numeric values from text
                numeric_values = []
                for value in self.df[col]:
                    if pd.isna(value):
                        numeric_values.append(np.nan)
                    else:
                        # Extract first number found in the text
                        numbers = re.findall(r'(\d+\.?\d*)', str(value))
                        if numbers:
                            try:
                                numeric_values.append(float(numbers[0]))
                            except ValueError:
                                numeric_values.append(np.nan)
                        else:
                            numeric_values.append(np.nan)
                
                # Create new numeric column
                new_col_name = f'{col}_numeric'
                self.df[new_col_name] = numeric_values
        
        return self.df
    
    def handle_missing_values(self, strategy='median', k_neighbors=5):
        """
        Handle missing values using imputation
        
        Parameters:
        strategy: 'drop', 'median', 'mean', 'knn'
        k_neighbors: number of neighbors for KNN imputation
        """
        # Get numeric columns that actually exist in the DataFrame
        potential_numeric_cols = ['molecular_weight', 'logp', 'h_bond_donors', 'h_bond_acceptors',
                                'polar_surface_area', 'rotatable_bonds', 'bioavailability', 
                                'pka', 'half-life_numeric', 'protein-binding_numeric',
                                'volume-of-distribution_numeric', 'clearance_numeric']
        
        numeric_cols = [col for col in potential_numeric_cols if col in self.df.columns]
        
        if not numeric_cols:
            print("No numeric columns found for imputation.")
            return self.df
        
        print(f"Processing {len(numeric_cols)} numeric columns: {numeric_cols}")
        
        if strategy == 'drop':
            # Drop rows with any missing values in numeric columns
            initial_shape = self.df.shape
            self.df = self.df.dropna(subset=numeric_cols)
            print(f"Dropped {initial_shape[0] - self.df.shape[0]} rows with missing values")
        
        elif strategy in ['median', 'mean', 'knn']:
            # First check if any columns have all missing values
            cols_with_all_missing = [col for col in numeric_cols if self.df[col].isna().all()]
            
            if cols_with_all_missing:
                print(f"Warning: Columns with all missing values will be dropped: {cols_with_all_missing}")
                numeric_cols = [col for col in numeric_cols if col not in cols_with_all_missing]
                
            if not numeric_cols:
                print("No columns left for imputation after removing all-missing columns")
                return self.df
                
            if strategy in ['median', 'mean']:
                # Use SimpleImputer
                self.imputer = SimpleImputer(strategy=strategy)
                imputed_data = self.imputer.fit_transform(self.df[numeric_cols])
            elif strategy == 'knn':
                # Use KNN imputation
                self.imputer = KNNImputer(n_neighbors=k_neighbors)
                imputed_data = self.imputer.fit_transform(self.df[numeric_cols])
            
            # Convert back to DataFrame to maintain column names
            imputed_df = pd.DataFrame(imputed_data, columns=numeric_cols, index=self.df.index)
            
            # Update the original DataFrame
            for col in numeric_cols:
                self.df[col] = imputed_df[col]
            
            print(f"Imputed missing values using {strategy} strategy")
        
        return self.df
    
    def normalize_features(self, scaler_type='minmax'):
        """
        Normalize numeric features
        
        Parameters:
        scaler_type: 'minmax' or 'standard'
        """
        # Get numeric columns that actually exist in the DataFrame
        potential_numeric_cols = ['molecular_weight', 'logp', 'h_bond_donors', 'h_bond_acceptors',
                                'polar_surface_area', 'rotatable_bonds', 'bioavailability', 
                                'pka', 'half-life_numeric', 'protein-binding_numeric',
                                'volume-of-distribution_numeric', 'clearance_numeric']
        
        numeric_cols = [col for col in potential_numeric_cols if col in self.df.columns]
        
        if not numeric_cols:
            print("No numeric columns found for normalization.")
            return self.df
        
        print(f"Normalizing {len(numeric_cols)} columns using {scaler_type} scaler")
        
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError("scaler_type must be 'minmax' or 'standard'")
        
        # Fit and transform the data
        scaled_data = self.scaler.fit_transform(self.df[numeric_cols])
        
        # Convert back to DataFrame to maintain column names
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols, index=self.df.index)
        
        # Update the original DataFrame
        for col in numeric_cols:
            self.df[col] = scaled_df[col]
        
        return self.df
    
    def get_feature_statistics(self):
        """
        Get statistics about extracted features
        """
        potential_numeric_cols = ['molecular_weight', 'logp', 'h_bond_donors', 'h_bond_acceptors',
                                'polar_surface_area', 'rotatable_bonds', 'bioavailability', 
                                'pka', 'half-life_numeric', 'protein-binding_numeric',
                                'volume-of-distribution_numeric', 'clearance_numeric']
        
        numeric_cols = [col for col in potential_numeric_cols if col in self.df.columns]
        
        stats = {}
        for col in numeric_cols:
            if col in self.df.columns:
                stats[col] = {
                    'count': self.df[col].count(),
                    'missing': self.df[col].isna().sum(),
                    'mean': self.df[col].mean(),
                    'std': self.df[col].std(),
                    'min': self.df[col].min(),
                    'max': self.df[col].max()
                }
        
        return pd.DataFrame(stats).T
    
    def preprocess(self, missing_strategy='median', scaler_type='minmax', k_neighbors=5):
        """
        Complete preprocessing pipeline
        """
        print("Extracting physicochemical properties...")
        self.extract_physicochemical_properties()
        
        print("Extracting numeric properties from existing columns...")
        self.extract_numeric_properties()
        
        print("Handling missing values...")
        self.handle_missing_values(strategy=missing_strategy, k_neighbors=k_neighbors)
        
        print("Normalizing features...")
        self.normalize_features(scaler_type=scaler_type)
        
        print("Preprocessing complete!")
        return self.df

# Example usage:
def main():
    # METHOD 1: Load dataset using file path directly in constructor
    preprocessor = DrugBankPreprocessor(data_path='/Users/sachidhoka/Desktop/converter/drugbank_final.csv')
    
    # OR METHOD 2: Load dataset first, then pass DataFrame
    # df = pd.read_csv('path/to/your/drugbank_dataset.csv')
    # preprocessor = DrugBankPreprocessor(df=df)
    
    # Run complete preprocessing pipeline
    processed_df = preprocessor.preprocess(
        missing_strategy='median',  # or 'mean', 'knn', 'drop'
        scaler_type='minmax',      # or 'standard'
        k_neighbors=5
    )
    
    # Get feature statistics
    stats = preprocessor.get_feature_statistics()
    print("\nFeature Statistics:")
    print(stats)
    
    # Save processed data
    processed_df.to_csv('processed_drugbank_data.csv', index=False)
    print("\nProcessed data saved to 'processed_drugbank_data.csv'")

if __name__ == "__main__":
    main()