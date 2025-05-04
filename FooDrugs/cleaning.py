import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from collections import Counter
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# Path to your CSV file
file_path = "/Users/sachidhoka/Desktop/food-drug interactions.csv"

# Try to read with error_bad_lines=False (renamed to on_bad_lines in newer pandas)
try:
    # For newer pandas versions (1.3.0+)
    df = pd.read_csv(file_path, on_bad_lines='skip')
except TypeError:
    # For older pandas versions
    df = pd.read_csv(file_path, error_bad_lines=False)

# Display the first few rows to check the data
print(f"DataFrame shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# Check for column count inconsistency
print("\nColumn count per row:")
with open(file_path, 'r') as f:
    for i, line in enumerate(f):
        if i <= 5 or i == 21516 or i == 21517 or i == 21518:  # Show first few rows and problematic row
            fields = line.strip().split(',')
            print(f"Line {i+1}: {len(fields)} fields")

file_path = "/Users/sachidhoka/Desktop/food-drug interactions.csv"
def clean_foodrugs_dataset(file_path, output_path=None):
    """
    Clean the Food-Drug Interactions dataset for use in an AI prediction model.
    
    Parameters:
    -----------
    file_path : str
        Path to the raw CSV file
    output_path : str, optional
        Path to save the cleaned dataset

    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame
    """
    print("Loading data...")
    # Read with more robust settings to handle formatting issues
    try:
        df = pd.read_csv("/Users/sachidhoka/Desktop/food-drug interactions.csv", 
                       on_bad_lines='skip', 
                       low_memory=False,
                       encoding='utf-8')
    except:
        # Try alternative encoding if the first attempt fails
        df = pd.read_csv("/Users/sachidhoka/Desktop/food-drug interactions.csv", 
                       on_bad_lines='skip', 
                       low_memory=False,
                       encoding='latin1')
    
    print(f"Original dataset shape: {df.shape}")
    
    # STEP 1: Fix column data types
    print("Fixing data types...")
    # First, examine the columns to identify issues
    for col in df.columns:
        sample_values = df[col].dropna().head(3).tolist()
        print(f"Sample values for {col}: {sample_values}")
    
    # Convert ID columns to strings first to handle mixed types
    for col in ['TM_interactions_ID', 'texts_ID', 'start_index', 'end_index']:
        if col in df.columns:
            # First convert everything to string
            df[col] = df[col].astype(str)
            
            # Check for and handle URLs
            url_mask = df[col].str.contains('http', na=False)
            if url_mask.any():
                print(f"Found {url_mask.sum()} URLs in {col}")
                # Create URL column if it doesn't exist
                if 'url' not in df.columns:
                    df['url'] = np.nan
                df.loc[url_mask, 'url'] = df.loc[url_mask, col]
                df.loc[url_mask, col] = np.nan
            
            # Try numeric conversion for non-URL values
            try:
                # Use regex to check if the column contains only digits
                numeric_mask = df[col].notna() & df[col].str.match(r'^\d+$', na=False)
                if numeric_mask.any():
                    df.loc[numeric_mask, col] = pd.to_numeric(df.loc[numeric_mask, col], errors='coerce')
                else:
                    print(f"Column {col} does not contain numeric values")
            except Exception as e:
                print(f"Error converting {col}: {e}")
    
    # STEP 2: Handle unwanted columns
    print("Handling unwanted columns...")
    # Remove likely erroneous columns
    cols_to_drop = [')', 'LOCK', 'INSERT']
    cols_to_drop += [col for col in df.columns if col.startswith('Unnamed:')]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # STEP 3: Clean food and drug names
    print("Cleaning food and drug names...")
    
    # Function to clean text fields
    def clean_text(text):
        if pd.isna(text):
            return np.nan
        
        # Convert to string if not already
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters but keep hyphen and space
        text = re.sub(r'[^\w\s\-]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text if text else np.nan
    
    # Apply cleaning to food and drug columns
    df['food'] = df['food'].apply(clean_text)
    df['drug'] = df['drug'].apply(clean_text)
    
    # STEP 4: Remove duplicates
    print("Removing duplicates...")
    # First check how many duplicates exist
    n_duplicates = df.duplicated(subset=['food', 'drug']).sum()
    print(f"Found {n_duplicates} duplicate food-drug pairs")
    
    # Remove duplicate food-drug combinations
    df = df.drop_duplicates(subset=['food', 'drug'])
    
    # STEP 5: Handle missing values
    print("Handling missing values...")
    # Remove rows where both food and drug are missing
    df = df.dropna(subset=['food', 'drug'], how='all')
    
    # For rows with either food or drug missing
    missing_food = df['food'].isna().sum()
    missing_drug = df['drug'].isna().sum()
    print(f"Rows with missing food: {missing_food}")
    print(f"Rows with missing drug: {missing_drug}")
    
    # Drop rows with missing food or drug for better model training
    df = df.dropna(subset=['food', 'drug'], how='any')
    
    # STEP 6: Handle rows with URLs as text_ID
    if 'texts_ID' in df.columns:
        url_mask = df['texts_ID'].astype(str).str.contains('http', na=False)
        if url_mask.any():
            print(f"Found {url_mask.sum()} rows with URLs in texts_ID")
            # Create URL column if it doesn't exist
            if 'url' not in df.columns:
                df['url'] = np.nan
            
            # Move URLs to dedicated column
            df.loc[url_mask, 'url'] = df.loc[url_mask, 'texts_ID']
            df.loc[url_mask, 'texts_ID'] = np.nan
    
    # STEP 7: Standardize food and drug names
    print("Standardizing food and drug names...")
    
    # Food normalization for common food items in drug interactions
    food_replacements = {
        'grapefruit': 'grapefruit',
        'grapefruit juice': 'grapefruit',
        'grapefruits': 'grapefruit',
        'dairy': 'dairy',
        'milk': 'dairy',
        'cheese': 'dairy',
        'yogurt': 'dairy',
        'citrus': 'citrus',
        'orange': 'citrus',
        'lemon': 'citrus',
        'lime': 'citrus',
        'alcohol': 'alcohol',
        'wine': 'alcohol',
        'beer': 'alcohol',
        'spirits': 'alcohol',
        'caffeine': 'caffeine',
        'coffee': 'caffeine',
        'tea': 'caffeine',
        'energy drink': 'caffeine',
        'leafy greens': 'leafy greens',
        'spinach': 'leafy greens',
        'kale': 'leafy greens'
    }
    
    # Apply food normalization where matches exist
    for old, new in food_replacements.items():
        df.loc[df['food'] == old, 'food'] = new
    
    # STEP 8: Final cleanup
    print("Performing final cleanup...")
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Save cleaned file if output path is provided
    if output_path:
        # Make sure the directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            print(f"Creating directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            
        print(f"Saving cleaned data to {output_path}")
        df.to_csv(output_path, index=False)
    
    print(f"Final dataset shape: {df.shape}")
    
    # Print information about unique foods and drugs
    print(f"Number of unique foods: {df['food'].nunique()}")
    print(f"Number of unique drugs: {df['drug'].nunique()}")
    print(f"Number of unique food-drug interactions: {df.shape[0]}")
    
    return df

def analyze_foodrugs_dataset(df):
    """
    Analyze the cleaned Food-Drug Interactions dataset to gain insights
    for the Drug-Food Interaction Predictor project.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned DataFrame
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    print("\n--- ANALYSIS ---")
    print(f"Dataset shape: {df.shape}")
    
    # Basic statistics
    results = {}
    results['total_interactions'] = len(df)
    results['unique_foods'] = df['food'].nunique()
    results['unique_drugs'] = df['drug'].nunique()
    
    print("\n--- Basic Statistics ---")
    print(f"Total interactions: {results['total_interactions']}")
    print(f"Unique foods: {results['unique_foods']}")
    print(f"Unique drugs: {results['unique_drugs']}")
    
    # Most common foods involved in interactions
    top_foods = df['food'].value_counts().head(20)
    results['top_foods'] = top_foods.to_dict()
    
    print("\n--- Top 20 Foods Involved in Drug Interactions ---")
    for food, count in top_foods.items():
        print(f"{food}: {count}")
    
    # Most common drugs involved in food interactions
    top_drugs = df['drug'].value_counts().head(20)
    results['top_drugs'] = top_drugs.to_dict()
    
    print("\n--- Top 20 Drugs Involved in Food Interactions ---")
    for drug, count in top_drugs.items():
        print(f"{drug}: {count}")
    
    # Check for common food-drug patterns
    # Group data by drug and count unique foods that interact with each drug
    drug_food_counts = df.groupby('drug')['food'].nunique().sort_values(ascending=False)
    results['drugs_with_most_food_interactions'] = drug_food_counts.head(20).to_dict()
    
    print("\n--- Drugs with Most Diverse Food Interactions ---")
    for drug, count in drug_food_counts.head(20).items():
        print(f"{drug}: {count} different food interactions")
    
    # Group data by food and count unique drugs that interact with each food
    food_drug_counts = df.groupby('food')['drug'].nunique().sort_values(ascending=False)
    results['foods_with_most_drug_interactions'] = food_drug_counts.head(20).to_dict()
    
    print("\n--- Foods with Most Diverse Drug Interactions ---")
    for food, count in food_drug_counts.head(20).items():
        print(f"{food}: {count} different drug interactions")
    
    # Analyze food-drug pair frequency
    pair_counts = df.groupby(['food', 'drug']).size().reset_index(name='count')
    top_pairs = pair_counts.sort_values('count', ascending=False).head(20)
    results['top_food_drug_pairs'] = top_pairs.to_dict('records')
    
    print("\n--- Most Common Food-Drug Interaction Pairs ---")
    for _, row in top_pairs.iterrows():
        print(f"{row['food']} - {row['drug']}: {row['count']} occurrences")
    
    return results

def prepare_for_modeling_with_smote(df, output_path=None, test_size=0.2, random_state=42):
    """
    Prepare the cleaned dataset for use in an AI prediction model with SMOTE applied.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned DataFrame
    output_path : str, optional
        Path to save the model-ready dataset
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing model-ready data
    """
    print("\n--- PREPARING FOR MODELING WITH SMOTE ---")
    
    # 1. Encode categorical variables
    # For simplicity we'll use label encoding for food and drug
    from sklearn.preprocessing import LabelEncoder
    
    # Create encoders
    food_encoder = LabelEncoder()
    drug_encoder = LabelEncoder()
    
    # Apply encoders
    df['food_encoded'] = food_encoder.fit_transform(df['food'])
    df['drug_encoded'] = drug_encoder.fit_transform(df['drug'])
    
    # 2. Create binary interaction indicator (for classification)
    # All rows in the original dataset represent interactions
    df['has_interaction'] = 1
    
    # 3. Create negative samples (non-interacting pairs)
    print("Creating negative samples for balanced training...")
    
    # Get unique foods and drugs
    unique_foods = df['food'].unique()
    unique_drugs = df['drug'].unique()
    
    # Create a set of existing food-drug pairs for quick lookup
    existing_pairs = set(zip(df['food'], df['drug']))
    
    # Create negative samples
    neg_samples = []
    n_neg_samples = min(len(df), 5000)  # Limit to avoid huge datasets
    
    np.random.seed(random_state)
    while len(neg_samples) < n_neg_samples:
        food = np.random.choice(unique_foods)
        drug = np.random.choice(unique_drugs)
        
        # Check that this pair doesn't exist in our dataset
        if (food, drug) not in existing_pairs:
            neg_samples.append({
                'food': food, 
                'drug': drug,
                'has_interaction': 0,  # No interaction
                'food_encoded': food_encoder.transform([food])[0],
                'drug_encoded': drug_encoder.transform([drug])[0]
            })
            # Add to existing pairs to avoid duplicates
            existing_pairs.add((food, drug))
    
    # Create DataFrame from negative samples
    neg_df = pd.DataFrame(neg_samples)
    
    # 4. Combine positive and negative samples
    print("Combining positive and negative samples...")
    model_df = pd.concat([
        df[['food', 'drug', 'food_encoded', 'drug_encoded', 'has_interaction']],
        neg_df
    ], ignore_index=True)
    
    # 5. Prepare for train-test split
    print("Preparing train-test split...")
    X = model_df[['food_encoded', 'drug_encoded']]
    y = model_df['has_interaction']
    
    # 6. Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 7. Apply SMOTE to handle class imbalance
    print("Applying SMOTE to training data...")
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Report class balance after SMOTE
    print(f"Class distribution before SMOTE: {Counter(y_train)}")
    print(f"Class distribution after SMOTE: {Counter(y_train_smote)}")
    
    # 8. Add frequency-based features
    model_df['food_frequency'] = model_df['food'].map(model_df['food'].value_counts())
    model_df['drug_frequency'] = model_df['drug'].map(model_df['drug'].value_counts())
    
    # 9. Save preprocessed data if path provided
    if output_path:
        # Make sure the directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            print(f"Creating directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            
        # Save main dataset
        model_df.to_csv(output_path, index=False)
        print(f"Saved model-ready data to {output_path}")
        
        # Save SMOTE-processed training data
        smote_df = pd.DataFrame(
            data=np.column_stack([X_train_smote, y_train_smote]),
            columns=['food_encoded', 'drug_encoded', 'has_interaction']
        )
        smote_path = os.path.join(os.path.dirname(output_path), "smote_training_data.csv")
        smote_df.to_csv(smote_path, index=False)
        print(f"Saved SMOTE-processed training data to {smote_path}")
    
    print("Data preparation with SMOTE complete.")
    print("\nSample of model-ready data:")
    print(model_df.head())
    
    return {
        'model_df': model_df,
        'X_train': X_train,
        'X_test': X_test, 
        'y_train': y_train,
        'y_test': y_test,
        'X_train_smote': X_train_smote,
        'y_train_smote': y_train_smote
    }

def visualize_insights(df, output_dir):
    """
    Create visualizations from the food-drug interaction data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned DataFrame
    output_dir : str
        Directory to save visualizations
    """
    print("\n--- CREATING VISUALIZATIONS ---")
    
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. Top foods visualization
        plt.figure(figsize=(12, 8))
        top_foods = df['food'].value_counts().head(15)
        sns.barplot(x=top_foods.values, y=top_foods.index)
        plt.title("Top 15 Foods Involved in Drug Interactions")
        plt.xlabel("Number of Interactions")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_foods_interactions.png"))
        plt.close()
        
        # 2. Top drugs visualization
        plt.figure(figsize=(12, 8))
        top_drugs = df['drug'].value_counts().head(15)
        sns.barplot(x=top_drugs.values, y=top_drugs.index)
        plt.title("Top 15 Drugs Involved in Food Interactions")
        plt.xlabel("Number of Interactions")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_drugs_interactions.png"))
        plt.close()
        
        # 3. Foods with most diverse drug interactions
        plt.figure(figsize=(12, 8))
        food_drug_counts = df.groupby('food')['drug'].nunique().sort_values(ascending=False).head(15)
        sns.barplot(x=food_drug_counts.values, y=food_drug_counts.index)
        plt.title("Foods Interacting with Most Diverse Drugs")
        plt.xlabel("Number of Different Drugs")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "foods_diverse_interactions.png"))
        plt.close()
        
        print(f"Visualizations saved to {output_dir}")
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def process_foodrugs_data(input_file, output_dir):
    """
    Complete processing workflow for Food-Drug Interactions dataset with SMOTE
    
    Parameters:
    -----------
    input_file : str
        Path to raw data file
    output_dir : str
        Directory to save outputs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths
    cleaned_path = os.path.join(output_dir, "cleaned_food_drug_interactions.csv")
    model_ready_path = os.path.join(output_dir, "model_ready_food_drug_interactions.csv")
    
    # Step 1: Clean the dataset
    print("\n=== CLEANING DATASET ===")
    cleaned_df = clean_foodrugs_dataset(input_file, cleaned_path)
    
    # Step 2: Analyze the cleaned dataset
    print("\n=== ANALYZING DATASET ===")
    analysis_results = analyze_foodrugs_dataset(cleaned_df)
    
    # Step 3: Prepare data for modeling with SMOTE
    print("\n=== PREPARING FOR MODELING WITH SMOTE ===")
    model_data = prepare_for_modeling_with_smote(cleaned_df, model_ready_path)
    
    # Step 4: Create visualizations
    print("\n=== CREATING VISUALIZATIONS ===")
    vis_dir = os.path.join(output_dir, "visualizations")
    visualize_insights(cleaned_df, vis_dir)
    
    print("\n=== PROCESSING COMPLETE ===")
    print(f"Files saved to: {output_dir}")
    
    return {
        'cleaned_df': cleaned_df,
        'model_data': model_data,
        'analysis': analysis_results
    }

# Example usage
if __name__ == "__main__":
    # Use actual file path for input
    input_file = '/Users/sachidhoka/Desktop/food-drug interactions.csv'
    
    # Use actual directory for output files
    output_dir = 'food_drug_analysis'
    
    # Process the data
    results = process_foodrugs_data(input_file, output_dir)