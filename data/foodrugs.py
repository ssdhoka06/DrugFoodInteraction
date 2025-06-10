import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the dataset with error handling
file_path = '/Users/sachidhoka/Desktop/food-drug interactions.csv'

# Try different parsing options to handle the CSV parsing error
try:
    # First, try with error handling for bad lines
    df = pd.read_csv(file_path, on_bad_lines='skip')
    print("Successfully loaded with bad lines skipped")
except Exception as e:
    print(f"First attempt failed: {e}")
    try:
        # Try with different separator or quoting
        df = pd.read_csv(file_path, sep=',', quotechar='"', on_bad_lines='skip')
        print("Successfully loaded with quote handling")
    except Exception as e:
        print(f"Second attempt failed: {e}")
        try:
            # Try reading with engine='python' for more robust parsing
            df = pd.read_csv(file_path, engine='python', on_bad_lines='skip')
            print("Successfully loaded with Python engine")
        except Exception as e:
            print(f"All attempts failed: {e}")
            exit()

print("Original dataset shape:", df.shape)
print("\nOriginal columns:", df.columns.tolist())

# Check for duplicates before processing
print(f"\nDuplicate Analysis:")
print(f"Total rows: {len(df)}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Duplicate percentage: {(df.duplicated().sum() / len(df) * 100):.2f}%")

# Remove duplicates
df_no_duplicates = df.drop_duplicates()
print(f"\nAfter removing duplicates:")
print(f"Remaining rows: {len(df_no_duplicates)}")
print(f"Rows removed: {len(df) - len(df_no_duplicates)}")

# Update df to use the deduplicated version
df = df_no_duplicates

print("\nFirst few rows after deduplication:")
print(df.head())

# Extract relevant columns
# Based on your columns, we have 'drug' and 'food' columns
# We'll need to create interaction_description and severity_label columns

# Create a clean dataframe with the required columns
processed_df = pd.DataFrame()

# Extract drug and food names
processed_df['drug_name'] = df['drug'].str.strip() if 'drug' in df.columns else None
processed_df['food_name'] = df['food'].str.strip() if 'food' in df.columns else None

# Create interaction description (combining available information)
processed_df['interaction_description'] = (
    "Interaction between " + 
    processed_df['drug_name'].astype(str) + 
    " and " + 
    processed_df['food_name'].astype(str)
)

# For demonstration, create severity labels based on common drug-food interactions
# In a real scenario, this would be based on actual interaction data or expert knowledge
def assign_severity(drug, food):
    """
    Assign severity based on known drug-food interactions
    This is a simplified example - in practice, you'd use a comprehensive database
    """
    drug = str(drug).lower()
    food = str(food).lower()
    
    # High-risk interactions
    high_risk_combinations = [
        ('warfarin', 'grapefruit'), ('simvastatin', 'grapefruit'),
        ('cyclosporine', 'grapefruit'), ('felodipine', 'grapefruit'),
        ('warfarin', 'cranberry'), ('phenytoin', 'alcohol'),
        ('metronidazole', 'alcohol')
    ]
    
    # Moderate-risk interactions
    moderate_risk_combinations = [
        ('calcium', 'dairy'), ('tetracycline', 'dairy'),
        ('iron', 'tea'), ('iron', 'coffee'),
        ('levothyroxine', 'soy'), ('digoxin', 'licorice')
    ]
    
    # Check for specific combinations
    for drug_pattern, food_pattern in high_risk_combinations:
        if drug_pattern in drug and food_pattern in food:
            return 'high-risk'
    
    for drug_pattern, food_pattern in moderate_risk_combinations:
        if drug_pattern in drug and food_pattern in food:
            return 'moderate-risk'
    
    # Default to low-risk for other combinations
    return 'low-risk'

# Apply severity assignment
processed_df['severity_label'] = processed_df.apply(
    lambda row: assign_severity(row['drug_name'], row['food_name']), axis=1
)

# Label encode the severity levels
label_encoder = LabelEncoder()
processed_df['severity_encoded'] = label_encoder.fit_transform(processed_df['severity_label'])

# Create mapping for reference
severity_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

print("\n" + "="*50)
print("PROCESSED DATASET")
print("="*50)

print(f"\nProcessed dataset shape: {processed_df.shape}")
print(f"\nColumns: {processed_df.columns.tolist()}")

print(f"\nSeverity Label Distribution:")
print(processed_df['severity_label'].value_counts())

print(f"\nLabel Encoding Mapping:")
for label, code in severity_mapping.items():
    print(f"  {label}: {code}")

print(f"\nFirst 10 rows of processed data:")
print(processed_df.head(10))

# Remove rows with missing drug or food names
processed_df_clean = processed_df.dropna(subset=['drug_name', 'food_name'])
processed_df_clean = processed_df_clean[
    (processed_df_clean['drug_name'] != 'nan') & 
    (processed_df_clean['food_name'] != 'nan')
]

# Remove duplicates from processed data as well (based on drug-food combinations)
print(f"\nDuplicate analysis in processed data:")
print(f"Total processed rows: {len(processed_df_clean)}")

# Check for duplicates based on drug-food combination
duplicate_combinations = processed_df_clean.duplicated(subset=['drug_name', 'food_name'])
print(f"Duplicate drug-food combinations: {duplicate_combinations.sum()}")

# Remove duplicate drug-food combinations, keeping the first occurrence
processed_df_clean = processed_df_clean.drop_duplicates(subset=['drug_name', 'food_name'], keep='first')
print(f"After removing duplicate combinations: {len(processed_df_clean)}")

print(f"\nAfter removing missing values and duplicates: {processed_df_clean.shape}")
print(f"\nFinal severity distribution:")
print(processed_df_clean['severity_label'].value_counts())

# Save the processed dataset
output_path = '/Users/sachidhoka/Desktop/processed_drug_food_interactions.csv'
processed_df_clean.to_csv(output_path, index=False)
print(f"\nProcessed dataset saved to: {output_path}")

# Display sample of final dataset
print(f"\nSample of final processed dataset:")
print(processed_df_clean.sample(5) if len(processed_df_clean) > 5 else processed_df_clean)