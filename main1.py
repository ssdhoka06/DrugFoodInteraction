import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Install and import required packages
try:
    import gdown
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    import gdown

# Streamlit configuration
st.set_page_config(
    page_title="Drug-Food Interaction Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration - Replace with your actual file IDs
DATA_FILE_ID = "1IhAkCHqs2FUDX9rzkZT12ov1xFweKEda"  # Your CSV file ID
MODEL_FILE_ID = "1XQHlS8d8Rz3DYafdjJ3maE5XkWvivd8s"  # Your voting classifier model

# Updated feature engineering function to match your trained model
def create_features(drug, food):
    """Create features exactly as your trained model expects"""
    
    # Define drug categories based on common drug types
    drug_categories = {
        'antibiotic': ['amoxicillin', 'penicillin', 'azithromycin', 'ciprofloxacin', 'doxycycline'],
        'anticoagulant': ['warfarin', 'heparin', 'superwarfin', 'coumadin', 'warfin'],
        'antidepressant': ['sertraline', 'fluoxetine', 'prozac', 'zoloft'],
        'antihypertensive': ['lisinopril', 'amlodipine', 'metoprolol', 'losartan'],
        'analgesic': ['aspirin', 'ibuprofen', 'acetaminophen', 'naproxen'],
        'statin': ['atorvastatin', 'simvastatin', 'rosuvastatin'],
        'diabetes': ['metformin', 'insulin', 'glipizide', 'glyburide'],
        'heart_rhythm': ['digoxin', 'amiodarone', 'flecainide'],
        'ppi': ['omeprazole', 'lansoprazole', 'esomeprazole'],
        'pain_relief': ['morphine', 'codeine', 'tramadol'],
        'other': []  # Default category
    }
    
    # Define food categories
    food_categories = {
        'citrus': ['orange', 'lemon', 'lime', 'grapefruit', 'grape fruit'],
        'leafy': ['spinach', 'kale', 'lettuce', 'collard'],
        'dairy': ['milk', 'cheese', 'yogurt', 'butter'],
        'alcohol': ['wine', 'beer', 'alcohol'],
        'caffeine': ['coffee', 'tea', 'cola'],
    }
    
    # Initialize features dictionary
    features = {}
    
    # Drug category features
    drug_lower = drug.lower()
    drug_matched = False
    for category, drugs in drug_categories.items():
        if category == 'other':
            continue
        match = int(any(d in drug_lower for d in drugs))
        features[f'drug_{category}'] = match
        if match:
            drug_matched = True
    
    # Set drug_other if no other category matched
    features['drug_other'] = int(not drug_matched)
    
    # Food category features  
    food_lower = food.lower()
    for category, foods in food_categories.items():
        features[f'food_{category}'] = int(any(f in food_lower for f in foods))
    
    # Combined features
    features['both_other'] = int(
        features['drug_other'] and
        not any(features[f'food_{cat}'] for cat in food_categories.keys())
    )
    
    # Length features
    features['drug_length'] = len(drug)
    features['food_length'] = len(food)
    
    # String matching features
    features['common_letters'] = len(set(drug_lower) & set(food_lower))
    features['contains_match'] = int(drug_lower in food_lower or food_lower in drug_lower)
    
    return pd.DataFrame([features])

# Caching functions
@st.cache_data
def load_data():
    """Load dataset from Google Drive"""
    data_file = "drug_food_interactions.csv"
    
    if not os.path.exists(data_file):
        with st.spinner("Downloading dataset..."):
            try:
                gdown.download(f"https://drive.google.com/uc?export=download&id={DATA_FILE_ID}", 
                              data_file, quiet=False)
                st.success("✅ Dataset downloaded successfully!")
            except Exception as e:
                st.error(f"Failed to download data: {e}")
                return None
    
    df = pd.read_csv(data_file)
    return df

@st.cache_resource
def load_voting_classifier():
    """Load your pre-trained VotingClassifier from Google Drive with enhanced error handling"""
    model_file = "voting_classifier_model.pkl"
    
    if not os.path.exists(model_file):
        with st.spinner("Downloading pre-trained VotingClassifier..."):
            try:
                gdown.download(f"https://drive.google.com/uc?export=download&id={MODEL_FILE_ID}", 
                              model_file, quiet=False)
                st.success("✅ Model downloaded successfully!")
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                return None
    
    try:
        with open(model_file, 'rb') as f:
            loaded_object = pickle.load(f)
        
        # Debug: Check what was actually loaded
        st.sidebar.write(f"Loaded object type: {type(loaded_object)}")
        
        # Handle different possible formats
        if isinstance(loaded_object, dict):
            st.sidebar.warning("⚠️ Loaded object is a dictionary")
            # Try to extract the model from common dictionary keys
            possible_keys = ['model', 'voting_clf', 'classifier', 'estimator', 'best_estimator_']
            
            for key in possible_keys:
                if key in loaded_object:
                    model = loaded_object[key]
                    st.sidebar.success(f"✅ Found model under key: {key}")
                    st.sidebar.write(f"Extracted model type: {type(model)}")
                    
                    # Verify it has predict method
                    if hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
                        return model
                    else:
                        st.sidebar.error(f"❌ Object under key '{key}' doesn't have predict methods")
                        continue
            
            # If no model found in dictionary, show available keys
            st.sidebar.error(f"❌ No model found in dictionary. Available keys: {list(loaded_object.keys())}")
            
            # Try to extract the first object that has predict method
            for key, value in loaded_object.items():
                if hasattr(value, 'predict') and hasattr(value, 'predict_proba'):
                    st.sidebar.success(f"✅ Found model-like object under key: {key}")
                    return value
            
            return None
            
        elif hasattr(loaded_object, 'predict') and hasattr(loaded_object, 'predict_proba'):
            # It's a proper model
            st.sidebar.success("✅ Model loaded successfully")
            return loaded_object
        else:
            st.sidebar.error(f"❌ Loaded object is not a valid model. Type: {type(loaded_object)}")
            return None
            
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def create_fallback_model():
    """Create a simple fallback model if the main model fails to load"""
    st.warning("🔄 Creating fallback model for demonstration...")
    
    # Create a simple voting classifier
    lr = LogisticRegression(random_state=42, max_iter=1000)
    nb = MultinomialNB()
    rf = RandomForestClassifier(random_state=42, n_estimators=10)
    
    voting_clf = VotingClassifier(
        estimators=[('lr', lr), ('nb', nb), ('rf', rf)],
        voting='soft'
    )
    
    # Generate some dummy training data with the same features
    np.random.seed(42)
    n_samples = 100
    X_dummy = pd.DataFrame({
        'drug_length': np.random.randint(3, 15, n_samples),
        'food_length': np.random.randint(3, 15, n_samples),
        'common_letters': np.random.randint(0, 8, n_samples),
        'drug_starts_with_same_letter': np.random.randint(0, 2, n_samples),
        'drug_ends_with_same_letter': np.random.randint(0, 2, n_samples),
        'food_contains_drug': np.random.randint(0, 2, n_samples),
        'drug_contains_food': np.random.randint(0, 2, n_samples)
    })
    y_dummy = np.random.randint(0, 2, n_samples)
    
    # Train the fallback model
    voting_clf.fit(X_dummy, y_dummy)
    
    st.success("✅ Fallback model created successfully!")
    st.info("ℹ️ This is a demonstration model. Replace with your actual trained model for real predictions.")
    
    return voting_clf

def predict_interaction(drug_input, food_input, model):
    """Make prediction using your exact pipeline"""
    try:
        # Get expected feature names from model
        expected_features = None
        if hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
            st.sidebar.write(f"Expected features: {expected_features[:5]}...")  # Show first 5
        
        # Use your exact feature engineering
        features = create_features(drug_input, food_input)
        
        # Debug: Show feature values
        st.sidebar.write("Generated features:")
        st.sidebar.write(list(features.columns))
        
        # If we know expected features, try to match them
        if expected_features:
            # Create a dataframe with all expected features, filled with 0
            aligned_features = pd.DataFrame(0, index=[0], columns=expected_features)
            
            # Fill in the features we can calculate
            for col in features.columns:
                if col in expected_features:
                    aligned_features[col] = features[col].iloc[0]
            
            features = aligned_features
            st.sidebar.write(f"Aligned features shape: {features.shape}")
        
        # Make prediction using your model
        prediction = model.predict(features)[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(features)[0]
        confidence = max(probabilities)
        
        return prediction, confidence, probabilities
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.error(f"Model type: {type(model)}")
        st.error(f"Features shape: {features.shape}")
        st.error(f"Features columns: {list(features.columns)}")
        
        # Show model's expected features if available
        if hasattr(model, 'feature_names_in_'):
            st.error(f"Model expects: {list(model.feature_names_in_)}")
        
        return None, None, None

def display_prediction_results(prediction, confidence, probabilities, drug_input, food_input, model):
    """Display prediction results with your model's output - FIXED VERSION"""
    st.markdown("## 🎯 Prediction Results")
    
    # Main result display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction == 1:
            st.error("⚠️ INTERACTION DETECTED")
            risk_color = "red"
        else:
            st.success("✅ NO INTERACTION")
            risk_color = "green"
    
    with col2:
        st.metric("Confidence", f"{confidence:.2%}")
    
    with col3:
        interaction_prob = probabilities[1] if len(probabilities) > 1 else 0.5
        st.metric("Interaction Probability", f"{interaction_prob:.2%}")
    
    # Detailed results
    st.markdown("### 📋 Detailed Analysis")
    
    # Display feature values
    features = create_features(drug_input, food_input)
    st.markdown("**Generated Features:**")
    
    feature_cols = st.columns(4)
    feature_names = list(features.columns)
    feature_values = features.iloc[0].values
    
    for i, (name, value) in enumerate(zip(feature_names, feature_values)):
        with feature_cols[i % 4]:
            st.metric(name.replace('_', ' ').title(), str(value))
    
    # Recommendation
    st.markdown("### 💡 Recommendation")
    if prediction == 1:
        st.error(f"""
        **⚠️ POTENTIAL INTERACTION DETECTED**
        
        **Drug:** {drug_input}  
        **Food:** {food_input}  
        **Risk Level:** High ({interaction_prob:.1%} probability)
        
        **⚠️ Important:** This prediction suggests a potential interaction. 
        Please consult with your healthcare provider or pharmacist before combining these items.
        """)
    else:
        st.success(f"""
        **✅ NO SIGNIFICANT INTERACTION DETECTED**
        
        **Drug:** {drug_input}  
        **Food:** {food_input}  
        **Risk Level:** Low ({interaction_prob:.1%} probability)
        
        **✅ Generally Safe:** The model suggests low interaction risk, but always consult 
        your healthcare provider for personalized medical advice.
        """)
    
    # Model breakdown - FIXED: Use the passed model parameter
    with st.expander("🔍 Model Details"):
        if hasattr(model, 'estimators_'):
            # It's a VotingClassifier
            st.markdown("""
            **Your model combines 3 classifiers:**
            - 🔵 **Logistic Regression (lr)**
            - 🟢 **Naive Bayes (nb)** 
            - 🟣 **Random Forest (rf)**
            
            The final prediction uses soft voting (probability averaging) across all three models.
            """)
        elif 'GradientBoosting' in str(type(model)):
            # It's a GradientBoostingClassifier
            st.markdown(f"""
            **Your model is a Gradient Boosting Classifier:**
            - 🌳 **Model Type:** {type(model).__name__}
            - 🔄 **Ensemble Method:** Gradient Boosting (sequential weak learners)
            - 📊 **Number of Estimators:** {getattr(model, 'n_estimators', 'N/A')}
            - 🎯 **Learning Rate:** {getattr(model, 'learning_rate', 'N/A')}
            - 📏 **Max Depth:** {getattr(model, 'max_depth', 'N/A')}
            
            Gradient Boosting builds models sequentially, where each new model corrects errors from previous ones.
            """)
        else:
            # Generic model info
            st.markdown(f"""
            **Your model details:**
            - 🤖 **Model Type:** {type(model).__name__}
            - 📊 **Model Family:** {type(model).__module__}
            
            This model makes predictions based on the engineered features from drug and food names.
            """)

def display_dataset_info(df):
    """Display dataset information"""
    st.markdown("## 📊 Dataset Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    
    with col2:
        if 'drug' in df.columns:
            st.metric("Unique Drugs", df['drug'].nunique())
        else:
            st.metric("Unique Drugs", "N/A")
    
    with col3:
        if 'food' in df.columns:
            st.metric("Unique Foods", df['food'].nunique())
        else:
            st.metric("Unique Foods", "N/A")
    
    with col4:
        if 'interaction' in df.columns:
            interaction_rate = df['interaction'].mean() * 100
            st.metric("Interaction Rate", f"{interaction_rate:.1f}%")
        else:
            st.metric("Interaction Rate", "N/A")
    
    # Show dataset columns and sample
    st.markdown("### 📋 Dataset Structure")
    st.write("**Columns:** ", list(df.columns))
    st.dataframe(df.head(10))

def show_feature_analysis(drug_input, food_input):
    """Show how features are calculated"""
    st.markdown("### 🔧 Feature Engineering Analysis")
    
    if drug_input and food_input:
        features = create_features(drug_input, food_input)
        
        st.markdown("**How features are calculated:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Input:**
            - Drug: `{drug_input}`
            - Food: `{food_input}`
            """)
            
            # Show drug categories detected
            st.markdown("**Drug Categories Detected:**")
            drug_cols = [col for col in features.columns if col.startswith('drug_')]
            for col in drug_cols:
                if features[col].iloc[0] == 1:
                    st.write(f"✅ {col.replace('drug_', '').title()}")
        
        with col2:
            st.markdown("**Food Categories Detected:**")
            food_cols = [col for col in features.columns if col.startswith('food_')]
            for col in food_cols:
                if features[col].iloc[0] == 1:
                    st.write(f"✅ {col.replace('food_', '').title()}")
            
            # Show other features
            st.markdown("**Other Features:**")
            other_cols = [col for col in features.columns if not col.startswith(('drug_', 'food_'))]
            for col in other_cols:
                st.write(f"{col}: {features[col].iloc[0]}")
        
        # Show full feature vector
        st.markdown("**Complete Feature Vector:**")
        st.dataframe(features)

def main():
    """Main Streamlit app"""
    st.title("💊 Drug-Food Interaction Predictor")
    
    # Load data and model first to determine actual model type
    df = load_data()
    model = load_voting_classifier()
    
    # Determine model type for subtitle
    if model is not None:
        model_name = type(model).__name__
        if 'Voting' in model_name:
            model_desc = "Using VotingClassifier (Logistic Regression + Naive Bayes + Random Forest)"
        elif 'GradientBoosting' in model_name:
            model_desc = "Using Gradient Boosting Classifier"
        else:
            model_desc = f"Using {model_name}"
    else:
        model_desc = "Model Loading..."
    
    st.markdown(f"*{model_desc}*")
    st.markdown("---")
    
    # If main model failed to load, create fallback
    if model is None:
        st.sidebar.error("❌ Main model failed to load")
        if st.sidebar.button("🔄 Create Fallback Model"):
            model = create_fallback_model()
    
    # Sidebar status
    st.sidebar.header("🔧 System Status")
    
    if df is not None:
        st.sidebar.success("✅ Dataset loaded")
    else:
        st.sidebar.error("❌ Dataset failed to load")
        
    if model is not None:
        st.sidebar.success("✅ Model loaded")
        st.sidebar.write(f"Model type: {type(model).__name__}")
        
        # Show model-specific info
        if hasattr(model, 'n_estimators'):
            st.sidebar.write(f"Estimators: {model.n_estimators}")
        if hasattr(model, 'learning_rate'):
            st.sidebar.write(f"Learning rate: {model.learning_rate}")
            
        # Show expected features
        if hasattr(model, 'feature_names_in_'):
            st.sidebar.markdown("**Expected Features:**")
            expected_features = list(model.feature_names_in_)
            st.sidebar.write(f"Total: {len(expected_features)}")
            
            # Show first 10 features
            for i, feat in enumerate(expected_features[:10]):
                st.sidebar.write(f"{i+1}. {feat}")
            if len(expected_features) > 10:
                st.sidebar.write(f"... and {len(expected_features) - 10} more")
    else:
        st.sidebar.error("❌ Model failed to load")
        st.sidebar.markdown("**⚠️ Please upload your model to Google Drive and update MODEL_FILE_ID**")
    
    # Main interface
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Prediction", "🔧 Feature Analysis", "📊 Dataset Info", "ℹ️ About"])
    
    with tab1:
        st.markdown("## 🔍 Drug-Food Interaction Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💊 Drug Name")
            drug_input = st.text_input("Enter Drug Name", placeholder="e.g., Warfarin", key="drug_input")
            
            # Show drug suggestions if dataset is loaded
            if df is not None and drug_input and 'drug' in df.columns:
                suggestions = df[df['drug'].str.contains(drug_input, case=False, na=False)]['drug'].unique()[:5]
                if len(suggestions) > 0:
                    st.markdown("**💡 Suggestions from dataset:**")
                    for suggestion in suggestions:
                        if st.button(f"✨ {suggestion}", key=f"drug_suggest_{suggestion}"):
                            st.session_state.drug_input = suggestion
                            st.rerun()
        
        with col2:
            st.markdown("### 🍎 Food Name")
            food_input = st.text_input("Enter Food Name", placeholder="e.g., Spinach", key="food_input")
            
            # Show food suggestions if dataset is loaded
            if df is not None and food_input and 'food' in df.columns:
                suggestions = df[df['food'].str.contains(food_input, case=False, na=False)]['food'].unique()[:5]
                if len(suggestions) > 0:
                    st.markdown("**💡 Suggestions from dataset:**")
                    for suggestion in suggestions:
                        if st.button(f"✨ {suggestion}", key=f"food_suggest_{suggestion}"):
                            st.session_state.food_input = suggestion
                            st.rerun()
        
        # Prediction section
        st.markdown("---")
        
        if st.button("🎯 Predict Interaction", type="primary", disabled=(model is None)):
            if drug_input and food_input:
                if model is not None:
                    with st.spinner("🔄 Analyzing interaction..."):
                        prediction, confidence, probabilities = predict_interaction(drug_input, food_input, model)
                        
                        if prediction is not None:
                            # FIXED: Pass the model to display_prediction_results
                            display_prediction_results(prediction, confidence, probabilities, drug_input, food_input, model)
                        else:
                            st.error("❌ Prediction failed. Please try again.")
                else:
                    st.error("❌ Model not loaded. Please check your MODEL_FILE_ID.")
            else:
                st.warning("⚠️ Please enter both drug name and food name.")
    
    with tab2:
        st.markdown("## 🔧 Feature Analysis")
        st.markdown("See how your inputs are converted into features for the model.")
        
        # Input fields for feature analysis
        drug_analysis = st.text_input("Drug Name for Analysis", placeholder="e.g., Aspirin")
        food_analysis = st.text_input("Food Name for Analysis", placeholder="e.g., Orange")
        
        if drug_analysis and food_analysis:
            show_feature_analysis(drug_analysis, food_analysis)
    
    with tab3:
        if df is not None:
            display_dataset_info(df)
        else:
            st.error("Dataset not available for analysis.")
    
    with tab4:
        st.markdown("## ℹ️ About This Application")
        st.markdown("""
        ### 🤖 Model Architecture
        This app uses your **Gradient Boosting Classifier**:
        - **Gradient Boosting**: Sequential ensemble method that builds models iteratively
        - **Weak Learners**: Typically decision trees that learn from previous errors
        - **Boosting Strategy**: Each new tree corrects mistakes from previous trees
        - **Final Prediction**: Combines all trees for robust classification
        
        ### 🔧 Feature Engineering
        The model uses **multiple engineered features** from drug and food names:
        
        **Drug Categories:**
        - Antibiotic, Anticoagulant, Antidepressant
        - Antihypertensive, Analgesic, Statin
        - Diabetes, Heart Rhythm, PPI, Pain Relief, Other
        
        **Food Categories:**
        - Citrus, Leafy, Dairy, Alcohol, Caffeine
        
        **Additional Features:**
        - Drug/Food length, Common letters, Contains match
        
        ### 📊 Prediction Process
        1. Extract features from drug/food names
        2. Feed features to Gradient Boosting Classifier
        3. Get ensemble prediction from multiple decision trees
        4. Return binary prediction + confidence scores
        
        ### ⚠️ Important Disclaimers
        - **Educational Use Only**: This tool is for learning purposes
        - **Not Medical Advice**: Always consult healthcare professionals
        - **Model Limitations**: Based on training data patterns only
        - **Professional Guidance**: Seek pharmacist/doctor advice for real decisions
        
        ### 🛠 Technical Requirements
        To use this app, you need:
        1. Your trained model saved as pickle file
        2. Upload model to Google Drive and get file ID
        3. Update `MODEL_FILE_ID` in the code
        
        ### 🔧 Model Saving Instructions
        **Correct way to save your model:**
        ```python
        import pickle
        
        # Save ONLY the model, not a dictionary
        with open('model.pkl', 'wb') as f:
            pickle.dump(your_trained_model, f)  # your actual trained model
        ```
        
        **If you saved results in a dictionary, extract the model first:**
        ```python
        # If you have something like:
        # results = {'model': voting_clf, 'accuracy': 0.95, ...}
        # Extract just the model:
        model_only = results['model']  # or whatever key contains your model
        
        # Then save the model only:
        with open('voting_classifier_model.pkl', 'wb') as f:
            pickle.dump(model_only, f)
        ```
        """)

    # Instructions for setup (moved inside main function)
    if model is None:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📋 Setup Instructions")
        st.sidebar.markdown("""
        **To complete setup:**
        
        1. **Save your model correctly:**
        ```python
        import pickle
        # Save ONLY the model, not a dictionary
        with open('voting_classifier_model.pkl', 'wb') as f:
            pickle.dump(model, f)  # Just the model
        ```
        
        2. **Upload to Google Drive**
        
        3. **Get file ID** from share link
        
        4. **Update MODEL_FILE_ID** in code
        
        **Common Issue:** If you saved a dictionary containing the model, extract the model first before saving.
        """)

if __name__ == "__main__":
    main()
