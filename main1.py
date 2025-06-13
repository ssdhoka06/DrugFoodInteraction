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
MODEL_FILE_ID = "1XQHlS8d8Rz3DYafdjJ3maE5XkWvivd8s"  # ⚠️ REPLACE WITH YOUR PICKLE FILE ID

# Your exact feature engineering function from main_copy.py
def create_features(drug, food):
    """Create features exactly as in your original code"""
    features = {
        'drug_length': len(drug),
        'food_length': len(food),
        'common_letters': len(set(drug.lower()) & set(food.lower())),
        'drug_starts_with_same_letter': int(drug[0].lower() == food[0].lower()),
        'drug_ends_with_same_letter': int(drug[-1].lower() == food[-1].lower()),
        'food_contains_drug': int(drug.lower() in food.lower()),
        'drug_contains_food': int(food.lower() in drug.lower())
    }
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
    """Load your pre-trained VotingClassifier from Google Drive"""
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
            voting_clf = pickle.load(f)
        return voting_clf
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def predict_interaction(drug_input, food_input, voting_clf):
    """Make prediction using your exact pipeline"""
    try:
        # Use your exact feature engineering
        features = create_features(drug_input, food_input)
        
        # Make prediction using your VotingClassifier
        prediction = voting_clf.predict(features)[0]
        
        # Get prediction probabilities (since you use voting='soft')
        probabilities = voting_clf.predict_proba(features)[0]
        confidence = max(probabilities)
        
        return prediction, confidence, probabilities
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

def display_prediction_results(prediction, confidence, probabilities, drug_input, food_input):
    """Display prediction results with your model's output"""
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
    
    # Model breakdown (since you use VotingClassifier)
    with st.expander("🔍 Model Breakdown (VotingClassifier Details)"):
        st.markdown("""
        **Your model combines 3 classifiers:**
        - 🔵 **Logistic Regression (lr)**
        - 🟢 **Naive Bayes (nb)** 
        - 🟣 **Random Forest (rf)**
        
        The final prediction uses soft voting (probability averaging) across all three models.
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
        
        with col2:
            st.markdown(f"""
            **Calculated Features:**
            - Drug Length: `{len(drug_input)}`
            - Food Length: `{len(food_input)}`
            - Common Letters: `{len(set(drug_input.lower()) & set(food_input.lower()))}`
            - Same Start Letter: `{drug_input[0].lower() == food_input[0].lower()}`
            - Same End Letter: `{drug_input[-1].lower() == food_input[-1].lower()}`
            - Food Contains Drug: `{drug_input.lower() in food_input.lower()}`
            - Drug Contains Food: `{food_input.lower() in drug_input.lower()}`
            """)

def main():
    """Main Streamlit app"""
    st.title("💊 Drug-Food Interaction Predictor")
    st.markdown("*Using VotingClassifier (Logistic Regression + Naive Bayes + Random Forest)*")
    st.markdown("---")
    
    # Load data and model
    df = load_data()
    voting_clf = load_voting_classifier()
    
    # Sidebar status
    st.sidebar.header("🔧 System Status")
    
    if df is not None:
        st.sidebar.success("✅ Dataset loaded")
    else:
        st.sidebar.error("❌ Dataset failed to load")
        
    if voting_clf is not None:
        st.sidebar.success("✅ VotingClassifier loaded")
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
                            st.experimental_rerun()
        
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
                            st.experimental_rerun()
        
        # Prediction section
        st.markdown("---")
        
        if st.button("🎯 Predict Interaction", type="primary", disabled=(voting_clf is None)):
            if drug_input and food_input:
                if voting_clf is not None:
                    with st.spinner("🔄 Analyzing interaction using VotingClassifier..."):
                        prediction, confidence, probabilities = predict_interaction(drug_input, food_input, voting_clf)
                        
                        if prediction is not None:
                            display_prediction_results(prediction, confidence, probabilities, drug_input, food_input)
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
        This app uses your **VotingClassifier** that combines:
        - **Logistic Regression (lr)**: Linear classification
        - **Naive Bayes (nb)**: Probabilistic classification  
        - **Random Forest (rf)**: Ensemble tree-based classification
        
        ### 🔧 Feature Engineering
        The model uses **7 handcrafted features** from drug and food names:
        1. **Drug Length**: Character count of drug name
        2. **Food Length**: Character count of food name  
        3. **Common Letters**: Number of shared letters
        4. **Same Start Letter**: Whether first letters match
        5. **Same End Letter**: Whether last letters match
        6. **Food Contains Drug**: Whether drug name appears in food name
        7. **Drug Contains Food**: Whether food name appears in drug name
        
        ### 📊 Prediction Process
        1. Extract features from drug/food names
        2. Feed features to VotingClassifier
        3. Get soft voting result (probability averaging)
        4. Return binary prediction + confidence scores
        
        ### ⚠️ Important Disclaimers
        - **Educational Use Only**: This tool is for learning purposes
        - **Not Medical Advice**: Always consult healthcare professionals
        - **Model Limitations**: Based on training data patterns only
        - **Professional Guidance**: Seek pharmacist/doctor advice for real decisions
        
        ### 🛠 Technical Requirements
        To use this app, you need:
        1. Your trained `voting_clf` model saved as pickle file
        2. Upload model to Google Drive and get file ID
        3. Update `MODEL_FILE_ID` in the code
        """)

# Instructions for setup
if voting_clf is None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Setup Instructions")
    st.sidebar.markdown("""
    **To complete setup:**
    
    1. **Save your model:**
    ```python
    import pickle
    with open('voting_classifier_model.pkl', 'wb') as f:
        pickle.dump(voting_clf, f)
    ```
    
    2. **Upload to Google Drive**
    
    3. **Get file ID** from share link
    
    4. **Update MODEL_FILE_ID** in code
    """)

if __name__ == "__main__":
    main()
