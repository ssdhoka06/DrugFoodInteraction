# Add this to your Streamlit app to understand what your model expects

def analyze_model_requirements(model):
    """Analyze what features the model expects"""
    st.markdown("## 🔍 Model Analysis")
    
    # Check if model has feature names
    if hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
        st.write(f"**Total expected features:** {len(expected_features)}")
        
        # Categorize features
        tfidf_features = [f for f in expected_features if 'tfidf' in f.lower()]
        drug_features = [f for f in expected_features if f.startswith('drug_')]
        food_features = [f for f in expected_features if f.startswith('food_')]
        mechanism_features = [f for f in expected_features if f.startswith('mechanism_')]
        risk_features = [f for f in expected_features if f.startswith('risk_')]
        other_features = [f for f in expected_features if not any(x in f for x in ['tfidf', 'drug_', 'food_', 'mechanism_', 'risk_'])]
        
        st.write(f"**TF-IDF features:** {len(tfidf_features)}")
        st.write(f"**Drug category features:** {len(drug_features)}")
        st.write(f"**Food category features:** {len(food_features)}")
        st.write(f"**Mechanism features:** {len(mechanism_features)}")
        st.write(f"**Risk features:** {len(risk_features)}")
        st.write(f"**Other features:** {len(other_features)}")
        
        # Show actual feature names
        with st.expander("📋 All Expected Feature Names"):
            for i, feature in enumerate(expected_features):
                st.write(f"{i+1:3d}. {feature}")
        
        with st.expander("🏷 Feature Categories"):
            if drug_features:
                st.write("**Drug Categories:**", [f.replace('drug_', '') for f in drug_features])
            if food_features:
                st.write("**Food Categories:**", [f.replace('food_', '') for f in food_features])
            if mechanism_features:
                st.write("**Mechanisms:**", [f.replace('mechanism_', '') for f in mechanism_features])
            if risk_features:
                st.write("**Risk Levels:**", [f.replace('risk_', '') for f in risk_features])
            if other_features:
                st.write("**Other Features:**", other_features)
        
        return expected_features
    else:
        st.error("❌ Model doesn't have feature_names_in_ attribute")
        return None

def create_minimal_features_from_csv(drug_input, food_input, df, expected_features):
    """Create features by learning from your CSV data"""
    
    # Initialize all features with 0
    features = pd.DataFrame(0, index=[0], columns=expected_features)
    
    # Basic features that we can calculate
    features['drug_length'] = len(drug_input) if 'drug_length' in expected_features else None
    features['food_length'] = len(food_input) if 'food_length' in expected_features else None
    
    # Try to infer categories from CSV data
    if df is not None:
        # Look for similar drugs in the dataset
        drug_matches = df[df['drug'].str.contains(drug_input, case=False, na=False)]
        food_matches = df[df['food'].str.contains(food_input, case=False, na=False)]
        
        if not drug_matches.empty:
            st.info(f"Found {len(drug_matches)} similar drugs in dataset")
            # Try to infer drug category from dataset patterns
            
        if not food_matches.empty:
            st.info(f"Found {len(food_matches)} similar foods in dataset")
            # Try to infer food category from dataset patterns
    
    # Simple string similarity
    if 'name_similarity' in expected_features:
        from difflib import SequenceMatcher
        features['name_similarity'] = SequenceMatcher(None, drug_input.lower(), food_input.lower()).ratio()
    
    # Basic categorical guessing (you can expand this)
    drug_lower = drug_input.lower()
    food_lower = food_input.lower()
    
    # Drug categories (expand based on what your model expects)
    if 'drug_anticoagulant' in expected_features:
        features['drug_anticoagulant'] = int(any(x in drug_lower for x in ['warfarin', 'coumadin', 'heparin']))
    
    if 'drug_analgesic' in expected_features:
        features['drug_analgesic'] = int(any(x in drug_lower for x in ['fentanyl', 'morphine', 'codeine', 'tramadol']))
    
    # Food categories
    if 'food_citrus' in expected_features:
        features['food_citrus'] = int(any(x in food_lower for x in ['grapefruit', 'grape fruit', 'orange', 'lemon']))
    
    if 'food_leafy_greens' in expected_features:
        features['food_leafy_greens'] = int(any(x in food_lower for x in ['spinach', 'kale', 'lettuce']))
    
    # Fill in TF-IDF features with zeros (not ideal, but better than nothing)
    tfidf_columns = [col for col in expected_features if 'tfidf' in col.lower()]
    for col in tfidf_columns:
        features[col] = 0.0
    
    return features

# Add this to your main app
def debug_tab():
    """Add a debug tab to understand your model"""
    st.markdown("## 🔧 Model Debug")
    
    if model is not None:
        expected_features = analyze_model_requirements(model)
        
        if expected_features:
            st.markdown("### 🧪 Test Feature Creation")
            test_drug = st.text_input("Test Drug:", "fentanyl")
            test_food = st.text_input("Test Food:", "grapefruit")
            
            if st.button("🔍 Analyze Features"):
                test_features = create_minimal_features_from_csv(test_drug, test_food, df, expected_features)
                
                st.markdown("**Generated Features (non-zero only):**")
                non_zero = test_features.loc[0][test_features.loc[0] != 0]
                for feature, value in non_zero.items():
                    st.write(f"• {feature}: {value}")
                
                # Try prediction with these features
                try:
                    pred = model.predict(test_features)[0]
                    probs = model.predict_proba(test_features)[0]
                    st.write(f"**Prediction:** {pred}")
                    st.write(f"**Probabilities:** {probs}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
    else:
        st.error("Model not loaded")
