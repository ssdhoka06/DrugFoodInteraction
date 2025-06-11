import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from datetime import datetime

# Load the model package
@st.cache_resource
def load_model():
    with open('best_drug_food_interaction_model.pkl', 'rb') as f:
        model_package = pickle.load(f)
    return model_package

model_package = load_model()
xai_system = model_package['xai_system']
predict_new_interaction_with_explanation = globals()['predict_new_interaction_with_explanation']
get_educational_insights = globals()['get_educational_insights']
conduct_case_studies = globals()['conduct_case_studies']

# Streamlit UI
st.set_page_config(page_title="Drug-Food Interaction Predictor", layout="wide")
st.title("🔬 Drug-Food Interaction Predictor")
st.markdown("Analyze potential interactions between medications and foods with explainable AI insights.")

# Sidebar for input
st.sidebar.header("Input Drug and Food")
drug_input = st.sidebar.text_input("Enter Drug Name", value="warfarin")
food_input = st.sidebar.text_input("Enter Food Name", value="spinach")
predict_button = st.sidebar.button("Predict Interaction")

# Main content
if predict_button:
    with st.spinner("Analyzing interaction..."):
        result = predict_new_interaction_with_explanation(drug_input, food_input)
        
        if 'error' not in result:
            st.header(f"{drug_input.title()} + {food_input.title()} Interaction Analysis")
            
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Interaction", "YES" if result['interaction_predicted'] else "NO")
            with col2:
                st.metric("Confidence", f"{result['probability']:.3f}")
            with col3:
                st.metric("Risk Level", result['risk_level'])
            
            # Basic Information
            st.subheader("📋 Basic Information")
            st.write(f"**Drug Category**: {result['drug_category']}")
            st.write(f"**Food Category**: {result['food_category']}")
            st.write(f"**Mechanism**: {result['mechanism']}")
            
            # Educational Insights
            st.subheader("💡 Educational Insights")
            insights = get_educational_insights(drug_input, food_input)
            st.write(f"**Patient Explanation**: {insights['patient_explanation']}")
            st.write(f"**Technical Details**: {insights['professional_details']}")
            
            # XAI Explanations
            st.subheader("🔍 Explainable AI Insights")
            if 'explanation' in result and isinstance(result['explanation'], dict):
                pathway = result['explanation']['decision_pathway']
                st.write("**Decision Pathway**")
                st.write(f"- Prediction: {pathway['prediction']}")
                st.write(f"- Confidence: {pathway['confidence']}")
                st.write(f"- Drug Category: {pathway['drug_category']}")
                st.write(f"- Food Category: {pathway['food_category']}")
                st.write(f"- Mechanism: {pathway['mechanism']}")
                st.write(f"- Risk Level: {pathway['risk_level']}")
                
                # SHAP Feature Importance (for the first test instance, as an example)
                explanation = xai_system.explain_prediction(0)
                if 'shap' in explanation['explanations'] and isinstance(explanation['explanations']['shap'], pd.DataFrame):
                    st.write("**Top SHAP Features**")
                    st.dataframe(explanation['explanations']['shap'])
                    fig = px.bar(
                        explanation['explanations']['shap'],
                        x='SHAP Value',
                        y='Feature',
                        orientation='h',
                        title="Top SHAP Feature Contributions"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error(f"Prediction failed: {result['error']}")

# Case Studies Section
st.header("📚 Case Studies")
case_studies = conduct_case_studies()
for case in case_studies:
    with st.expander(f"Case Study: {case['case_name']}"):
        st.write(f"**Description**: {case['description']}")
        st.write(f"**Drug**: {case['drug']}")
        st.write(f"**Food**: {case['food']}")
        st.write(f"**Prediction**: {case['prediction']}")
        st.write(f"**Confidence**: {case['confidence']}")
        st.write(f"**Drug Category**: {case['drug_category']}")
        st.write(f"**Food Category**: {case['food_category']}")
        st.write(f"**Mechanism**: {case['mechanism']}")
        st.write(f"**Risk Level**: {case['risk_level']}")

# Global Feature Importance
st.header("📊 Global Feature Importance")
global_importance = xai_system.global_feature_importance()
if isinstance(global_importance, pd.DataFrame):
    st.dataframe(global_importance)
    fig = px.bar(
        global_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Top 10 Global Feature Importances"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning(global_importance)

# Footer
st.markdown("---")
st.markdown(f"Model Version: {model_package['model_version']} | Training Date: {model_package['training_date']}")
st.markdown("Developed by Sachidhoka | Powered by xAI")