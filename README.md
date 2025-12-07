Drug-Food Interaction Predictor
Machine learning system for predicting drug-food interactions with explainable AI. Identifies potentially harmful medication-food combinations and provides risk assessments with mechanistic explanations.

Features
8 ML Models: LightGBM, XGBoost, CatBoost, Random Forest, Extra Trees, Gradient Boosting, MLP, Voting Ensemble
Risk Classification: Automatic HIGH/MODERATE/LOW categorization
Explainable AI: SHAP and LIME analysis for model interpretability
Web Interface: Interactive dashboard with real-time search
REST API: JSON endpoints for programmatic access

Usage
Web Interface
Select medication from dropdown
Select food item
Click "Analyze Interaction"
View risk level, mechanism, and recommendations

Quick Start
# Clone repository
git clone https://github.com/yourusername/drug-food-interaction-predictor.git
cd drug-food-interaction-predictor

# Install dependencies
pip install flask pandas numpy scikit-learn matplotlib seaborn
pip install lightgbm xgboost catboost shap lime joblib

# Train model (optional - pre-trained model included)
python main.py

# Run web application
python app.py



Disclaimer
⚠️ This is a research prototype for educational purposes only. Not validated for clinical use. Always consult healthcare professionals before making medication decisions.
