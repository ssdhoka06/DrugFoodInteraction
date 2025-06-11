
# XAI IMPLEMENTATION
print("\n🔍 PHASE 4: EXPLAINABLE AI (XAI) IMPLEMENTATION")
print("-" * 50)

class DrugFoodXAI:
    """Comprehensive XAI analysis for drug-food interactions"""
    
    def __init__(self, model, X_train, X_test, y_test, feature_names):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.shap_explainer = None
        self.lime_explainer = None
        
    def initialize_explainers(self):
        """Initialize SHAP and LIME explainers"""
        print("🔧 Initializing XAI explainers...")
        
        # SHAP explainer
        if hasattr(self.model, 'predict_proba'):
            try:
                # Use a subset for faster computation
                background_data = self.X_train.sample(min(100, len(self.X_train)))
                self.shap_explainer = shap.TreeExplainer(self.model, background_data)
                print("✅ SHAP TreeExplainer initialized")
            except:
                try:
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict_proba, 
                        self.X_train.sample(min(50, len(self.X_train)))
                    )
                    print("✅ SHAP KernelExplainer initialized")
                except Exception as e:
                    print(f"⚠️ SHAP initialization failed: {e}")
        
        # LIME explainer
        try:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_train.values,
                feature_names=self.feature_names,
                class_names=['No Interaction', 'Interaction'],
                mode='classification'
            )
            print("✅ LIME explainer initialized")
        except Exception as e:
            print(f"⚠️ LIME initialization failed: {e}")
    
    def explain_prediction(self, instance_idx, method='both'):
        """Explain a specific prediction"""
        instance = self.X_test.iloc[instance_idx:instance_idx+1]
        true_label = self.y_test.iloc[instance_idx]
        pred_label = self.model.predict(instance)[0]
        pred_proba = self.model.predict_proba(instance)[0, 1] if hasattr(self.model, 'predict_proba') else pred_label
        
        print(f"\n🔍 EXPLAINING PREDICTION #{instance_idx}")
        print(f"True Label: {true_label}, Predicted: {pred_label}, Probability: {pred_proba:.3f}")
        print("-" * 40)
        
        explanations = {}
        
        # SHAP explanation
        if method in ['shap', 'both'] and self.shap_explainer:
            try:
                shap_values = self.shap_explainer.shap_values(instance)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification
                
                explanations['shap'] = {
                    'values': shap_values[0],
                    'base_value': self.shap_explainer.expected_value[1] if isinstance(self.shap_explainer.expected_value, np.ndarray) else self.shap_explainer.expected_value,
                    'feature_names': self.feature_names
                }
                
                # Print SHAP values instead of plotting
                print("Top 5 SHAP feature contributions:")
                shap_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'shap_value': shap_values[0]
                }).sort_values('shap_value', key=abs, ascending=False)
                print(shap_df.head())
                
            except Exception as e:
                print(f"⚠️ SHAP explanation failed: {e}")
        
        # LIME explanation
        if method in ['lime', 'both'] and self.lime_explainer:
            try:
                lime_exp = self.lime_explainer.explain_instance(
                    instance.values[0],
                    self.model.predict_proba,
                    num_features=10
                )
                
                explanations['lime'] = lime_exp
                
                
                
            except Exception as e:
                print(f"⚠️ LIME explanation failed: {e}")
        
        return explanations
    
    def global_feature_importance(self):
        """Global feature importance analysis using SHAP"""
        print("\n📊 GLOBAL FEATURE IMPORTANCE ANALYSIS")
        print("-" * 40)
        
        if not self.shap_explainer:
            print("⚠️ SHAP explainer not available")
            return
        
        try:
            # Calculate SHAP values for test set (subset for performance)
            test_subset = self.X_test.sample(min(100, len(self.X_test)))
            shap_values = self.shap_explainer.shap_values(test_subset)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
            
            # Print feature importance instead of plotting
            mean_shap = np.abs(shap_values).mean(0)
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': mean_shap
            }).sort_values('importance', ascending=False)
            print("Top 10 Most Important Features:")
            print(importance_df.head(10))
            
            return shap_values
            
        except Exception as e:
            print(f"⚠️ Global analysis failed: {e}")
            return None
    
    def feature_interaction_analysis(self):
        """Analyze feature interactions"""
        print("\n🔗 FEATURE INTERACTION ANALYSIS")
        print("-" * 40)
        
        if not self.shap_explainer:
            print("⚠️ SHAP explainer not available")
            return
        
        try:
            # Get interaction values
            test_subset = self.X_test.sample(min(50, len(self.X_test)))
            interaction_values = self.shap_explainer.shap_interaction_values(test_subset)
            
            return interaction_values
            
        except Exception as e:
            print(f"⚠️ Interaction analysis failed: {e}")
            return None
    
    def decision_pathway_analysis(self, drug_name, food_name):
        """Trace decision pathway for specific drug-food pair"""
        print(f"\n🛤️ DECISION PATHWAY: {drug_name.upper()} + {food_name.upper()}")
        print("-" * 50)
        
        # Get prediction for this pair
        result = predict_new_interaction_with_explanation(drug_name, food_name)
        
        if 'error' in result:
            print(f"⚠️ Error in prediction: {result['error']}")
            return
        
        print(f"📋 Basic Information:")
        print(f"   Drug Category: {result['drug_category']}")
        print(f"   Food Category: {result['food_category']}")
        print(f"   Mechanism: {result['mechanism']}")
        print(f"   Risk Level: {result['risk_level']}")
        
        print(f"\n🤖 Model Decision:")
        print(f"   Prediction: {'INTERACTION' if result['interaction_predicted'] else 'NO INTERACTION'}")
        print(f"   Confidence: {result['probability']:.3f}")
        
        # Create a pathway visualization
        pathway_data = {
            'Input': [drug_name, food_name],
            'Categories': [result['drug_category'], result['food_category']],
            'Risk_Assessment': [result['risk_level']],
            'Mechanism': [result['mechanism']],
            'Final_Prediction': [f"{'INTERACTION' if result['interaction_predicted'] else 'NO INTERACTION'} ({result['probability']:.3f})"]
        }
        
        # Create decision tree visualization
        fig = go.Figure()
        
        # Add pathway nodes
        steps = ['Input', 'Categorization', 'Risk Assessment', 'Mechanism Analysis', 'Final Prediction']
        values = [
            f"{drug_name} + {food_name}",
            f"{result['drug_category']} + {result['food_category']}",
            result['risk_level'],
            result['mechanism'],
            f"{'INTERACTION' if result['interaction_predicted'] else 'NO INTERACTION'}"
        ]
        
        colors = ['lightblue', 'lightgreen', 'yellow', 'orange', 'red' if result['interaction_predicted'] else 'green']
        
        fig.add_trace(go.Scatter(
            x=list(range(len(steps))),
            y=[1]*len(steps),
            mode='markers+text',
            marker=dict(size=60, color=colors),
            text=values,
            textposition='middle center',
            textfont=dict(size=10),
            name='Decision Pathway'
        ))
        
        fig.update_layout(
            title=f"Decision Pathway: {drug_name} + {food_name}",
            xaxis=dict(tickmode='array', tickvals=list(range(len(steps))), ticktext=steps),
            yaxis=dict(visible=False),
            showlegend=False,
            height=400
        )
        
        return result

# Initialize XAI system
xai_system = DrugFoodXAI(
    model=models[results_df.iloc[0]['model']],
    X_train=X_train,
    X_test=X_test,
    y_test=y_test,
    feature_names=feature_info['feature_names']
)

xai_system.initialize_explainers()

def conduct_case_studies():
    """Conduct detailed case studies of high-risk interactions"""
    print("\n📚 CASE STUDY ANALYSIS")
    print("=" * 50)
    
    # Define case studies
    case_studies = [
        {
            'name': 'Warfarin-Spinach Interaction',
            'drug': 'warfarin',
            'food': 'spinach',
            'description': 'Classic vitamin K antagonist interaction'
        },
        {
            'name': 'Statin-Grapefruit Interaction',
            'drug': 'simvastatin',
            'food': 'grapefruit',
            'description': 'CYP3A4 enzyme inhibition leading to toxicity'
        },
        {
            'name': 'Antibiotic-Dairy Interaction',
            'drug': 'tetracycline',
            'food': 'milk',
            'description': 'Calcium chelation reducing absorption'
        }
    ]
    
    for i, case in enumerate(case_studies):
        print(f"\n📖 CASE STUDY {i+1}: {case['name']}")
        print("-" * 40)
        print(f"Description: {case['description']}")
        
        # Get detailed analysis
        result = xai_system.decision_pathway_analysis(case['drug'], case['food'])
        
        # Get educational insights
        insights = get_educational_insights(case['drug'], case['food'])
        print(f"\n💡 Educational Insight:")
        print(f"   {insights['patient_explanation']}")
        print(f"   Technical: {insights['professional_details']}")
        
        # Find similar cases in test set
        similar_cases = find_similar_interactions(case['drug'], case['food'])
        if similar_cases:
            print(f"   Similar cases found: {len(similar_cases)}")
        
        print("\n" + "="*50)

def find_similar_interactions(target_drug, target_food, top_n=5):
    """Find similar interactions in the dataset"""
    target_drug_cat = categorize_entity(target_drug, drug_categories)
    target_food_cat = categorize_entity(target_food, food_categories)
    
    # Find interactions with same categories
    similar = df_final[
        (df_final['drug_category'] == target_drug_cat) & 
        (df_final['food_category'] == target_food_cat)
    ].head(top_n)
    
    return similar[['drug', 'food', 'interaction', 'risk_level']].to_dict('records')

def create_interactive_dashboard():
    """Create interactive explanation dashboard"""
    print("\n📊 CREATING INTERACTIVE EXPLANATION DASHBOARD")
    print("-" * 50)
    
    # Sample predictions for dashboard
    sample_predictions = []
    for drug, food in [('warfarin', 'spinach'), ('aspirin', 'alcohol'), ('metformin', 'banana')]:
        result = predict_new_interaction_with_explanation(drug, food)
        sample_predictions.append(result)
    
    # Create dashboard with multiple visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Risk Distribution', 'Model Confidence', 'Category Interactions', 'Prediction Timeline'),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "heatmap"}, {"type": "scatter"}]]
    )
    
    # Risk distribution pie chart
    risk_counts = df_final['risk_level'].value_counts()
    fig.add_trace(
        go.Pie(labels=risk_counts.index, values=risk_counts.values, name="Risk Distribution"),
        row=1, col=1
    )
    
    # Model confidence bar chart
    confidences = [pred['probability'] for pred in sample_predictions]
    labels = [f"{pred['drug']}-{pred['food']}" for pred in sample_predictions]
    fig.add_trace(
        go.Bar(x=labels, y=confidences, name="Model Confidence"),
        row=1, col=2
    )
    
    # Category interaction heatmap
    interaction_matrix = pd.crosstab(df_final['drug_category'], df_final['food_category'], df_final['interaction'], aggfunc='mean')
    fig.add_trace(
        go.Heatmap(z=interaction_matrix.values, x=interaction_matrix.columns, y=interaction_matrix.index),
        row=2, col=1
    )
    
    # Prediction timeline (if we had timestamps)
    fig.add_trace(
        go.Scatter(x=list(range(len(sample_predictions))), y=confidences, mode='lines+markers', name="Predictions"),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Drug-Food Interaction XAI Dashboard")

def explain_model_behavior():
    """Explain overall model behavior patterns"""
    print("\n🧠 MODEL BEHAVIOR ANALYSIS")
    print("-" * 40)
    
    # Analyze model predictions by categories
    behavior_analysis = df_final.groupby(['drug_category', 'food_category']).agg({
        'interaction': ['count', 'mean'],
        'risk_level': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'
    }).round(3)
    
    print("Model Decision Patterns by Category Combinations:")
    print(behavior_analysis.head(10))
    
    # Identify model biases
    print("\n⚖️ Potential Model Biases:")
    high_confidence_wrong = 0  # This would need actual implementation with confidence thresholds
    
    # Category bias analysis
    category_bias = df_final.groupby('drug_category')['interaction'].mean().sort_values(ascending=False)
    print(f"Drug categories with highest interaction rates:")
    print(category_bias.head())
    
    food_bias = df_final.groupby('food_category')['interaction'].mean().sort_values(ascending=False)
    print(f"\nFood categories with highest interaction rates:")
    print(food_bias.head())

# Create ensemble model with top performers
print("\n🏆 FINAL MODEL RANKINGS")
print("=" * 40)

# Show final results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('f1', ascending=False)
print("Final Model Performance Rankings:")
print(results_df[['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']].round(4))

print(f"\n🥇 Best Model: {results_df.iloc[0]['model']} (F1: {results_df.iloc[0]['f1']:.4f})")

# Risk Analysis
print("\n⚠️ RISK LEVEL ANALYSIS")
print("=" * 40)

# Analyze predictions by risk level
risk_analysis = df_final.groupby('risk_level').agg({
    'interaction': ['count', 'sum', 'mean']
}).round(3)

risk_analysis.columns = ['Total_Samples', 'Positive_Interactions', 'Interaction_Rate']
print("Risk Level Distribution:")
print(risk_analysis)

# Plot risk level distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
risk_counts = df_final['risk_level'].value_counts()
plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Risk Levels')

plt.subplot(1, 2, 2)
risk_interaction_rates = df_final.groupby('risk_level')['interaction'].mean()
colors = ['green', 'orange', 'red']
plt.bar(risk_interaction_rates.index, risk_interaction_rates.values, color=colors)
plt.title('Interaction Rate by Risk Level')
plt.xlabel('Risk Level')
plt.ylabel('Interaction Rate')
plt.ylim(0, 1)

# Add value labels on bars
for i, v in enumerate(risk_interaction_rates.values):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Feature importance analysis for best model
print("\n📈 FEATURE IMPORTANCE ANALYSIS")
print("=" * 40)

best_model_name = results_df.iloc[0]['model']
if best_model_name in models:
    best_model = models[best_model_name]
    
    try:
        # Get feature importance
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
        elif hasattr(best_model, 'coef_'):
            importances = np.abs(best_model.coef_[0])
        else:
            print(f"Feature importance not available for {best_model_name}")
            importances = None
        
        if importances is not None:
            # Create feature importance DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': feature_info['feature_names'],
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Plot top 20 features
            top_features = feature_importance_df.head(20)
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 20 Feature Importances - {best_model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
            print("Top 10 Most Important Features:")
            print(feature_importance_df.head(10)[['feature', 'importance']].round(4))
    
    except Exception as e:
        print(f"Error analyzing feature importance: {str(e)}")

# Enhanced prediction function
def predict_new_interaction_with_explanation(drug_name, food_name, explain=True):
    """Enhanced prediction with XAI explanations"""
    # Get base prediction
    result = predict_new_interaction(drug_name, food_name)
    
    if explain and 'error' not in result:
        # Add explanation
        try:
            # Create temporary instance for explanation
            temp_df = pd.DataFrame({
                'drug': [drug_name.lower()],
                'food': [food_name.lower()],
                'interaction': [0]
            })
            
            # Apply preprocessing
            temp_df['drug_category'] = temp_df['drug'].apply(lambda x: categorize_entity(x, drug_categories))
            temp_df['food_category'] = temp_df['food'].apply(lambda x: categorize_entity(x, food_categories))
            
            # Get decision pathway
            pathway = xai_system.decision_pathway_analysis(drug_name, food_name)
            
            result['explanation'] = {
                'decision_pathway': pathway,
                'key_factors': f"Categories: {result['drug_category']} + {result['food_category']}",
                'confidence_level': 'High' if result['probability'] > 0.7 else 'Medium' if result['probability'] > 0.4 else 'Low'
            }
            
        except Exception as e:
            result['explanation'] = f"Explanation failed: {e}"
    
    return result

def predict_new_interaction(drug_name, food_name, model=None, return_risk=True):
    """Predict interaction for new drug-food pair"""
    
    if model is None:
        model = models[results_df.iloc[0]['model']]  # Use best model
    
    # Create a temporary dataframe for the new pair
    new_df = pd.DataFrame({
        'drug': [drug_name.lower().strip()],
        'food': [food_name.lower().strip()],
        'interaction': [0]  # Placeholder
    })
    
    # Apply same preprocessing
    new_df['drug_category'] = new_df['drug'].apply(lambda x: categorize_entity(x, drug_categories))
    new_df['food_category'] = new_df['food'].apply(lambda x: categorize_entity(x, food_categories))
    new_df[['mechanism', 'risk_level']] = new_df.apply(
        lambda x: pd.Series(get_interaction_details(x['drug_category'], x['food_category'])), 
        axis=1
    )
    
    # Create features
    try:
        X_new, _ = create_enhanced_features(new_df)
        # Ensure all required columns exist and same order
        missing_cols = set(feature_info['feature_names']) - set(X_new.columns)
        for col in missing_cols:
            X_new[col] = 0
            
        # Ensure same order as training data
        X_new = X_new[feature_info['feature_names']]
        
        # Scale features if needed
        if model.__class__.__name__ == 'MLPClassifier':
            X_new = scaler.transform(X_new)
        
        # Make prediction
        prediction = model.predict(X_new)[0]
        
        try:
            probability = model.predict_proba(X_new)[0, 1]
        except:
            probability = prediction
        
        result = {
            'drug': drug_name,
            'food': food_name,
            'interaction_predicted': bool(prediction),
            'probability': float(probability),
            'drug_category': new_df['drug_category'].iloc[0],
            'food_category': new_df['food_category'].iloc[0],
            'mechanism': new_df['mechanism'].iloc[0],
            'risk_level': new_df['risk_level'].iloc[0]
        }
        
        return result
        
    except Exception as e:
        return {
            'drug': drug_name,
            'food': food_name,
            'error': str(e),
            'interaction_predicted': None,
            'probability': None
        }

def get_personalized_warning(drug_name, food_name, age=None, gender=None, conditions=None):
    """Generate personalized warnings based on patient factors"""
    base_result = predict_new_interaction_with_explanation(drug_name, food_name)
    
    # Adjust risk based on patient factors
    risk_multiplier = 1.0
    if age and age > 65:
        risk_multiplier += 0.2  # Higher risk for elderly
    if conditions and 'liver_disease' in conditions:
        risk_multiplier += 0.3
    if conditions and 'kidney_disease' in conditions:
        risk_multiplier += 0.2
    
    adjusted_probability = min(base_result['probability'] * risk_multiplier, 1.0)
    
    return {
        **base_result,
        'adjusted_probability': adjusted_probability,
        'personalized_warning': f"Risk adjusted for age: {age}, conditions: {conditions}"
    }

def check_meal_plan_compatibility(medications, meal_plan):
    """Check if meal plan is compatible with medications"""
    interactions_found = []
    
    for drug in medications:
        for food in meal_plan:
            result = predict_new_interaction_with_explanation(drug, food)
            if result['interaction_predicted'] and result['probability'] > 0.5:
                interactions_found.append(result)
    
    return {
        'safe': len(interactions_found) == 0,
        'interactions': interactions_found,
        'recommendations': f"Found {len(interactions_found)} potential interactions"
    }

def get_educational_insights(drug_name, food_name):
    """Provide educational explanations"""
    result = predict_new_interaction_with_explanation(drug_name, food_name)
    
    mechanism_explanations = {
        'cyp3a4_inhibition': "This food blocks liver enzymes that break down the medication, potentially causing dangerous buildup.",
        'calcium_chelation': "Calcium in this food binds to the medication, reducing absorption.",
        'vitamin_k_competition': "This food contains vitamin K which can interfere with blood-thinning effects.",
        'absorption_interference': "This food can slow down or reduce medication absorption in the stomach."
    }
    
    explanation = mechanism_explanations.get(result['mechanism'], "The interaction mechanism is not well understood.")
    
    return {
        **result,
        'patient_explanation': explanation,
        'professional_details': f"Mechanism: {result['mechanism']}, Category interaction: {result['drug_category']} + {result['food_category']}"
    }

# Example predictions
print("\n🔮 EXAMPLE PREDICTIONS")
print("=" * 40)

test_pairs = [
    ('warfarin', 'spinach'),      # Known high-risk interaction
    ('simvastatin', 'grapefruit'), # Known high-risk interaction
    ('aspirin', 'alcohol'),       # Known moderate-risk interaction
    ('amoxicillin', 'milk'),      # Known moderate-risk interaction
    ('metformin', 'banana'),      # Likely low-risk
    ('ibuprofen', 'coffee')       # Likely low-risk
]

for drug, food in test_pairs:
    result = predict_new_interaction_with_explanation(drug, food)
    
    if 'error' not in result:
        print(f"\n{drug.title()} + {food.title()}:")
        print(f"  Interaction: {'YES' if result['interaction_predicted'] else 'NO'}")
        print(f"  Probability: {result['probability']:.3f}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Mechanism: {result['mechanism']}")
        print(f"  Categories: {result['drug_category']} + {result['food_category']}")
    else:
        print(f"\n{drug.title()} + {food.title()}: Error - {result['error']}")

# EXECUTE XAI ANALYSIS
print("\n🔍 EXECUTING COMPREHENSIVE XAI ANALYSIS")
print("=" * 60)

## Simple test without visualizations
def test_xai_simple():
    print("\n🔍 SIMPLE XAI TEST")
    print("-" * 30)
    
    # Test a few predictions
    test_pairs = [('warfarin', 'spinach'), ('aspirin', 'coffee')]
    
    for drug, food in test_pairs:
        result = predict_new_interaction_with_explanation(drug, food, explain=False)
        if 'error' not in result:
            print(f"{drug} + {food}: {'INTERACTION' if result['interaction_predicted'] else 'NO INTERACTION'} (p={result['probability']:.3f})")

# Run simple test
test_xai_simple()
