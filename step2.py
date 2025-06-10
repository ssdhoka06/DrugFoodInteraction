import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                           precision_recall_curve, roc_auc_score, 
                           f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import json
import joblib
warnings.filterwarnings('ignore')

class TwoStageRiskClassifier:
    def __init__(self, random_state=42):
        """
        Two-stage classifier for handling severe class imbalance in drug-food interactions
        
        Stage 1: Risk vs No-Risk (Binary)
        Stage 2: Moderate vs High Risk (among risk-positive samples)
        """
        self.random_state = random_state
        self.stage1_model = None  # Risk vs No-Risk
        self.stage2_model = None  # Moderate vs High Risk
        self.scaler = StandardScaler()
        self.feature_selector = None
        
        # Training history
        self.stage1_history = {}
        self.stage2_history = {}
        self.validation_results = {}
        
    def analyze_class_distribution(self, y, stage_name=""):
        """Analyze and visualize class distribution"""
        print(f"\n📊 Class Distribution Analysis {stage_name}")
        print("-" * 50)
        
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        for cls, count in zip(unique, counts):
            percentage = (count / total) * 100
            print(f"Class {cls}: {count:,} samples ({percentage:.2f}%)")
        
        # Calculate imbalance ratio
        if len(unique) == 2:
            ratio = max(counts) / min(counts)
            print(f"Imbalance Ratio: {ratio:.1f}:1")
            
            if ratio > 10:
                print("⚠️  SEVERE IMBALANCE - Will use advanced resampling")
            elif ratio > 3:
                print("⚠️  MODERATE IMBALANCE - Will use basic resampling")
            else:
                print("✅ BALANCED - No resampling needed")
        
        return dict(zip(unique, counts))
    
    def prepare_two_stage_labels(self, df):
        """
        Prepare labels for two-stage classification
        
        Stage 1: Risk (1) vs No-Risk (0)
        Stage 2: Moderate (0) vs High (1) - only for risk samples
        """
        print("\n🎯 Preparing Two-Stage Labels")
        print("-" * 40)
        
        # Analyze original severity distribution
        severity_dist = self.analyze_class_distribution(df['severity_encoded'], "(Original)")
        
        # Stage 1: Binary Risk Classification
        # Combine moderate (1) + high (2) = Risk (1), low (0) = No-Risk (0)
        stage1_labels = (df['severity_encoded'] > 0).astype(int)
        
        print(f"\n🔄 Stage 1 Transformation:")
        print(f"   Low Risk (0) → No-Risk (0)")
        print(f"   Moderate (1) + High (2) → Risk (1)")
        
        stage1_dist = self.analyze_class_distribution(stage1_labels, "(Stage 1)")
        
        # Stage 2: Severity Classification (only for risk samples)
        risk_mask = df['severity_encoded'] > 0
        stage2_df = df[risk_mask].copy()
        
        if len(stage2_df) > 0:
            # Moderate (1) → 0, High (2) → 1
            stage2_labels = (stage2_df['severity_encoded'] == 2).astype(int)
            
            print(f"\n🔄 Stage 2 Transformation (Risk samples only):")
            print(f"   Moderate (1) → 0")
            print(f"   High (2) → 1")
            
            stage2_dist = self.analyze_class_distribution(stage2_labels, "(Stage 2)")
        else:
            stage2_labels = np.array([])
            stage2_dist = {}
        
        return stage1_labels, stage2_labels, stage2_df, {
            'original': severity_dist,
            'stage1': stage1_dist,
            'stage2': stage2_dist
        }
    
    def select_optimal_resampler(self, X, y, stage_name):
        """
        Select optimal resampling strategy based on data characteristics
        """
        print(f"\n🔧 Selecting Optimal Resampler for {stage_name}")
        
        # Analyze class distribution
        class_counts = Counter(y)
        minority_class = min(class_counts.values())
        majority_class = max(class_counts.values())
        imbalance_ratio = majority_class / minority_class
        
        print(f"Imbalance ratio: {imbalance_ratio:.1f}:1")
        print(f"Minority class size: {minority_class}")
        print(f"Feature dimensionality: {X.shape[1]}")
        
        # Strategy selection based on characteristics
        if minority_class < 50:
            # Very small minority class - use ADASYN or BorderlineSMOTE
            if X.shape[1] > 100:  # High dimensional
                resampler = ADASYN(random_state=self.random_state, n_neighbors=3)
                strategy_name = "ADASYN (adaptive, few neighbors)"
            else:
                resampler = BorderlineSMOTE(random_state=self.random_state, k_neighbors=3)
                strategy_name = "BorderlineSMOTE (focus on borderline cases)"
        
        elif imbalance_ratio > 20:
            # Severe imbalance - use combined approach
            resampler = SMOTETomek(random_state=self.random_state)
            strategy_name = "SMOTETomek (oversample + clean)"
        
        elif imbalance_ratio > 5:
            # Moderate imbalance - use standard SMOTE
            resampler = SMOTE(random_state=self.random_state)
            strategy_name = "SMOTE (synthetic oversampling)"
        
        else:
            # Mild imbalance - no resampling
            resampler = None
            strategy_name = "No resampling (balanced enough)"
        
        print(f"✅ Selected: {strategy_name}")
        return resampler
    
    def train_stage1_models(self, X, y):
        """
        Train Stage 1: Risk vs No-Risk classification
        """
        print("\n🎯 STAGE 1: Risk vs No-Risk Classification")
        print("=" * 50)
        
        # Select resampler
        resampler = self.select_optimal_resampler(X, y, "Stage 1")
        
        # Define candidate models with class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        weight_dict = {i: w for i, w in enumerate(class_weights)}
        
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight=weight_dict,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                random_state=self.random_state
            ),
            'LogisticRegression': LogisticRegression(
                class_weight=weight_dict,
                random_state=self.random_state,
                max_iter=1000,
                solver='liblinear'
            ),
            'SVM': SVC(
                class_weight=weight_dict,
                random_state=self.random_state,
                probability=True,
                kernel='rbf'
            )
        }
        
        # Evaluate models with cross-validation
        print("\n🔄 Model Evaluation with Cross-Validation")
        best_model = None
        best_score = 0
        results = {}
        
        for name, model in models.items():
            print(f"\nTesting {name}...")
            
            # Create pipeline with resampling
            if resampler is not None:
                pipeline = ImbPipeline([
                    ('resampler', resampler),
                    ('classifier', model)
                ])
            else:
                pipeline = model
            
            # Stratified K-Fold cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            # Multiple metrics
            f1_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1', n_jobs=-1)
            precision_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='precision', n_jobs=-1)
            recall_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='recall', n_jobs=-1)
            
            # Store results
            results[name] = {
                'f1_mean': f1_scores.mean(),
                'f1_std': f1_scores.std(),
                'precision_mean': precision_scores.mean(),
                'recall_mean': recall_scores.mean(),
                'pipeline': pipeline
            }
            
            print(f"   F1: {f1_scores.mean():.4f} (±{f1_scores.std():.4f})")
            print(f"   Precision: {precision_scores.mean():.4f}")
            print(f"   Recall: {recall_scores.mean():.4f}")
            
            # Select best model (prioritize F1 for imbalanced data)
            if f1_scores.mean() > best_score:
                best_score = f1_scores.mean()
                best_model = name
        
        print(f"\n✅ Best Stage 1 Model: {best_model} (F1: {best_score:.4f})")
        
        # Train final model on full dataset
        self.stage1_model = results[best_model]['pipeline']
        self.stage1_model.fit(X, y)
        self.stage1_history = results
        
        return self.stage1_model, results
    
    def train_stage2_models(self, X, y):
        """
        Train Stage 2: Moderate vs High Risk classification
        """
        print("\n🎯 STAGE 2: Moderate vs High Risk Classification")
        print("=" * 50)
        
        if len(X) == 0 or len(np.unique(y)) < 2:
            print("⚠️ Insufficient data for Stage 2 training")
            return None, {}
        
        # Select resampler
        resampler = self.select_optimal_resampler(X, y, "Stage 2")
        
        # Define models optimized for small datasets
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        weight_dict = {i: w for i, w in enumerate(class_weights)}
        
        models = {
            'RandomForest_Small': RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight=weight_dict,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'LogisticRegression_L1': LogisticRegression(
                class_weight=weight_dict,
                random_state=self.random_state,
                max_iter=1000,
                solver='liblinear',
                penalty='l1',
                C=0.1
            ),
            'GradientBoosting_Small': GradientBoostingClassifier(
                n_estimators=50,
                learning_rate=0.2,
                max_depth=4,
                subsample=0.8,
                random_state=self.random_state
            )
        }
        
        # Evaluate models
        print("\n🔄 Model Evaluation for Stage 2")
        best_model = None
        best_score = 0
        results = {}
        
        for name, model in models.items():
            print(f"\nTesting {name}...")
            
            try:
                # Create pipeline
                if resampler is not None:
                    pipeline = ImbPipeline([
                        ('resampler', resampler),
                        ('classifier', model)
                    ])
                else:
                    pipeline = model
                
                # Use fewer folds for small datasets
                n_folds = max(2, min(5, len(y) // 10))
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
                
                # Evaluate
                f1_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1', n_jobs=-1)
                precision_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='precision', n_jobs=-1)
                recall_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='recall', n_jobs=-1)
                
                results[name] = {
                    'f1_mean': f1_scores.mean(),
                    'f1_std': f1_scores.std(),
                    'precision_mean': precision_scores.mean(),
                    'recall_mean': recall_scores.mean(),
                    'pipeline': pipeline
                }
                
                print(f"   F1: {f1_scores.mean():.4f} (±{f1_scores.std():.4f})")
                print(f"   Precision: {precision_scores.mean():.4f}")
                print(f"   Recall: {recall_scores.mean():.4f}")
                
                if f1_scores.mean() > best_score:
                    best_score = f1_scores.mean()
                    best_model = name
                    
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                continue
        
        if best_model:
            print(f"\n✅ Best Stage 2 Model: {best_model} (F1: {best_score:.4f})")
            
            # Train final model
            self.stage2_model = results[best_model]['pipeline']
            self.stage2_model.fit(X, y)
            self.stage2_history = results
        else:
            print("\n⚠️ No viable Stage 2 model found")
        
        return self.stage2_model, results
    
    def predict_two_stage(self, X):
        """
        Make predictions using two-stage approach
        """
        if self.stage1_model is None:
            raise ValueError("Stage 1 model not trained")
        
        # Stage 1: Risk vs No-Risk
        stage1_pred = self.stage1_model.predict(X)
        stage1_proba = self.stage1_model.predict_proba(X)[:, 1]  # Risk probability
        
        # Initialize final predictions
        final_pred = np.zeros(len(X))  # 0 = Low risk
        final_proba = np.zeros((len(X), 3))  # [Low, Moderate, High]
        
        # No-risk samples stay as low risk (0)
        no_risk_mask = stage1_pred == 0
        final_pred[no_risk_mask] = 0
        final_proba[no_risk_mask, 0] = 1.0
        
        # Risk samples go to Stage 2
        risk_mask = stage1_pred == 1
        
        if np.sum(risk_mask) > 0 and self.stage2_model is not None:
            X_risk = X[risk_mask]
            stage2_pred = self.stage2_model.predict(X_risk)
            stage2_proba = self.stage2_model.predict_proba(X_risk)
            
            # Map Stage 2 predictions: 0 = Moderate (1), 1 = High (2)
            final_pred[risk_mask] = stage2_pred + 1
            
            # Distribute probabilities
            final_proba[risk_mask, 0] = 0  # No low risk for stage 2
            final_proba[risk_mask, 1] = stage2_proba[:, 0]  # Moderate
            final_proba[risk_mask, 2] = stage2_proba[:, 1]  # High
            
        elif np.sum(risk_mask) > 0:
            # If Stage 2 model not available, classify all risk as moderate
            final_pred[risk_mask] = 1
            final_proba[risk_mask, 1] = 1.0
        
        return final_pred.astype(int), final_proba, stage1_pred, stage1_proba
    
    def evaluate_model(self, X_test, y_test, stage_name=""):
        """
        Comprehensive model evaluation
        """
        print(f"\n📊 Model Evaluation {stage_name}")
        print("-" * 40)
        
        # Make predictions
        if stage_name == "Stage 1":
            y_pred = self.stage1_model.predict(X_test)
            y_proba = self.stage1_model.predict_proba(X_test)[:, 1]
        elif stage_name == "Stage 2":
            if self.stage2_model is None:
                print("⚠️ Stage 2 model not available")
                return {}
            y_pred = self.stage2_model.predict(X_test)
            y_proba = self.stage2_model.predict_proba(X_test)[:, 1]
        else:
            # Two-stage evaluation
            y_pred, y_proba_full, _, _ = self.predict_two_stage(X_test)
            y_proba = y_proba_full[:, 2]  # High risk probability
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # AUC (for binary classification)
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, y_proba)
            print(f"AUC: {auc:.4f}")
        
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        return {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc if len(np.unique(y_test)) == 2 else None,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_proba
        }
    
    def create_critical_validation_set(self, df, n_high_risk=None):
        """
        Create a critical validation set focusing on high-risk interactions
        """
        print("\n🔍 Creating Critical Validation Set")
        print("-" * 40)
        
        # Extract all high-risk interactions
        high_risk_mask = df['severity_encoded'] == 2
        high_risk_df = df[high_risk_mask].copy()
        
        print(f"Found {len(high_risk_df)} high-risk interactions")
        
        if len(high_risk_df) == 0:
            print("⚠️ No high-risk interactions found")
            return None, None
        
        # Add known dangerous pairs if possible
        dangerous_pairs = [
            ('warfarin', 'vitamin k'),
            ('warfarin', 'green leafy vegetables'),
            ('monoamine oxidase inhibitors', 'tyramine'),
            ('digoxin', 'licorice'),
            ('theophylline', 'caffeine')
        ]
        
        print(f"\n🎯 Critical Validation Set Contents:")
        print(f"High-risk interactions: {len(high_risk_df)}")
        
        # Sample for manual verification if too many
        if n_high_risk and len(high_risk_df) > n_high_risk:
            validation_df = high_risk_df.sample(n=n_high_risk, random_state=self.random_state)
            print(f"Sampled to {n_high_risk} for manual verification")
        else:
            validation_df = high_risk_df
        
        # Create validation features
        validation_indices = validation_df.index.tolist()
        
        return validation_df, validation_indices
    
    def fit(self, X, y, df=None):
        """
        Fit the two-stage classifier
        """
        print("\n🚀 TWO-STAGE RISK CLASSIFIER TRAINING")
        print("=" * 60)
        
        # Prepare labels
        if df is not None:
            stage1_labels, stage2_labels, stage2_df, label_info = self.prepare_two_stage_labels(df)
        else:
            # Assume y contains severity labels
            stage1_labels = (y > 0).astype(int)
            stage2_mask = y > 0
            stage2_labels = (y[stage2_mask] == 2).astype(int) if np.sum(stage2_mask) > 0 else np.array([])
        
        # Train Stage 1
        self.train_stage1_models(X, stage1_labels)
        
        # Train Stage 2 (only on risk samples)
        if len(stage2_labels) > 0:
            X_stage2 = X[y > 0] if df is None else X[df['severity_encoded'] > 0]
            self.train_stage2_models(X_stage2, stage2_labels)
        
        # Create critical validation set
        if df is not None:
            critical_df, critical_indices = self.create_critical_validation_set(df, n_high_risk=50)
            self.validation_results['critical_validation'] = {
                'indices': critical_indices,
                'df': critical_df
            }
        
        print("\n✅ Two-Stage Classifier Training Complete!")
        return self
    
    def save_models(self, filepath_prefix="two_stage_classifier"):
        """Save trained models"""
        models_to_save = {
            'stage1_model': self.stage1_model,
            'stage2_model': self.stage2_model,
            'scaler': self.scaler,
            'stage1_history': self.stage1_history,
            'stage2_history': self.stage2_history,
            'validation_results': self.validation_results
        }
        
        for name, model in models_to_save.items():
            if model is not None:
                filename = f"{filepath_prefix}_{name}.pkl"
                joblib.dump(model, filename)
                print(f"✅ Saved {name} to {filename}")


# Example usage function
def run_two_stage_classification():
    """
    Complete example of running two-stage classification
    """
    print("🚀 DRUG-FOOD INTERACTION TWO-STAGE CLASSIFICATION")
    print("=" * 60)
    
    # Load your data (replace with actual file paths)
    print("📂 Loading data...")
    
    # Load features from Step 1
    final_features = np.load('final_features.npy')
    merged_df = pd.read_csv('merged_drug_food_data.csv')
    
    print(f"✅ Data loaded: {final_features.shape[0]} samples, {final_features.shape[1]} features")
    
    # Initialize classifier
    classifier = TwoStageRiskClassifier(random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        final_features, 
        merged_df['severity_encoded'],
        test_size=0.2,
        random_state=42,
        stratify=merged_df['severity_encoded']
    )
    
    # Fit the classifier
    classifier.fit(X_train, y_train, merged_df.iloc[X_train.index] if hasattr(X_train, 'index') else None)
    
    # Evaluate
    print("\n📊 Final Evaluation")
    results = classifier.evaluate_model(X_test, y_test, "Two-Stage")
    
    # Save models
    classifier.save_models("drug_food_classifier")
    
    print("\n🎉 Two-Stage Classification Complete!")
    return classifier, results

if __name__ == "__main__":
    # Run the complete pipeline
    classifier, results = run_two_stage_classification()