import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestClassifier, 
                             GradientBoostingClassifier,
                             HistGradientBoostingClassifier,
                             StackingClassifier,
                             VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import (train_test_split, StratifiedKFold, 
                                   cross_val_score, GridSearchCV)
from sklearn.metrics import (classification_report, confusion_matrix, 
                           precision_recall_curve, PrecisionRecallDisplay,
                           f1_score, roc_auc_score, accuracy_score,
                           precision_score, recall_score, make_scorer)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class EnhancedDrugInteractionClassifier:
    def __init__(self, drugbank_path):
        self.drugbank = self._load_dataset('/Users/sachidhoka/Desktop/processed_drugbank_data.csv')
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.models = {}
        self.optimal_thresholds = {}
        self.feature_names = None
        self.is_fitted = False
        self.class_weights = None
        
        # Store all metrics for comprehensive evaluation
        self.metrics_history = {}
    
    def _load_dataset(self, path):
        """Load dataset with proper error handling"""
        try:
            df = pd.read_csv(path)
            print(f"Loaded dataset with shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"Error loading {path}: {str(e)}")
            return None
    
    def _create_interaction_labels(self):
        """Enhanced rule-based interaction potential labels"""
        print("Creating enhanced interaction potential labels...")
        self.drugbank['interaction_potential'] = 'Low'  # Default
        
        # Enhanced high-risk categories
        high_risk_categories = [
            'anticoagulant', 'antiarrhythmic', 'immunosuppressant',
            'cytochrome p450', 'narrow therapeutic index', 'warfarin',
            'digoxin', 'lithium', 'phenytoin', 'theophylline',
            'chemotherapy', 'antipsychotic', 'tricyclic antidepressant'
        ]
        
        # Enhanced medium-risk categories
        medium_risk_categories = [
            'antidepressant', 'antihypertensive', 'antibiotic',
            'nsaid', 'beta blocker', 'ace inhibitor', 'diuretic',
            'anticonvulsant', 'corticosteroid', 'opioid'
        ]
        
        def safe_numeric_conversion(value, default=0):
            """Safely convert value to numeric, handling strings and NaN"""
            if pd.isna(value):
                return default
            try:
                if isinstance(value, str):
                    import re
                    numeric_str = re.sub(r'[^\d.-]', '', value)
                    if numeric_str:
                        return float(numeric_str)
                    else:
                        return default
                return float(value)
            except (ValueError, TypeError):
                return default
        
        for idx, row in self.drugbank.iterrows():
            risk_score = 0
            
            # Enhanced molecular weight scoring
            if 'molecular_weight' in self.drugbank.columns:
                mw = safe_numeric_conversion(row.get('molecular_weight'))
                if mw > 800:  # Very large molecules
                    risk_score += 2
                elif mw > 500:
                    risk_score += 1
            
            # Enhanced protein binding scoring
            protein_binding_cols = [col for col in self.drugbank.columns 
                                  if 'protein' in col.lower() and 'binding' in col.lower()]
            for pb_col in protein_binding_cols:
                pb_val = safe_numeric_conversion(row.get(pb_col))
                if pb_val > 95:  # Very high protein binding
                    risk_score += 2
                elif pb_val > 90:
                    risk_score += 1
                break
            
            # Enhanced half-life scoring
            if 'half-life_numeric' in self.drugbank.columns:
                hl_val = safe_numeric_conversion(row.get('half-life_numeric'))
                if hl_val > 48:  # Very long half-life
                    risk_score += 2
                elif hl_val > 24:
                    risk_score += 1
            
            # Enhanced clearance scoring
            if 'clearance_numeric' in self.drugbank.columns:
                cl_val = safe_numeric_conversion(row.get('clearance_numeric'))
                if 0 < cl_val < 5:  # Very low clearance
                    risk_score += 2
                elif 0 < cl_val < 10:
                    risk_score += 1
            
            # Volume of distribution scoring (new)
            if 'volume-of-distribution_numeric' in self.drugbank.columns:
                vd_val = safe_numeric_conversion(row.get('volume-of-distribution_numeric'))
                if vd_val > 10:  # High volume of distribution
                    risk_score += 1
            
            # Bioavailability scoring (new)
            if 'bioavailability' in self.drugbank.columns:
                ba_val = safe_numeric_conversion(row.get('bioavailability'))
                if ba_val < 30:  # Low bioavailability can indicate complex metabolism
                    risk_score += 1
            
            # Enhanced text analysis
            text_fields = ['categories', 'description', 'indication', 'mechanism-of-action', 'groups']
            combined_text = ''
            for field in text_fields:
                if field in self.drugbank.columns:
                    field_val = str(row.get(field, '')).lower()
                    combined_text += ' ' + field_val
            
            # Enhanced category scoring
            high_risk_matches = sum(1 for cat in high_risk_categories if cat in combined_text)
            medium_risk_matches = sum(1 for cat in medium_risk_categories if cat in combined_text)
            
            if high_risk_matches >= 2:
                risk_score += 4
            elif high_risk_matches >= 1:
                risk_score += 3
            elif medium_risk_matches >= 2:
                risk_score += 2
            elif medium_risk_matches >= 1:
                risk_score += 1
            
            # Enhanced ATC code analysis
            if 'atc-codes' in self.drugbank.columns:
                atc_codes = str(row.get('atc-codes', ''))
                if atc_codes:
                    code_count = len(atc_codes.split(';'))
                    if code_count > 3:
                        risk_score += 2
                    elif code_count > 2:
                        risk_score += 1
            
            # Metabolism complexity scoring
            if 'metabolism' in self.drugbank.columns:
                metabolism_text = str(row.get('metabolism', '')).lower()
                complex_terms = ['cyp', 'cytochrome', 'enzyme', 'inhibit', 'induce', 'substrate']
                metabolism_score = sum(1 for term in complex_terms if term in metabolism_text)
                if metabolism_score >= 3:
                    risk_score += 2
                elif metabolism_score >= 1:
                    risk_score += 1
            
            # Final label assignment with more nuanced thresholds
            if risk_score >= 5:
                self.drugbank.at[idx, 'interaction_potential'] = 'High'
            elif risk_score >= 2:
                self.drugbank.at[idx, 'interaction_potential'] = 'Moderate'
        
        # Print enhanced distribution
        dist = self.drugbank['interaction_potential'].value_counts()
        print(f"Enhanced label distribution:\n{dist}")
        print(f"Percentages:\n{dist / len(self.drugbank) * 100}")
    
    def prepare_data(self, sampling_strategy='smote', test_size=0.2):
        """Enhanced data preparation with multiple sampling strategies"""
        if self.drugbank is None:
            raise ValueError("DrugBank data not loaded")
        
        if 'interaction_potential' not in self.drugbank.columns:
            self._create_interaction_labels()
        
        # Feature engineering: create interaction features
        self._create_interaction_features()
        
        # Select numeric features
        numeric_cols = self.drugbank.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['interaction_potential']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        print(f"Using {len(numeric_cols)} features: {numeric_cols}")
        
        X = self.drugbank[numeric_cols].copy()
        y = self.drugbank['interaction_potential'].copy()
        
        # Enhanced feature selection
        valid_cols = []
        for col in X.columns:
            non_null_count = X[col].notna().sum()
            if non_null_count > 50:  # More lenient threshold
                valid_cols.append(col)
            else:
                print(f"Removing column {col} - only {non_null_count} non-null values")
        
        X = X[valid_cols]
        
        # Imputation and scaling
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=valid_cols,
            index=X.index
        )
        
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_imputed),
            columns=X_imputed.columns,
            index=X_imputed.index
        )
        
        self.feature_names = X_scaled.columns.tolist()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Calculate and store class weights
        classes = np.unique(y_encoded)
        weights = compute_class_weight('balanced', classes=classes, y=y_encoded)
        self.class_weights = dict(zip(classes, weights))
        print(f"Class weights: {self.class_weights}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, 
            stratify=y_encoded, random_state=42
        )
        
        # Apply sampling strategy
        if sampling_strategy and sampling_strategy != 'none':
            X_train, y_train = self._apply_sampling_strategy(
                X_train, y_train, sampling_strategy
            )
        
        return X_train, X_test, y_train, y_test
    
    def _create_interaction_features(self):
        """Create new features that might indicate interaction potential"""
        # Molecular complexity indicator
        if 'molecular_weight' in self.drugbank.columns and 'h_bond_acceptors' in self.drugbank.columns:
            mw = pd.to_numeric(self.drugbank['molecular_weight'], errors='coerce')
            hba = pd.to_numeric(self.drugbank['h_bond_acceptors'], errors='coerce')
            self.drugbank['molecular_complexity'] = (mw * hba).fillna(0)
        
        # Pharmacokinetic risk score
        pk_columns = ['half-life_numeric', 'protein-binding_numeric', 'clearance_numeric']
        pk_values = []
        for col in pk_columns:
            if col in self.drugbank.columns:
                pk_values.append(pd.to_numeric(self.drugbank[col], errors='coerce').fillna(0))
        
        if pk_values:
            # Normalize and combine PK parameters
            pk_matrix = np.column_stack(pk_values)
            pk_matrix = (pk_matrix - np.nanmean(pk_matrix, axis=0)) / (np.nanstd(pk_matrix, axis=0) + 1e-8)
            self.drugbank['pk_risk_score'] = np.nansum(np.abs(pk_matrix), axis=1)
        
        print("Created interaction features: molecular_complexity, pk_risk_score")
    
    def _apply_sampling_strategy(self, X_train, y_train, strategy):
        """Apply various sampling strategies to handle class imbalance"""
        print(f"Applying {strategy} sampling strategy...")
        original_dist = pd.Series(y_train).value_counts()
        print(f"Original distribution: {original_dist.to_dict()}")
        
        if strategy == 'smote':
            sampler = SMOTE(random_state=42, k_neighbors=3)
        elif strategy == 'adasyn':
            sampler = ADASYN(random_state=42, n_neighbors=3)
        elif strategy == 'borderline_smote':
            sampler = BorderlineSMOTE(random_state=42, k_neighbors=3)
        elif strategy == 'smote_tomek':
            sampler = SMOTETomek(random_state=42)
        elif strategy == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        else:
            return X_train, y_train
        
        try:
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            new_dist = pd.Series(y_resampled).value_counts()
            print(f"New distribution: {new_dist.to_dict()}")
            return X_resampled, y_resampled
        except Exception as e:
            print(f"Sampling failed: {str(e)}, using original data")
            return X_train, y_train
    
    def initialize_models(self):
        """Initialize models with enhanced configurations and hyperparameter tuning"""
        # Calculate pos_weight for XGBoost (assuming binary classification)
        if len(self.class_weights) == 2:
            pos_weight = self.class_weights[0] / self.class_weights[1]
        else:
            pos_weight = 1.0
        
        self.models = {
            'RF': RandomForestClassifier(
                n_estimators=200,  # Increased
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            ),
            'XGB': XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=pos_weight,
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1
            ),
            'LGBM': LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42,
                verbose=-1
            ),
            'GBC': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'HGB': HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            ),
            'SVM': SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            ),
            'LR': LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
        }
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Enhanced model training with comprehensive metrics"""
        if not self.models:
            self.initialize_models()
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Training set class distribution: {pd.Series(y_train).value_counts().to_dict()}")
        
        results = {}
        
        # Define custom scoring for imbalanced data
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='weighted', zero_division=0),
            'recall': make_scorer(recall_score, average='weighted', zero_division=0),
            'f1': make_scorer(f1_score, average='weighted', zero_division=0)
        }
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Fit the model
                model.fit(X_train, y_train)
                
                # Cross-validation with multiple metrics
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
                cv_results = {}
                for metric_name, scorer in scoring.items():
                    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scorer, n_jobs=-1)
                    cv_results[f'cv_{metric_name}_mean'] = np.mean(scores)
                    cv_results[f'cv_{metric_name}_std'] = np.std(scores)
                
                # Test set predictions
                y_pred = model.predict(X_test)
                y_pred_proba = None
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                
                # Calculate comprehensive test metrics
                test_metrics = {
                    'test_accuracy': accuracy_score(y_test, y_pred),
                    'test_precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'test_recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'test_f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                }
                
                # Calculate AUC for binary classification
                if y_pred_proba is not None and len(np.unique(y_test)) == 2:
                    try:
                        test_metrics['test_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                    except:
                        test_metrics['test_auc'] = np.nan
                
                # Combine results
                results[name] = {
                    'model': model,
                    **cv_results,
                    **test_metrics,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                # Find optimal threshold
                if y_pred_proba is not None:
                    self._find_optimal_threshold(model, X_test, y_test, name)
                
                # Print comprehensive results
                print(f"{name} Results:")
                print(f"  CV F1: {cv_results['cv_f1_mean']:.3f} ± {cv_results['cv_f1_std']:.3f}")
                print(f"  Test Accuracy: {test_metrics['test_accuracy']:.3f}")
                print(f"  Test Precision: {test_metrics['test_precision']:.3f}")
                print(f"  Test Recall: {test_metrics['test_recall']:.3f}")
                print(f"  Test F1: {test_metrics['test_f1']:.3f}")
                if 'test_auc' in test_metrics and not np.isnan(test_metrics['test_auc']):
                    print(f"  Test AUC: {test_metrics['test_auc']:.3f}")
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        # Create enhanced ensemble
        self._create_enhanced_ensemble(X_train, y_train, X_test, y_test, results)
        
        self.is_fitted = True
        self.metrics_history = results
        return results
    
    def _create_enhanced_ensemble(self, X_train, y_train, X_test, y_test, results):
        """Create both stacking and voting ensembles"""
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if len(valid_results) < 3:
            print("Not enough models for ensemble creation")
            return
        
        # Select top models based on F1 score
        sorted_models = sorted(valid_results.items(), 
                             key=lambda x: x[1]['cv_f1_mean'], reverse=True)
        top_models = sorted_models[:3]
        
        print(f"Creating ensembles with: {[name for name, _ in top_models]}")
        
        estimators = [(name, results[name]['model']) for name, _ in top_models]
        
        # Stacking Ensemble
        try:
            stacking_model = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(class_weight='balanced', random_state=42),
                cv=3,
                n_jobs=-1
            )
            
            stacking_model.fit(X_train, y_train)
            self._evaluate_ensemble_model(stacking_model, 'Stacking', X_train, y_train, X_test, y_test, results)
            
        except Exception as e:
            print(f"Error creating stacking ensemble: {str(e)}")
        
        # Voting Ensemble
        try:
            voting_model = VotingClassifier(
                estimators=estimators,
                voting='soft',
                n_jobs=-1
            )
            
            voting_model.fit(X_train, y_train)
            self._evaluate_ensemble_model(voting_model, 'Voting', X_train, y_train, X_test, y_test, results)
            
        except Exception as e:
            print(f"Error creating voting ensemble: {str(e)}")
    
    def _evaluate_ensemble_model(self, model, name, X_train, y_train, X_test, y_test, results):
        """Evaluate ensemble model with comprehensive metrics"""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Cross-validation scores
        cv_accuracy = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        cv_precision = cross_val_score(model, X_train, y_train, cv=cv, scoring='precision_weighted', n_jobs=-1)
        cv_recall = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall_weighted', n_jobs=-1)
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
        
        # Test predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Test metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        test_metrics = {
            'cv_accuracy_mean': np.mean(cv_accuracy),
            'cv_accuracy_std': np.std(cv_accuracy),
            'cv_precision_mean': np.mean(cv_precision),
            'cv_precision_std': np.std(cv_precision),
            'cv_recall_mean': np.mean(cv_recall),
            'cv_recall_std': np.std(cv_recall),
            'cv_f1_mean': np.mean(cv_f1),
            'cv_f1_std': np.std(cv_f1),
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'model': model
        }
        
        # Calculate AUC if binary classification
        if y_pred_proba is not None and len(np.unique(y_test)) == 2:
            try:
                test_metrics['test_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            except:
                test_metrics['test_auc'] = np.nan
        
        results[name] = test_metrics
        self.models[name] = model
        
        print(f"{name} Ensemble:")
        print(f"  CV F1: {np.mean(cv_f1):.3f} ± {np.std(cv_f1):.3f}")
        print(f"  Test Accuracy: {test_accuracy:.3f}")
        print(f"  Test Precision: {test_precision:.3f}")
        print(f"  Test Recall: {test_recall:.3f}")
        print(f"  Test F1: {test_f1:.3f}")
        if 'test_auc' in test_metrics and not np.isnan(test_metrics['test_auc']):
            print(f"  Test AUC: {test_metrics['test_auc']:.3f}")
    
    def _find_optimal_threshold(self, model, X_test, y_test, model_name):
        """Find optimal threshold for binary classification"""
        if len(np.unique(y_test)) != 2:
            return
        
        y_proba = model.predict_proba(X_test)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
        
        # Find threshold that maximizes F1 score
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        
        if optimal_idx < len(thresholds):
            self.optimal_thresholds[model_name] = thresholds[optimal_idx]
        else:
            self.optimal_thresholds[model_name] = 0.5
        
        print(f"  Optimal threshold: {self.optimal_thresholds[model_name]:.3f}")
    
    def evaluate_models(self, X_test, y_test, results):
        """Enhanced model evaluation with comprehensive metrics explanation"""
        print("\n" + "="*70)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*70)
        
        print("\n" + "="*50)
        print("WHY F1-SCORE IS MOST IMPORTANT HERE:")
        print("="*50)
        print("1. SEVERE CLASS IMBALANCE: 94.6% Moderate vs 5.4% High risk")
        print("2. ACCURACY PARADOX: A model predicting all 'Moderate' gets 94.6% accuracy!")
        print("3. F1-SCORE: Balances precision (avoiding false alarms) and recall (catching true risks)")
        print("4. CLINICAL IMPORTANCE: Missing high-risk drugs (low recall) can be dangerous")
        print("5. RESOURCE ALLOCATION: Too many false positives (low precision) wastes resources")
        print("6. F1 = 2 * (Precision * Recall) / (Precision + Recall) - harmonic mean")
        print("="*50)
        
        # Create summary table
        summary_data = []
        for name, result in results.items():
            if 'error' in result:
                continue
            
            summary_data.append({
                'Model': name,
                'Accuracy': f"{result.get('test_accuracy', 0):.3f}",
                'Precision': f"{result.get('test_precision', 0):.3f}",
                'Recall': f"{result.get('test_recall', 0):.3f}",
                'F1-Score': f"{result.get('test_f1', 0):.3f}",
                'AUC': f"{result.get('test_auc', 0):.3f}" if 'test_auc' in result else 'N/A'
            })
        
        # Print summary table
        if summary_data:
            print(f"\n{'='*80}")
            print("MODEL PERFORMANCE SUMMARY")
            print(f"{'='*80}")
            print(f"{'Model':<12} {'Accuracy':<9} {'Precision':<10} {'Recall':<8} {'F1-Score':<9} {'AUC':<8}")
            print("-" * 80)
            for row in summary_data:
                print(f"{row['Model']:<12} {row['Accuracy']:<9} {row['Precision']:<10} {row['Recall']:<8} {row['F1-Score']:<9} {row['AUC']:<8}")
        
        # Detailed evaluation for each model
        for name, result in results.items():
            if 'error' in result:
                print(f"\n{name}: FAILED - {result['error']}")
                continue
            
            print(f"\n{name.upper()} MODEL - DETAILED ANALYSIS:")
            print("-" * 50)
            
            y_pred = result['y_pred']
            y_pred_labels = self.label_encoder.inverse_transform(y_pred)
            y_true_labels = self.label_encoder.inverse_transform(y_test)
            
            # Classification report
            print("Classification Report:")
            print(classification_report(y_true_labels, y_pred_labels, digits=3))
            
            # Confusion matrix
            cm = confusion_matrix(y_true_labels, y_pred_labels)
            print(f"\nConfusion Matrix:")
            print(f"{'Predicted':<15} {'Low':<8} {'Moderate':<10} {'High':<8}")
            print("-" * 45)
            class_names = self.label_encoder.classes_
            for i, true_class in enumerate(class_names):
                row_str = f"Actual {true_class:<8}"
                for j in range(len(class_names)):
                    if j < len(cm[i]):
                        row_str += f" {cm[i][j]:<8}"
                    else:
                        row_str += f" {0:<8}"
                print(row_str)
        
        # Find and highlight best model
        best_model_name = max(results.keys(), 
                             key=lambda x: results[x].get('test_f1', 0) if 'error' not in results[x] else 0)
        
        print(f"\n{'='*60}")
        print(f"🏆 BEST MODEL: {best_model_name}")
        print(f"🏆 F1-Score: {results[best_model_name]['test_f1']:.3f}")
        print(f"{'='*60}")
        
        return results
    
    def plot_model_comparison(self, results):
        """Create comprehensive visualization of model performance"""
        # Filter out failed models
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            print("No valid results to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Enhanced Drug Interaction Classifier - Model Comparison', fontsize=16, fontweight='bold')
        
        models = list(valid_results.keys())
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
        
        # Plot each metric
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]
            values = [valid_results[model][metric] for model in models]
            
            bars = ax.bar(models, values, color=colors[:len(models)], alpha=0.7, edgecolor='black')
            ax.set_title(f'{name} Comparison', fontweight='bold')
            ax.set_ylabel(name)
            ax.set_ylim(0, 1.1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Highlight best performing model
            best_idx = values.index(max(values))
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(2)
            
            ax.grid(True, alpha=0.3)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
        
        # Create a radar chart for comprehensive comparison
        self._create_radar_chart(valid_results)
    
    def _create_radar_chart(self, results):
        """Create radar chart for multi-dimensional model comparison"""
        try:
            from math import pi
            
            models = list(results.keys())
            metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
            metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            
            # Number of metrics
            N = len(metrics)
            
            # Angles for each metric on the radar chart
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]  # Complete the circle
            
            # Create radar chart
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
            
            for i, model in enumerate(models[:7]):  # Limit to 7 models for readability
                values = [results[model][metric] for metric in metrics]
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i % len(colors)])
                ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
            
            # Add labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels)
            ax.set_ylim(0, 1)
            ax.set_title('Model Performance Radar Chart', size=16, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Radar chart requires matplotlib with polar projection support")
    
    def get_feature_importance(self, model_name='RF'):
        """Get and visualize feature importance for tree-based models"""
        if not self.is_fitted or model_name not in self.models:
            print(f"Model {model_name} not found or not fitted")
            return None
        
        model = self.models[model_name]['model'] if isinstance(self.models[model_name], dict) else self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = self.feature_names
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(20)
            
            bars = plt.barh(range(len(top_features)), top_features['importance'], color='skyblue', edgecolor='navy')
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'{model_name} - Top 20 Feature Importances')
            plt.gca().invert_yaxis()
            
            # Add value labels
            for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
                plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{importance:.3f}', va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
            return importance_df
        
        else:
            print(f"Model {model_name} doesn't support feature importance")
            return None
    
    def predict_interaction_risk(self, drug_features, model_name='best', return_proba=True):
        """Predict interaction risk for new drug features"""
        if not self.is_fitted:
            print("Models not fitted yet. Please train models first.")
            return None
        
        # Determine which model to use
        if model_name == 'best':
            # Find best model based on F1 score
            valid_results = {k: v for k, v in self.metrics_history.items() if 'error' not in v}
            model_name = max(valid_results.keys(), key=lambda x: valid_results[x]['test_f1'])
            print(f"Using best model: {model_name}")
        
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return None
        
        model = self.models[model_name]['model'] if isinstance(self.models[model_name], dict) else self.models[model_name]
        
        # Ensure features are in correct format
        if isinstance(drug_features, dict):
            # Convert dict to DataFrame
            drug_df = pd.DataFrame([drug_features])
        elif isinstance(drug_features, pd.Series):
            drug_df = drug_features.to_frame().T
        else:
            drug_df = pd.DataFrame(drug_features)
        
        # Align features with training features
        for feature in self.feature_names:
            if feature not in drug_df.columns:
                drug_df[feature] = 0  # Fill missing features with 0
        
        drug_df = drug_df[self.feature_names]  # Ensure correct order
        
        # Apply same preprocessing as training
        drug_imputed = pd.DataFrame(
            self.imputer.transform(drug_df),
            columns=self.feature_names
        )
        
        drug_scaled = pd.DataFrame(
            self.scaler.transform(drug_imputed),
            columns=self.feature_names
        )
        
        # Make predictions
        prediction = model.predict(drug_scaled)[0]
        prediction_label = self.label_encoder.inverse_transform([prediction])[0]
        
        result = {'predicted_class': prediction_label}
        
        if return_proba and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(drug_scaled)[0]
            prob_dict = {self.label_encoder.inverse_transform([i])[0]: prob 
                        for i, prob in enumerate(probabilities)}
            result['probabilities'] = prob_dict
            
            # Apply optimal threshold if available
            if model_name in self.optimal_thresholds and len(probabilities) == 2:
                optimal_pred = 1 if probabilities[1] >= self.optimal_thresholds[model_name] else 0
                result['optimal_threshold_pred'] = self.label_encoder.inverse_transform([optimal_pred])[0]
        
        return result
    
    def save_model(self, filepath, model_name='best'):
        """Save trained model and preprocessing components"""
        import pickle
        
        if not self.is_fitted:
            print("No fitted models to save")
            return False
        
        # Determine which model to save
        if model_name == 'best':
            valid_results = {k: v for k, v in self.metrics_history.items() if 'error' not in v}
            model_name = max(valid_results.keys(), key=lambda x: valid_results[x]['test_f1'])
        
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return False
        
        # Prepare save data
        save_data = {
            'model': self.models[model_name]['model'] if isinstance(self.models[model_name], dict) else self.models[model_name],
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'optimal_thresholds': self.optimal_thresholds,
            'model_name': model_name,
            'metrics': self.metrics_history[model_name] if model_name in self.metrics_history else None
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            print(f"Model {model_name} saved successfully to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath):
        """Load saved model and preprocessing components"""
        import pickle
        
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            # Restore components
            self.models = {save_data['model_name']: save_data['model']}
            self.label_encoder = save_data['label_encoder']
            self.scaler = save_data['scaler']
            self.imputer = save_data['imputer']
            self.feature_names = save_data['feature_names']
            self.optimal_thresholds = save_data['optimal_thresholds']
            self.is_fitted = True
            
            if save_data['metrics']:
                self.metrics_history = {save_data['model_name']: save_data['metrics']}
            
            print(f"Model {save_data['model_name']} loaded successfully from {filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

def main():
    """Example usage of the Enhanced Drug Interaction Classifier"""
    print("Enhanced Drug Interaction Classifier")
    print("=" * 50)
    
    # Initialize classifier (replace with your data path)
    classifier = EnhancedDrugInteractionClassifier("path/to/drugbank_data.csv")
    
    # Prepare data with different sampling strategies
    sampling_strategies = ['smote', 'adasyn', 'borderline_smote', 'none']
    
    print("Testing different sampling strategies...")
    
    best_f1 = 0
    best_strategy = None
    best_results = None
    
    for strategy in sampling_strategies:
        print(f"\n{'='*60}")
        print(f"TESTING SAMPLING STRATEGY: {strategy.upper()}")
        print(f"{'='*60}")
        
        try:
            # Prepare data
            X_train, X_test, y_train, y_test = classifier.prepare_data(
                sampling_strategy=strategy, test_size=0.2
            )
            
            # Train models
            results = classifier.train_models(X_train, y_train, X_test, y_test)
            
            # Evaluate models
            classifier.evaluate_models(X_test, y_test, results)
            
            # Find best F1 score for this strategy
            valid_results = {k: v for k, v in results.items() if 'error' not in v}
            if valid_results:
                strategy_best_f1 = max(result['test_f1'] for result in valid_results.values())
                if strategy_best_f1 > best_f1:
                    best_f1 = strategy_best_f1
                    best_strategy = strategy
                    best_results = results
            
        except Exception as e:
            print(f"Error with strategy {strategy}: {str(e)}")
            continue
    
    # Final results
    if best_results:
        print(f"\n{'='*70}")
        print(f"🏆 BEST OVERALL CONFIGURATION:")
        print(f"🏆 Sampling Strategy: {best_strategy}")
        print(f"🏆 Best F1-Score: {best_f1:.3f}")
        print(f"{'='*70}")
        
        # Plot comparison for best strategy
        classifier.plot_model_comparison(best_results)
        
        # Show feature importance
        classifier.get_feature_importance('RF')
        
        # Save best model
        classifier.save_model('best_drug_interaction_model.pkl')

if __name__ == "__main__":
    main()