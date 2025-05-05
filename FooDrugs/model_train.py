import pandas as pd
import numpy as np
import joblib
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import time
from tqdm import tqdm

# Set PyTorch to use MPS for Apple Silicon if available
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())  
print("MPS available:", torch.backends.mps.is_available())

# Check if output files exist and remove them to avoid errors
for file_path in ['food_drug_rf_model.pkl', 'food_drug_encodings.pkl', 
                  'food_drug_gnn_model.pt', 'food_drug_graph_mappings.pkl']:
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Removed existing file: {file_path}")

# Part 1: Load and prepare the SMOTE-balanced data
print("\nLoading SMOTE-balanced training data...")
smote_train = pd.read_csv('/Users/sachidhoka/Desktop/Drug Food/food_drug_analysis/smote_training_data.csv')
X_smote = smote_train[['food_encoded', 'drug_encoded']]
y_smote = smote_train['has_interaction']
print(f"SMOTE data shape: {X_smote.shape}")

# Part 2: Train Random Forest model as before
print("\nTraining Random Forest model...")
start_time = time.time()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_smote, y_smote)
print(f"Random Forest training completed in {time.time() - start_time:.2f} seconds")

# Save the Random Forest model
joblib.dump(rf_model, 'food_drug_rf_model.pkl')
print("Random Forest model saved to 'food_drug_rf_model.pkl'")

# Part 3: Create encoders for user input
print("\nCreating lookup encoders for user input...")
# Load the model_ready dataset with text and encoded values
model_ready = pd.read_csv('/Users/sachidhoka/Desktop/Drug Food/food_drug_analysis/model_ready_food_drug_interactions.csv')
print(f"Model ready data shape: {model_ready.shape}")

# Create mappings from text to encoding
food_to_encoding = dict(zip(model_ready['food'], model_ready['food_encoded']))
drug_to_encoding = dict(zip(model_ready['drug'], model_ready['drug_encoded']))

# Create reverse mappings from encoding to text
encoding_to_food = dict(zip(model_ready['food_encoded'], model_ready['food']))
encoding_to_drug = dict(zip(model_ready['drug_encoded'], model_ready['drug']))

# Save these mappings for later use
with open('food_drug_encodings.pkl', 'wb') as f:
    pickle.dump({
        'food_to_encoding': food_to_encoding,
        'drug_to_encoding': drug_to_encoding,
        'encoding_to_food': encoding_to_food,
        'encoding_to_drug': encoding_to_drug
    }, f)
print("Encoders saved to 'food_drug_encodings.pkl'")

# Part 4: Implement and train an optimized Graph Neural Network (GNN)
print("\nPreparing data for Graph Neural Network...")

# Define an improved GNN model with more advanced architecture
class EnhancedFoodDrugGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels, num_layers=3, dropout=0.3, output_dim=1, 
                 conv_type='GCN', use_batch_norm=True, residual=True):
        super(EnhancedFoodDrugGNN, self).__init__()
        
        self.num_layers = num_layers
        self.residual = residual
        self.use_batch_norm = use_batch_norm
        
        # Initialize convolution layers
        self.convs = nn.ModuleList()
        
        # Initialize batch norm layers if needed
        if use_batch_norm:
            self.batch_norms = nn.ModuleList()
        
        # Input layer
        if conv_type == 'GCN':
            self.convs.append(GCNConv(input_dim, hidden_channels))
        elif conv_type == 'SAGE':
            self.convs.append(SAGEConv(input_dim, hidden_channels))
        elif conv_type == 'GAT':
            self.convs.append(GATConv(input_dim, hidden_channels, heads=1))
        
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for i in range(num_layers - 1):
            if conv_type == 'GCN':
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            elif conv_type == 'SAGE':
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            elif conv_type == 'GAT':
                self.convs.append(GATConv(hidden_channels, hidden_channels, heads=1))
            
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layers
        self.food_projection = nn.Linear(hidden_channels, hidden_channels)
        self.drug_projection = nn.Linear(hidden_channels, hidden_channels)
        
        # Final prediction layers with more capacity
        self.out_layers = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, output_dim)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        # Apply multiple GNN layers with residual connections
        prev_x = None
        
        for i in range(self.num_layers):
            if i == 0:
                # First layer
                h = self.convs[i](x, edge_index)
            else:
                # Apply convolution
                h = self.convs[i](x, edge_index)
                
                # Add residual connection if enabled
                if self.residual and prev_x is not None and prev_x.shape == h.shape:
                    h = h + prev_x
            
            # Apply batch normalization if enabled
            if self.use_batch_norm:
                h = self.batch_norms[i](h)
            
            # Apply activation and dropout
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
            # Store for residual connection
            prev_x = h
            x = h
        
        return x
    
    def predict_pair(self, node_embeddings, food_idx, drug_idx):
        # Extract and project embeddings for the specific food and drug
        food_embedding = self.food_projection(node_embeddings[food_idx])
        drug_embedding = self.drug_projection(node_embeddings[drug_idx])
        
        # Concatenate projected embeddings
        pair_embedding = torch.cat([food_embedding, drug_embedding], dim=0)
        
        # Pass through output layers
        prediction = self.out_layers(pair_embedding)
        
        return torch.sigmoid(prediction)

# Build a more efficient graph from the food-drug interaction data
def build_graph_from_data(model_ready_data, max_pairs=10000, with_edge_weights=True):
    print("Building graph from interaction data...")
    
    # Get unique foods and drugs with their encodings - ensure integers for node IDs
    unique_foods = sorted(set(int(id) for id in model_ready_data['food_encoded'].unique()))
    unique_drugs = sorted(set(int(id) for id in model_ready_data['drug_encoded'].unique()))
    
    # Additional features for nodes
    interactions_by_food = model_ready_data.groupby('food_encoded')['has_interaction'].agg(['mean', 'count'])
    interactions_by_drug = model_ready_data.groupby('drug_encoded')['has_interaction'].agg(['mean', 'count'])
    
    print(f"Number of unique foods: {len(unique_foods)}")
    print(f"Number of unique drugs: {len(unique_drugs)}")
    
    # Create nodes for foods and drugs
    num_food_nodes = len(unique_foods)
    num_drug_nodes = len(unique_drugs)
    total_nodes = num_food_nodes + num_drug_nodes
    
    # Create enhanced node features (4 features: node type, node ID, interaction rate, frequency)
    x = torch.zeros((total_nodes, 4), dtype=torch.float)
    
    # Food nodes have type 0, drug nodes have type 1
    food_mapping = {food_id: idx for idx, food_id in enumerate(unique_foods)}
    drug_mapping = {drug_id: idx + num_food_nodes for idx, drug_id in enumerate(unique_drugs)}
    
    # Set node features
    max_food_id = max(unique_foods) if unique_foods else 1
    max_drug_id = max(unique_drugs) if unique_drugs else 1
    
    # Set food node features
    for i, food_id in enumerate(unique_foods):
        x[i, 0] = 0  # Node type: food
        x[i, 1] = float(food_id) / max_food_id  # Normalized node ID
        
        # Add interaction rate and frequency if available
        try:
            if int(food_id) in interactions_by_food.index:
                stats = interactions_by_food.loc[int(food_id)]
                x[i, 2] = float(stats['mean'])  # Interaction rate
                x[i, 3] = min(1.0, float(stats['count']) / 100)  # Normalized frequency (capped at 1.0)
        except (KeyError, ValueError):
            pass  # Keep defaults if data not available
    
    # Set drug node features
    for i, drug_id in enumerate(unique_drugs):
        x[i + num_food_nodes, 0] = 1  # Node type: drug
        x[i + num_food_nodes, 1] = float(drug_id) / max_drug_id  # Normalized node ID
        
        # Add interaction rate and frequency if available
        try:
            if int(drug_id) in interactions_by_drug.index:
                stats = interactions_by_drug.loc[int(drug_id)]
                x[i + num_food_nodes, 2] = float(stats['mean'])  # Interaction rate
                x[i + num_food_nodes, 3] = min(1.0, float(stats['count']) / 100)  # Normalized frequency
        except (KeyError, ValueError):
            pass  # Keep defaults if data not available
    
    # Create edges - balance positive and negative interactions
    subset_data = model_ready_data
    if len(model_ready_data) > max_pairs:
        # Balance positive and negative examples
        pos_samples = model_ready_data[model_ready_data['has_interaction'] == 1]
        neg_samples = model_ready_data[model_ready_data['has_interaction'] == 0]
        
        pos_sample_count = min(max_pairs // 2, len(pos_samples))
        neg_sample_count = min(max_pairs - pos_sample_count, len(neg_samples))
        
        subset_pos = pos_samples.sample(pos_sample_count, random_state=42)
        subset_neg = neg_samples.sample(neg_sample_count, random_state=42)
        subset_data = pd.concat([subset_pos, subset_neg])
    
    # Create edges based on interactions
    print(f"Creating edges from {len(subset_data)} interactions...")
    edge_indices = []
    edge_attrs = []
    
    for _, row in subset_data.iterrows():
        try:
            food_idx = food_mapping[int(row['food_encoded'])]
            drug_idx = drug_mapping[int(row['drug_encoded'])]
            
            # Add directed edges in both directions
            edge_indices.append([food_idx, drug_idx])
            edge_attrs.append(float(row['has_interaction']))
            
            edge_indices.append([drug_idx, food_idx]) 
            edge_attrs.append(float(row['has_interaction']))
        except KeyError:
            # Skip pairs with missing mappings
            continue
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 1) if with_edge_weights else None
    
    print(f"Graph created with {x.shape[0]} nodes and {len(edge_indices)} edges")
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), food_mapping, drug_mapping

# Helper function to evaluate model during training
def evaluate_model(model, graph_data, val_pairs, val_labels, device):
    model.eval()
    with torch.no_grad():
        node_embeddings = model(graph_data.x, graph_data.edge_index)
        
        # Collect predictions
        all_preds = []
        for food_idx, drug_idx in val_pairs:
            pred = model.predict_pair(node_embeddings, food_idx, drug_idx)
            all_preds.append(pred)
        
        # Compute metrics
        preds = torch.cat(all_preds).cpu()
        labels = val_labels.cpu()
        
        # Binary accuracy
        binary_preds = (preds >= 0.5).float()
        accuracy = (binary_preds == labels).float().mean().item()
        
        # AUC and loss would require additional imports, compute simpler metrics for now
        mse = F.mse_loss(preds, labels).item()
        
        return {
            'accuracy': accuracy,
            'mse': mse,
            'avg_pred': preds.mean().item()
        }

# Determine device - try MPS first, fallback to CPU if needed
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS (Apple GPU)")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Build the graph with improved representation
graph_data, food_mapping, drug_mapping = build_graph_from_data(model_ready, max_pairs=10000)

# Move data to device
graph_data = graph_data.to(device)

# Define hyperparameters
HIDDEN_CHANNELS = 64  # Increased from 32
NUM_LAYERS = 3        # More layers
DROPOUT = 0.3         # Regularization
BATCH_SIZE = 128      # Increased batch size
EPOCHS = 60          # More epochs
LR = 0.001           # Lower learning rate
PATIENCE = 5         # Early stopping patience
CONV_TYPE = 'SAGE'   # Try different GNN layer types (GCN, SAGE, GAT)
USE_BATCH_NORM = True
USE_RESIDUAL = True
VALIDATION_SPLIT = 0.1  # 10% for validation

# Initialize model with improved architecture
print("\nTraining Graph Neural Network model...")
start_time = time.time()

model = EnhancedFoodDrugGNN(
    input_dim=graph_data.x.size(1), 
    hidden_channels=HIDDEN_CHANNELS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    conv_type=CONV_TYPE,
    use_batch_norm=USE_BATCH_NORM,
    residual=USE_RESIDUAL
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
criterion = torch.nn.BCEWithLogitsLoss()

# Learning rate scheduler
# Create scheduler - remove verbose flag for compatibility
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
print("LR scheduler initialized")

# Prepare training pairs
print("Preparing training pairs...")
train_pairs = []
train_labels = []

# Use more training pairs
sample_size = min(20000, len(smote_train))  # Increased from 5000
sample_data = smote_train.sample(sample_size, random_state=42)

for _, row in sample_data.iterrows():
    try:
        food_idx = food_mapping[int(row['food_encoded'])]
        drug_idx = drug_mapping[int(row['drug_encoded'])]
        train_pairs.append((food_idx, drug_idx))
        train_labels.append(float(row['has_interaction']))
    except KeyError:
        # Skip pairs not in the mapping
        continue

# Split into training and validation sets
train_val_split = int(len(train_pairs) * (1 - VALIDATION_SPLIT))
train_idx, val_idx = train_test_split(
    range(len(train_pairs)), 
    train_size=train_val_split,
    random_state=42,
    stratify=train_labels  # Ensure balanced split
)

val_pairs = [train_pairs[i] for i in val_idx]
val_labels = torch.tensor([train_labels[i] for i in val_idx], dtype=torch.float).to(device)
train_pairs = [train_pairs[i] for i in train_idx]
train_labels = [train_labels[i] for i in train_idx]

print(f"Training with {len(train_pairs)} pairs, validating with {len(val_pairs)} pairs")

# Training loop with optimization and early stopping
model.train()
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    epoch_start = time.time()
    
    # Shuffle training data
    indices = torch.randperm(len(train_pairs))
    shuffled_pairs = [train_pairs[i] for i in indices]
    shuffled_labels = [train_labels[i] for i in indices]
    
    # Process in batches
    epoch_loss = 0
    num_batches = (len(shuffled_pairs) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, len(shuffled_pairs))
        
        optimizer.zero_grad()
        
        # Forward pass - get node embeddings once per batch
        node_embeddings = model(graph_data.x, graph_data.edge_index)
        
        # Collect predictions for this batch
        batch_logits = []
        batch_labels = []
        
        for j in range(start_idx, end_idx):
            food_idx, drug_idx = shuffled_pairs[j]
            label = shuffled_labels[j]
            
            # Get prediction
            logit = model.predict_pair(node_embeddings, food_idx, drug_idx)
            batch_logits.append(logit)
            batch_labels.append(torch.tensor([label], device=device))
        
        # Stack all logits and compute loss
        batch_logits = torch.cat(batch_logits)
        batch_labels = torch.cat(batch_labels)
        
        loss = criterion(batch_logits, batch_labels)
        epoch_loss += loss.item()
        
        # Backpropagation
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
    
    # Calculate average loss for the epoch
    avg_loss = epoch_loss / num_batches
    
    # Evaluate on validation set
    val_metrics = evaluate_model(model, graph_data, val_pairs, val_labels, device)
    
    # Update learning rate based on validation loss
    scheduler.step(val_metrics['mse'])
    
    # Print progress
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, " 
              f"Val Acc: {val_metrics['accuracy']:.4f}, "
              f"Val MSE: {val_metrics['mse']:.4f}, "
              f"Time: {time.time() - epoch_start:.2f}s")
    
    # Check for early stopping
    if val_metrics['mse'] < best_val_loss:
        best_val_loss = val_metrics['mse']
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'food_drug_gnn_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# Load the best model for final evaluation
model.load_state_dict(torch.load('food_drug_gnn_model.pt'))
print(f"GNN training completed in {time.time() - start_time:.2f} seconds")
print(f"Best validation MSE: {best_val_loss:.4f}")

# Save the graph mappings for inference
with open('food_drug_graph_mappings.pkl', 'wb') as f:
    pickle.dump({
        'food_mapping': food_mapping,
        'drug_mapping': drug_mapping,
        'model_params': {
            'input_dim': graph_data.x.size(1),
            'hidden_channels': HIDDEN_CHANNELS,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
            'conv_type': CONV_TYPE,
            'use_batch_norm': USE_BATCH_NORM,
            'residual': USE_RESIDUAL
        }
    }, f)
print("Graph mappings and model parameters saved to 'food_drug_graph_mappings.pkl'")

# Part 5: Create an improved prediction function using both models
def predict_interaction(food_text, drug_text, rf_threshold=0.5, ensemble_weight=0.7, rebuild_graph=False):
    """
    Predict food-drug interaction using both Random Forest and GNN models
    
    Parameters:
    food_text: Text name of the food
    drug_text: Text name of the drug
    rf_threshold: Threshold for Random Forest prediction
    ensemble_weight: Weight for Random Forest (1-weight used for GNN)
    rebuild_graph: Whether to rebuild the graph (slower but ensures up-to-date data)
    
    Returns:
    Dictionary with prediction results
    """
    results = {
        "food": food_text,
        "drug": drug_text,
        "rf_prob": 0,
        "gnn_prob": 0,
        "ensemble_prob": 0,
        "predicted_interaction": False,
        "confidence": "Low",
        "status": "Unknown error"
    }
    
    # Load encoders
    try:
        with open('food_drug_encodings.pkl', 'rb') as f:
            encoders = pickle.load(f)
        
        # Check if food and drug exist in our data
        if food_text not in encoders['food_to_encoding']:
            results["status"] = f"Food '{food_text}' not found in training data"
            return results
            
        if drug_text not in encoders['drug_to_encoding']:
            results["status"] = f"Drug '{drug_text}' not found in training data"
            return results
            
        food_encoded = encoders['food_to_encoding'][food_text]
        drug_encoded = encoders['drug_to_encoding'][drug_text]
        
        # Random Forest prediction
        rf_model = joblib.load('food_drug_rf_model.pkl')
        rf_input = pd.DataFrame([[food_encoded, drug_encoded]], columns=['food_encoded', 'drug_encoded'])
        rf_prob = rf_model.predict_proba(rf_input)[0][1]  # Probability of class 1
        results["rf_prob"] = float(rf_prob)
        
        # Determine device
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        
        # Load the GNN model parameters and mappings
        with open('food_drug_graph_mappings.pkl', 'rb') as f:
            graph_mappings = pickle.load(f)
        
        # Get node indices
        try:
            food_idx = graph_mappings['food_mapping'][int(food_encoded)]
            drug_idx = graph_mappings['drug_mapping'][int(drug_encoded)]
            
            # Load graph data - either rebuild or use cached version
            if rebuild_graph:
                print("Rebuilding graph...")
                model_ready = pd.read_csv('/Users/sachidhoka/Desktop/Drug Food/food_drug_analysis/model_ready_food_drug_interactions.csv')
                graph_data, _, _ = build_graph_from_data(model_ready, max_pairs=10000)
            else:
                # For faster prediction, we can prepare this in memory
                # In a production setting, you might want to cache this
                model_ready = pd.read_csv('/Users/sachidhoka/Desktop/Drug Food/food_drug_analysis/model_ready_food_drug_interactions.csv')
                graph_data, _, _ = build_graph_from_data(model_ready, max_pairs=10000)
            
            # Create model with the saved parameters
            params = graph_mappings['model_params']
            model = EnhancedFoodDrugGNN(
                input_dim=params['input_dim'],
                hidden_channels=params['hidden_channels'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                conv_type=params['conv_type'],
                use_batch_norm=params['use_batch_norm'],
                residual=params['residual']
            ).to(device)
            
            # Load trained weights
            model.load_state_dict(torch.load('food_drug_gnn_model.pt'))
            model.eval()
            
            # Move data to device
            graph_data = graph_data.to(device)
            
            # Make prediction
            with torch.no_grad():
                node_embeddings = model(graph_data.x, graph_data.edge_index)
                gnn_prob = model.predict_pair(node_embeddings, food_idx, drug_idx).item()
                results["gnn_prob"] = float(gnn_prob)
        except KeyError as e:
            results["status"] = f"Food or drug not in GNN graph data: {str(e)}"
            results["gnn_prob"] = 0.5  # Neutral prediction
        except Exception as e:
            results["status"] = f"GNN prediction error: {str(e)}"
            results["gnn_prob"] = 0.5  # Neutral prediction
        
        # Ensemble prediction
        ensemble_prob = ensemble_weight * rf_prob + (1 - ensemble_weight) * results["gnn_prob"]
        results["ensemble_prob"] = float(ensemble_prob)
        results["predicted_interaction"] = ensemble_prob >= 0.5
        
        # Calculate confidence based on model agreement and prediction strength
        rf_decision = rf_prob >= 0.5
        gnn_decision = results["gnn_prob"] >= 0.5
        
        # Set confidence level
        if rf_decision == gnn_decision:
            # Models agree
            avg_deviation = abs(ensemble_prob - 0.5)
            if avg_deviation > 0.4:
                results["confidence"] = "High"
            elif avg_deviation > 0.2:
                results["confidence"] = "Medium"
            else:
                results["confidence"] = "Low"
        else:
            # Models disagree
            results["confidence"] = "Low"
        
        results["status"] = "Success"
        
    except Exception as e:
        results["status"] = f"Error: {str(e)}"
    
    return results

# Test the prediction with a few examples
print("\nTesting prediction functionality:")
examples = [
    ("grapefruit", "simvastatin"),  # Known interaction
    ("apple", "aspirin"),           # Likely no interaction
]

for food, drug in examples:
    print(f"\nPredicting interaction between {food} and {drug}:")
    result = predict_interaction(food, drug, rebuild_graph=False)
    print(f"Status: {result['status']}")
    if result['status'] == 'Success':
        print(f"RF Probability: {result['rf_prob']:.4f}")
        print(f"GNN Probability: {result['gnn_prob']:.4f}")
        print(f"Ensemble: {result['ensemble_prob']:.4f}")
        print(f"Predicted Interaction: {result['predicted_interaction']}")
        print(f"Confidence: {result['confidence']}")