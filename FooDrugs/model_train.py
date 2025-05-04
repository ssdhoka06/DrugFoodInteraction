

import pandas as pd
import numpy as np
import joblib
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from sklearn.ensemble import RandomForestClassifier
import os
import time

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

# Define a simplified GNN model using PyTorch Geometric
class FoodDrugGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels, output_dim=1):
        super(FoodDrugGNN, self).__init__()
        # Simplified architecture with fewer layers
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = nn.Linear(hidden_channels * 2, output_dim)
    
    def forward(self, x, edge_index):
        # Apply GNN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def predict_pair(self, node_embeddings, food_idx, drug_idx):
        # Extract embeddings for the specific food and drug
        food_embedding = node_embeddings[food_idx]
        drug_embedding = node_embeddings[drug_idx]
        
        # Concatenate and pass through output layer
        pair_embedding = torch.cat([food_embedding, drug_embedding], dim=0)
        prediction = self.out(pair_embedding)
        
        return torch.sigmoid(prediction)

# Build a smaller graph from the food-drug interaction data
def build_graph_from_data(model_ready_data, max_pairs=10000):
    print("Building graph from interaction data...")
    
    # Get unique foods and drugs with their encodings
    # Ensure integers for node IDs
    unique_foods = sorted(set(int(id) for id in model_ready_data['food_encoded'].unique()))
    unique_drugs = sorted(set(int(id) for id in model_ready_data['drug_encoded'].unique()))
    
    print(f"Number of unique foods: {len(unique_foods)}")
    print(f"Number of unique drugs: {len(unique_drugs)}")
    
    # Create nodes for foods and drugs
    num_food_nodes = len(unique_foods)
    num_drug_nodes = len(unique_drugs)
    total_nodes = num_food_nodes + num_drug_nodes
    
    # Create node features (2 features: node type and node ID)
    x = torch.zeros((total_nodes, 2), dtype=torch.float)
    
    # Food nodes have type 0, drug nodes have type 1
    food_mapping = {food_id: idx for idx, food_id in enumerate(unique_foods)}
    drug_mapping = {drug_id: idx + num_food_nodes for idx, drug_id in enumerate(unique_drugs)}
    
    # Set node features
    max_food_id = max(unique_foods) if unique_foods else 1
    max_drug_id = max(unique_drugs) if unique_drugs else 1
    
    for i, food_id in enumerate(unique_foods):
        x[i, 0] = 0  # Node type: food
        x[i, 1] = float(food_id) / max_food_id  # Normalized node ID
    
    for i, drug_id in enumerate(unique_drugs):
        x[i + num_food_nodes, 0] = 1  # Node type: drug
        x[i + num_food_nodes, 1] = float(drug_id) / max_drug_id  # Normalized node ID
    
    # Create edges - limit to max_pairs to reduce memory usage
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
        except KeyError as e:
            # Skip pairs with missing mappings
            continue
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    
    print(f"Graph created with {x.shape[0]} nodes and {len(edge_indices)} edges")
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), food_mapping, drug_mapping

# Build the graph with limited size
graph_data, food_mapping, drug_mapping = build_graph_from_data(model_ready, max_pairs=5000)

# Determine device - try MPS first, fallback to CPU if needed
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS (Apple GPU)")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Train the GNN model with performance optimizations
print("\nTraining Graph Neural Network model...")
start_time = time.time()

# Move data to device
graph_data = graph_data.to(device)

# Initialize model
model = FoodDrugGNN(input_dim=2, hidden_channels=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.BCEWithLogitsLoss()

# Prepare training pairs
print("Preparing training pairs...")
train_pairs = []
train_labels = []

# Limit to 5000 training pairs for faster training
sample_size = min(5000, len(smote_train))
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

train_labels = torch.tensor(train_labels, dtype=torch.float).to(device)
print(f"Training with {len(train_pairs)} pairs")

# Training loop with optimization
model.train()
batch_size = 64  # Larger batch size for better GPU utilization
epochs = 50      # Fewer epochs for faster training

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Forward pass - get node embeddings
    node_embeddings = model(graph_data.x, graph_data.edge_index)
    
    # Process in batches
    epoch_loss = 0
    num_batches = (len(train_pairs) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(train_pairs))
        
        # Collect predictions for this batch
        batch_logits = []
        
        for j in range(start_idx, end_idx):
            food_idx, drug_idx = train_pairs[j]
            food_embed = node_embeddings[food_idx]
            drug_embed = node_embeddings[drug_idx]
            
            # Concatenate embeddings
            pair_embed = torch.cat([food_embed, drug_embed])
            logit = model.out(pair_embed)
            batch_logits.append(logit)
        
        # Stack all logits and compute loss
        batch_logits = torch.cat(batch_logits)
        batch_labels = train_labels[start_idx:end_idx]
        
        loss = criterion(batch_logits, batch_labels)
        epoch_loss += loss.item()
        
        # Backpropagation
        loss.backward(retain_graph=(i < num_batches-1))
    
    # Update weights
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

print(f"GNN training completed in {time.time() - start_time:.2f} seconds")

# Save the GNN model
torch.save(model.state_dict(), 'food_drug_gnn_model.pt')
print("GNN model saved to 'food_drug_gnn_model.pt'")

# Save the graph mappings for inference
with open('food_drug_graph_mappings.pkl', 'wb') as f:
    pickle.dump({
        'food_mapping': food_mapping,
        'drug_mapping': drug_mapping
    }, f)
print("Graph mappings saved to 'food_drug_graph_mappings.pkl'")

# Part 5: Create an optimized prediction function using both models
def predict_interaction(food_text, drug_text, rf_threshold=0.5, ensemble_weight=0.7):
    """
    Predict food-drug interaction using both Random Forest and GNN models
    
    Parameters:
    food_text: Text name of the food
    drug_text: Text name of the drug
    rf_threshold: Threshold for Random Forest prediction
    ensemble_weight: Weight for Random Forest (1-weight used for GNN)
    
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
        "status": "Unknown error"
    }
    
    # Load encoders
    try:
        with open('food_drug_encodings.pkl', 'rb') as f:
            encoders = pickle.load(f)
        
        # Get encodings
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
        
        # Load the GNN model
        with open('food_drug_graph_mappings.pkl', 'rb') as f:
            graph_mappings = pickle.load(f)
        
        model = FoodDrugGNN(input_dim=2, hidden_channels=32).to(device)
        model.load_state_dict(torch.load('food_drug_gnn_model.pt'))
        model.eval()
        
        # Get node indices
        try:
            food_idx = graph_mappings['food_mapping'][int(food_encoded)]
            drug_idx = graph_mappings['drug_mapping'][int(drug_encoded)]
            
            # Load graph data
            graph_data, _, _ = build_graph_from_data(model_ready, max_pairs=5000)
            graph_data = graph_data.to(device)
            
            # Make prediction
            with torch.no_grad():
                node_embeddings = model(graph_data.x, graph_data.edge_index)
                gnn_prob = model.predict_pair(node_embeddings, food_idx, drug_idx).item()
                results["gnn_prob"] = float(gnn_prob)
        except KeyError:
            results["status"] = "Food or drug not in GNN graph data"
            results["gnn_prob"] = 0.5  # Neutral prediction
        
        # Ensemble prediction
        ensemble_prob = ensemble_weight * rf_prob + (1 - ensemble_weight) * results["gnn_prob"]
        results["ensemble_prob"] = float(ensemble_prob)
        results["predicted_interaction"] = ensemble_prob >= 0.5
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
    result = predict_interaction(food, drug)
    print(f"Status: {result['status']}")
    if result['status'] == 'Success':
        print(f"RF Probability: {result['rf_prob']:.4f}")
        print(f"GNN Probability: {result['gnn_prob']:.4f}")
        print(f"Ensemble: {result['ensemble_prob']:.4f}")
        print(f"Predicted Interaction: {result['predicted_interaction']}")