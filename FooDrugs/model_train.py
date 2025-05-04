import pandas as pd
import numpy as np
import joblib
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Part 1: Load and prepare the SMOTE-balanced data
print("Loading SMOTE-balanced training data...")
smote_train = pd.read_csv('/Users/sachidhoka/Desktop/Drug Food/food_drug_analysis/smote_training_data.csv')
X_smote = smote_train[['food_encoded', 'drug_encoded']]
y_smote = smote_train['has_interaction']

# Part 2: Train Random Forest model as before
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_smote, y_smote)

# Save the Random Forest model
joblib.dump(rf_model, 'food_drug_rf_model.pkl')
print("Random Forest model saved to 'food_drug_rf_model.pkl'")

# Part 3: Create encoders for user input (as before)
print("Creating lookup encoders for user input...")
# Load the model_ready dataset with text and encoded values
model_ready = pd.read_csv('/Users/sachidhoka/Desktop/Drug Food/food_drug_analysis/model_ready_food_drug_interactions.csv')

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

# Part 4: Implement and train a Graph Neural Network (GNN)
print("Preparing data for Graph Neural Network...")

# Define a GNN model using PyTorch Geometric
class FoodDrugGNN(torch.nn.Module):
    def __init__(self, num_food_nodes, num_drug_nodes, hidden_channels):
        super(FoodDrugGNN, self).__init__()
        # GCN layers
        self.conv1 = GCNConv(2, hidden_channels)  # 2 features: node type (food/drug) and node ID
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Output layer
        self.out = nn.Linear(hidden_channels * 2, 1)  # Concatenate features of both nodes
    
    def forward(self, x, edge_index):
        # Apply GNN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        return x
    
    def predict_pair(self, x, edge_index, food_idx, drug_idx):
        # Get node embeddings
        node_embeddings = self.forward(x, edge_index)
        
        # Extract embeddings for the specific food and drug
        food_embedding = node_embeddings[food_idx]
        drug_embedding = node_embeddings[drug_idx]
        
        # Concatenate and pass through output layer
        pair_embedding = torch.cat([food_embedding, drug_embedding], dim=0)
        prediction = self.out(pair_embedding)
        
        return torch.sigmoid(prediction)

# Build a graph from the food-drug interaction data
def build_graph_from_data(smote_data, model_ready_data):
    # Get unique foods and drugs with their encodings
    # Make sure we're using integers for the encoded values
    unique_foods = set(int(id) for id in model_ready_data['food_encoded'].unique())
    unique_drugs = set(int(id) for id in model_ready_data['drug_encoded'].unique())
    
    # Create nodes for foods and drugs
    num_food_nodes = len(unique_foods)
    num_drug_nodes = len(unique_drugs)
    total_nodes = num_food_nodes + num_drug_nodes
    
    # Create node features (2 features: node type and node ID)
    x = torch.zeros((total_nodes, 2), dtype=torch.float)
    
    # Food nodes have type 0, drug nodes have type 1
    # First num_food_nodes are foods, the rest are drugs
    food_mapping = {food_id: idx for idx, food_id in enumerate(unique_foods)}
    drug_mapping = {drug_id: idx + num_food_nodes for idx, drug_id in enumerate(unique_drugs)}
    
    # Set node features
    food_list = list(unique_foods)
    drug_list = list(unique_drugs)
    max_food_id = max(food_list) if food_list else 1  # Avoid division by zero
    max_drug_id = max(drug_list) if drug_list else 1  # Avoid division by zero
    
    for i in range(num_food_nodes):
        x[i, 0] = 0  # Node type: food
        x[i, 1] = float(food_list[i]) / max_food_id if max_food_id > 0 else 0  # Normalized node ID
    
    for i in range(num_drug_nodes):
        x[i + num_food_nodes, 0] = 1  # Node type: drug
        x[i + num_food_nodes, 1] = float(drug_list[i]) / max_drug_id if max_drug_id > 0 else 0  # Normalized node ID
    
    # Create edges based on interactions in training data
    edge_list = []
    edge_attr = []  # 1 for interaction, 0 for no interaction
    
    # Add edges for all food-drug pairs in the dataset
    for _, row in model_ready_data.iterrows():
        food_idx = food_mapping[row['food_encoded']]
        drug_idx = drug_mapping[row['drug_encoded']]
        
        # Add edge from food to drug
        edge_list.append([food_idx, drug_idx])
        edge_attr.append(row['has_interaction'])
        
        # Add edge from drug to food (making it undirected)
        edge_list.append([drug_idx, food_idx])
        edge_attr.append(row['has_interaction'])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), food_mapping, drug_mapping

print("Building graph from interaction data...")
graph_data, food_mapping, drug_mapping = build_graph_from_data(smote_train, model_ready)

# Train the GNN model
print("Training Graph Neural Network model...")
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = FoodDrugGNN(len(food_mapping), len(drug_mapping), hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Use BCEWithLogitsLoss instead of BCELoss to handle the sigmoid internally
criterion = torch.nn.BCEWithLogitsLoss()

graph_data = graph_data.to(device)

# Prepare data for batch training
all_pairs = []
all_targets = []

for _, row in smote_train.iterrows():
    food_idx = food_mapping[row['food_encoded']]
    drug_idx = drug_mapping[row['drug_encoded']]
    all_pairs.append((food_idx, drug_idx))
    all_targets.append(row['has_interaction'])

all_targets = torch.tensor(all_targets, dtype=torch.float).to(device)

# Training loop with batch processing
model.train()
total_epochs = 200
batch_size = 32  # Adjust based on your dataset size
for epoch in range(total_epochs):
    optimizer.zero_grad()
    
    # Forward pass - get node embeddings
    node_embeddings = model(graph_data.x, graph_data.edge_index)
    
    # Process in batches
    total_loss = 0
    num_batches = (len(all_pairs) + batch_size - 1) // batch_size  # Ceiling division
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(all_pairs))
        
        batch_predictions = []
        for j in range(start_idx, end_idx):
            food_idx, drug_idx = all_pairs[j]
            food_embedding = node_embeddings[food_idx]
            drug_embedding = node_embeddings[drug_idx]
            pair_embedding = torch.cat([food_embedding, drug_embedding], dim=0)
            pred = model.out(pair_embedding)
            batch_predictions.append(pred)
        
        # Stack predictions into a tensor
        batch_predictions = torch.cat(batch_predictions)
        batch_targets = all_targets[start_idx:end_idx]
        
        # Calculate loss
        loss = criterion(batch_predictions, batch_targets)
        total_loss += loss.item()
        
        # Backward pass
        loss.backward(retain_graph=(i < num_batches-1))  # Retain graph except for last batch
    
    # Step optimizer
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{total_epochs}, Loss: {total_loss/num_batches:.4f}")

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

# Part 5: Create a unified prediction function that uses both models
def predict_interaction(food_text, drug_text, rf_threshold=0.5, gnn_threshold=0.5, ensemble_weight=0.5):
    """
    Predict food-drug interaction using both Random Forest and GNN models
    
    Parameters:
    food_text: Text name of the food
    drug_text: Text name of the drug
    rf_threshold: Threshold for Random Forest prediction
    gnn_threshold: Threshold for GNN prediction  
    ensemble_weight: Weight for Random Forest (1-weight used for GNN)
    
    Returns:
    is_interaction: Boolean indicating predicted interaction
    rf_prob: Random Forest probability
    gnn_prob: GNN probability
    ensemble_prob: Weighted ensemble probability
    """
    # Load encoders
    with open('food_drug_encodings.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    # Get encodings
    try:
        food_encoded = encoders['food_to_encoding'][food_text]
        drug_encoded = encoders['drug_to_encoding'][drug_text]
    except KeyError:
        return None, 0, 0, 0, "Food or drug not found in training data"
    
    # Random Forest prediction
    rf_model = joblib.load('food_drug_rf_model.pkl')
    rf_input = pd.DataFrame([[food_encoded, drug_encoded]], columns=['food_encoded', 'drug_encoded'])
    rf_prob = rf_model.predict_proba(rf_input)[0][1]  # Probability of class 1
    
    # GNN prediction
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load the GNN model
    model = FoodDrugGNN(0, 0, hidden_channels=64).to(device)  # Placeholder values
    model.load_state_dict(torch.load('food_drug_gnn_model.pt'))
    model.eval()
    
    # Load graph mappings
    with open('food_drug_graph_mappings.pkl', 'rb') as f:
        graph_mappings = pickle.load(f)
    
    # Get node indices
    try:
        food_idx = graph_mappings['food_mapping'][food_encoded]
        drug_idx = graph_mappings['drug_mapping'][drug_encoded]
        
        # Load graph data - we need to load these data files again
        smote_train = pd.read_csv('/Users/sachidhoka/Desktop/Drug Food/food_drug_analysis/smote_training_data.csv')
        model_ready = pd.read_csv('/Users/sachidhoka/Desktop/Drug Food/food_drug_analysis/model_ready_food_drug_interactions.csv')
        graph_data, _, _ = build_graph_from_data(smote_train, model_ready)
        graph_data = graph_data.to(device)
        
        # Get prediction
        with torch.no_grad():
            node_embeddings = model(graph_data.x, graph_data.edge_index)
            food_embedding = node_embeddings[food_idx]
            drug_embedding = node_embeddings[drug_idx]
            pair_embedding = torch.cat([food_embedding, drug_embedding], dim=0)
            logit = model.out(pair_embedding)
            gnn_prob = torch.sigmoid(logit).item()
    except KeyError:
        gnn_prob = 0
        return None, rf_prob, gnn_prob, 0, "Food or drug not found in GNN data"
    
    # Ensemble prediction
    ensemble_prob = ensemble_weight * rf_prob + (1 - ensemble_weight) * gnn_prob
    
    # Make final prediction
    is_interaction = ensemble_prob >= 0.5
    
    return is_interaction, rf_prob, gnn_prob, ensemble_prob, "Success"

# Example of using the combined prediction function
if __name__ == "__main__":
    print("\nTesting combined prediction function:")
    food = "grapefruit"  # Example food
    drug = "simvastatin"  # Example drug
    
    result, rf_prob, gnn_prob, ensemble_prob, msg = predict_interaction(food, drug)
    
    if msg == "Success":
        print(f"Food: {food}, Drug: {drug}")
        print(f"Random Forest probability: {rf_prob:.4f}")
        print(f"GNN probability: {gnn_prob:.4f}")
        print(f"Ensemble probability: {ensemble_prob:.4f}")
        print(f"Predicted interaction: {result}")
    else:
        print(f"Error: {msg}")