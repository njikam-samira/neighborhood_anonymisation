import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import numpy as np
import time
from typing import List, Dict
from sklearn.metrics import accuracy_score, recall_score, f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GATClassifier(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, heads):
        super().__init__()
        # Première couche: Extraction de motifs locaux
        self.conv1 = GATConv(num_features, hidden_dim, heads=heads, dropout=0.55) 
        # Couche intermédiaire (sans changement de dimension)
        self.conv2 = GATConv(hidden_dim*heads, hidden_dim, heads=1, dropout=0.55)
        # Head de classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ELU(),
            nn.Dropout(0.55),
            nn.Linear(hidden_dim//2, num_classes)
        )
        
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.55, training=self.training)
        x = F.elu(self.conv2(x, edge_index))  # Maintient la dimension
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

def train_model(model: nn.Module, data: Data, k: int, epochs: int):
    # Configuration optimisée
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=20)
    
    best_acc = 0.0
    best_model = None
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out, data.y)
        
        # Conservation de votre pénalité k-anonymité
        with torch.no_grad():
            _, preds = torch.max(out, dim=1)
            unique, counts = torch.unique(preds, return_counts=True)
            penalty = (torch.sum(F.relu(k - counts).float())/ data.num_nodes) * 0.1
        
        total_loss = loss + penalty
        total_loss.backward()
        
        # Ajout du gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Évaluation
        with torch.no_grad():
            model.eval()
            preds = model(data.x, data.edge_index).argmax(dim=1)
            acc = (preds == data.y).float().mean().item()
            scheduler.step(acc)
            
            if acc > best_acc:
                best_acc = acc
                best_model = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch}: Loss={total_loss.item():.4f}, Acc={acc*100:.2f}% LR={optimizer.param_groups[0]["lr"]:.6f}')
            
            # Early stopping si pas d'amélioration après 50 epochs
            if patience_counter > 100:
                print(f"Early stopping at epoch {epoch}")
                break
            
    
    # Chargement du meilleur modèle
    model.load_state_dict(best_model)

    end_time = time.time() 
    training_time = end_time - start_time
    print(f"Temps d'entraînement: {training_time:.2f} secondes")

    return best_acc

def classify_gat(clusters, data: Data, k: int, epochs: int) -> np.ndarray:
    num_features = data.x.size(1)
    model = GATClassifier(
        num_features=num_features,
        hidden_dim=128,
        num_classes=len(clusters),
        heads=8
    ).to(device)
    
    data = data.to(device)
    best_acc = train_model(model, data, k, epochs)
    
    with torch.no_grad():
        model.eval()
        preds = model(data.x, data.edge_index).argmax(dim=1).cpu().numpy()
        y_true = data.y.cpu().numpy()
        
        # Calcul des métriques
        accuracy = accuracy_score(y_true, preds)
        recall = recall_score(y_true, preds, average='macro', zero_division=0)
        f1 = f1_score(y_true, preds, average='macro', zero_division=0)


    print(f"Meilleure accuracy pendant entraînement: {best_acc*100:.2f}%")
    print(f"Accuracy finale: {accuracy*100:.2f}%")
    print(f"Rappel: {recall*100:.2f}%")
    print(f"F-measure: {f1*100:.2f}%")

    unique, counts = np.unique(preds, return_counts=True)
    print("Distribution des clusters:", dict(zip(unique, counts)))
    
    return preds, accuracy, recall, f1