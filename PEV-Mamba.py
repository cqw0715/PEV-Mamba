import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, average_precision_score, classification_report
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import t
import esm
from mamba_ssm import Mamba
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import warnings
from transformers import T5Tokenizer, T5EncoderModel

warnings.filterwarnings('ignore')

# Feature cache paths
ESM_CACHE_PATH = 'esm_features_90__1587.pkl'
PROTT5_CACHE_PATH = 'prot_t5_features_90__1587.pkl'

# Best hyperparameters
BEST_HPARAMS = {
    'd_model': 256,
    'dropout_rate': 0.3,
    'num_mamba_blocks': 3,
    'max_lr_sup': 0.0005418282319533242,
    'weight_decay': 3.823475224675187e-06,
    'batch_size': 32,
    'gamma_focal_loss': 2.0,
    'ssl_reg_weight': 0.014290255329034685,
    'balance_ratio_msmote': 1.0
}

# Early stopping mechanism
class EarlyStopping:
    """Used to save the best model and stop training early"""
    def __init__(self, patience=10, delta=0, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_f1_max = -np.inf
        self.path = path

    def __call__(self, val_f1, model):
        score = val_f1
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_f1, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_f1, model)
            self.counter = 0

    def save_checkpoint(self, val_f1, model):
        """Save model when validation F1-Macro improves"""
        if self.verbose:
            print(f'Validation F1-Macro improved ({self.val_f1_max:.6f} --> {val_f1:.6f}). Saving model ...')
        self.val_f1_max = val_f1

# Random seed setting
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# Data loading and preprocessing
try:
    data = pd.read_csv('90%_1587.csv')
except FileNotFoundError:
    print("File '90%_1587.csv' not found")
    exit()

sequences = data['Sequence'].tolist()
# PEV: Subtract 1 from labels to start from 0
labels = data['Label'].values - 1 

# Non_PEV: Labels start from 0, no need to subtract
# labels = data['Label'].values

print(f"Data distribution: {np.bincount(labels)}")


# --- Feature Extraction Module ---

# ESM-2 feature extraction
def load_esm_model():
    """Load pre-trained ESM-2 model and batch converter"""
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D() 
    batch_converter = alphabet.get_batch_converter()
    return model.eval(), batch_converter

def get_enhanced_esm_features(sequences, cache_path=ESM_CACHE_PATH, batch_size=1): 
    """Extract multi-layer (10, 24, 28, 33) mean and max pooling features from ESM-2"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if os.path.exists(cache_path):
        print(f"Successfully loaded cached ESM features from {cache_path}...")
        with open(cache_path, 'rb') as f:
            features = pickle.load(f)
        return features
    print(f"Cached file {cache_path} not found. Starting ESM feature extraction...")
    
    try:
        esm_model, esm_batch_converter = load_esm_model()
    except Exception as e:
        raise RuntimeError(f"ESM model loading failed: {e}. Cached file not found and model cannot be loaded.")

    features = []
    layers = [10, 24, 28, 33] 
    
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        batch_data = [(str(idx), seq) for idx, seq in enumerate(batch)]
        
        batch_labels, batch_strs, batch_tokens = esm_batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)
        esm_model.to(device)
        
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=layers, return_contacts=False)
            
            for seq_idx in range(batch_tokens.size(0)):
                seq_features = []
                seq_len = (batch_tokens[seq_idx] != esm_model.alphabet.padding_idx).sum() - 2 
                
                for layer in layers:
                    layer_rep = results["representations"][layer][seq_idx, 1:seq_len+1] 
                    
                    mean_pool = layer_rep.mean(0).cpu().numpy()
                    max_pool = layer_rep.max(0)[0].cpu().numpy()
                    
                    seq_features.extend([mean_pool, max_pool])
                
                combined_features = np.concatenate(seq_features)
                features.append(combined_features)
        
        if (i + len(batch)) % 100 == 0:
            print(f"ESM processed {i + len(batch)}/{len(sequences)} sequences")
    
    features = np.array(features)
    print(f"Final ESM feature dimensions: {features.shape}")
    
    # Save features
    with open(cache_path, 'wb') as f:
        pickle.dump(features, f)
        
    return features

# ProtT5 feature extractor 
class ProtT5FeatureExtractor:
    def __init__(self):
        if T5Tokenizer is None or T5EncoderModel is None:
            raise RuntimeError("Failed to load Transformers library required for ProtT5")      
        self.tokenizer, self.model, self.device = self._load_model()
        
    def _load_model(self):
        """Load ProtT5 model from local directory"""
        model_dir = "./prot_t5_local" 
        try:
            tokenizer = T5Tokenizer.from_pretrained(model_dir)
            model = T5EncoderModel.from_pretrained(model_dir)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            model.eval()
            print(f"Successfully loaded ProtT5 model from local to {device}")
            return tokenizer, model, device
        except Exception as e:
            print(f"Failed to load ProtT5 model: {e}")
            raise
        
    def extract_features(self, sequences, cache_path=PROTT5_CACHE_PATH, batch_size=1): 
        """Extract ProtT5 mean pooling features"""
        if cache_path and os.path.exists(cache_path):
            print(f"Loading ProtT5 features from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
            
        print(f"Cached file {cache_path} not found. Starting ProtT5 feature extraction...")
        features = []
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            batch_with_spaces = [" ".join(list(seq.upper())) for seq in batch]
            
            try:
                ids = self.tokenizer(batch_with_spaces,
                                     return_tensors='pt',
                                     padding=True,
                                     truncation=True)
                
                ids = {key: val.to(self.device) for key, val in ids.items()}
                
                with torch.no_grad():
                    outputs = self.model(input_ids=ids['input_ids'],
                                         attention_mask=ids['attention_mask'])
                    
                    last_hidden_states = outputs.last_hidden_state
                    
                    for j in range(last_hidden_states.size(0)):
                        seq_len = (ids['attention_mask'][j] == 1).sum().item() - 1 
                        sequence_embeddings = last_hidden_states[j, :seq_len, :]
                        
                        if sequence_embeddings.size(0) > 0:
                            features.append(sequence_embeddings.mean(dim=0).cpu().numpy())
                        else:
                            features.append(np.zeros(last_hidden_states.size(-1))) 
                            
            except Exception as e:
                print(f"Error processing ProtT5 batch {i//batch_size}: {e}")
                continue
                
            if (i + len(batch)) % 100 == 0:
                print(f"ProtT5 completed {min(i+len(batch), len(sequences))}/{len(sequences)}")
                
        features = np.array(features)
        
        if cache_path:
            with open(cache_path, 'wb') as f:
                pickle.dump(features, f)
            print(f"ProtT5 features cached to: {cache_path}")
            
        return features

# Model and data classes

class ProteinDataset(Dataset):
    """Standard protein feature dataset"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X) 
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MambaClassifier(nn.Module):
    """
    Mamba block-based classifier with self-supervised learning (SSL) reconstruction head.
    Processes input feature vectors (treated as sequences of length 1) through Mamba modules for feature transformation.
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        
        d_model = BEST_HPARAMS['d_model']
        dropout_rate = BEST_HPARAMS['dropout_rate']
        num_mamba_blocks = BEST_HPARAMS['num_mamba_blocks']
        
        # Feature preprocessing
        self.preprocess = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Mamba module list: process feature sequences (length 1)
        self.mamba_blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2, dt_rank='auto')
            for _ in range(num_mamba_blocks) # Use optimal number of blocks
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )
        
        # Self-supervised learning head - for feature reconstruction
        self.ssl_head = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, input_dim) 
        )
        
        # Residual connection: map original input dimension to d_model
        self.residual = nn.Linear(input_dim, d_model)

    def forward(self, x, return_representation=False):
        
        x_pre = self.preprocess(x)
        representation = x_pre 
        representation_mamba = representation.unsqueeze(1) # [B, D] -> [B, 1, D]
        
        for mamba_block in self.mamba_blocks:
            representation_mamba = representation_mamba + mamba_block(representation_mamba)
            
        representation = representation_mamba.squeeze(1) # [B, 1, D] -> [B, D]
        x_res = self.residual(x)
        representation = representation + x_res # Residual connection
        
        if return_representation:
            return representation

        logits = self.classifier(representation)
        
        reconstruction = self.ssl_head(representation) 
        
        return logits, reconstruction 

class SSLModel(nn.Module):
    """Wrapper model for self-supervised pre-training, containing only Mamba core and SSL head"""
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        self.ssl_head = classifier.ssl_head

    def forward(self, x, return_representation=False):
        
        x_pre = self.classifier.preprocess(x)
        representation = x_pre
        representation_mamba = representation.unsqueeze(1)
        
        for mamba_block in self.classifier.mamba_blocks:
            representation_mamba = representation_mamba + mamba_block(representation_mamba) 
            
        representation = representation_mamba.squeeze(1)

        x_res = self.classifier.residual(x)
        representation = representation + x_res
        
        if return_representation:
            return representation
            
        reconstruction = self.ssl_head(representation)
        return reconstruction

class FeatureAugmentationDataset(Dataset):
    """Dataset for SSL training, providing original and noisy features"""
    def __init__(self, X):
        self.X = torch.FloatTensor(X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        original = self.X[idx]
        # Simple noise addition as data augmentation
        augmented = original + torch.randn_like(original) * 0.05 
        return original, augmented

# Training helper functions

# Self-supervised pre-training 
def pretrain_ssl(model, features_scaled, input_dim, device, num_epochs=50, batch_size=BEST_HPARAMS['batch_size']): # Use optimal batch_size
    
    print("Starting self-supervised pre-training...")
    
    ssl_dataset = FeatureAugmentationDataset(features_scaled)
    ssl_loader = DataLoader(ssl_dataset, batch_size=batch_size, shuffle=True) 
    ssl_model = SSLModel(model).to(device)
    
    mse_loss = nn.MSELoss()
    cosine_loss = nn.CosineEmbeddingLoss() # For contrastive learning
    
    ssl_optimizer = torch.optim.AdamW(ssl_model.parameters(), lr=1e-4, weight_decay=1e-5) 
    ssl_scheduler = torch.optim.lr_scheduler.OneCycleLR(ssl_optimizer, max_lr=5e-4, steps_per_epoch=len(ssl_loader),epochs=num_epochs)    
    ssl_model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        mse_loss_total = 0
        cosine_loss_total = 0
        
        for original, augmented in ssl_loader:
            original = original.to(device)
            augmented = augmented.to(device)  
            ssl_optimizer.zero_grad()

            # 1. Reconstruction loss
            reconstruction = ssl_model(original)
            mse_batch = mse_loss(reconstruction, original)
            
            # 2. Contrastive loss (original feature vs noisy feature representation)
            repr_original = ssl_model(original, return_representation=True)
            repr_augmented = ssl_model(augmented, return_representation=True)

            target = torch.ones(repr_original.size(0)).to(device)
            cosine_batch = cosine_loss(repr_original, repr_augmented, target)
            
            # Pre-training weights 0.7/0.3
            ssl_loss = mse_batch * 0.7 + cosine_batch * 0.3 
            ssl_loss.backward()
            torch.nn.utils.clip_grad_norm_(ssl_model.parameters(), max_norm=1.0)
            ssl_optimizer.step()
            
            epoch_loss += ssl_loss.item()
            mse_loss_total += mse_batch.item()
            cosine_loss_total += cosine_batch.item()
            
            ssl_scheduler.step()
            
        if epoch % 10 == 0:
            print(f'SSL Epoch {epoch}: Total Loss: {epoch_loss/len(ssl_loader):.4f}, '
                          f'MSE: {mse_loss_total/len(ssl_loader):.4f}, '
                          f'Cosine: {cosine_loss_total/len(ssl_loader):.4f}')

    print("Self-supervised pre-training completed!")

    model.load_state_dict(ssl_model.classifier.state_dict()) 
    return model

# MSMOTEBoost oversampling
def msmote_oversampling(X, y, k1=5, k2=9, balance_ratio=BEST_HPARAMS['balance_ratio_msmote']): # Use optimal balance_ratio
    """
    Improved MSMOTE oversampling algorithm designed to mitigate marginalization of minority class samples.
    k1 is used to select minority class neighbors, k2 is used for boundary evaluation (WON value).
    """
    
    counts = Counter(y)
    classes = sorted(list(counts.keys()))
    J = len(classes)
    
    majority_class = max(counts, key=counts.get)
    N_majority = counts[majority_class]
    
    N_target = int(N_majority * balance_ratio) 
    
    minority_classes = [c for c in classes if counts[c] < N_target and counts[c] > 0]

    X_synthetic = []
    y_synthetic = []
    
    # For boundary evaluation (WON value) calculation
    knn_all = NearestNeighbors(n_neighbors=k2 + 1)
    knn_all.fit(X)
    
    for c in minority_classes:
        X_min = X[y == c]
        y_min = y[y == c]
        N_min = len(X_min)
        
        N_to_generate = N_target - N_min
        
        if N_to_generate <= 0 or N_min < 2:
            continue

        N_oversample_per_instance = int(N_to_generate / N_min)
        remainder = N_to_generate % N_min
        
        indices_to_oversample = np.arange(N_min)

        # For selecting synthesis targets within minority class K-nearest neighbors
        knn_minority = NearestNeighbors(n_neighbors=min(k1 + 1, N_min))
        knn_minority.fit(X_min)

        for idx in indices_to_oversample:
            x_i = X_min[idx]
            
            # Find k1 neighbors of x_i within the minority class
            d_min, idx_min = knn_minority.kneighbors(x_i.reshape(1, -1), return_distance=True)
            S_nn_indices = idx_min.flatten()[1:] 

            if len(S_nn_indices) == 0:
                continue 
                
            S_nn_X = X_min[S_nn_indices]

            WON_values = []
            
            # Calculate WON value (Weighted Overlap Neighborhood) for each neighbor x_n
            for x_n in S_nn_X:
                distances_n, indices_n = knn_all.kneighbors(x_n.reshape(1, -1), return_distance=True)
                indices_n = indices_n.flatten()[1:]
                
                y_k2_neighbors_n = y[indices_n]
                
                N_k2_min_n = np.sum(y_k2_neighbors_n == c) # Number of minority class samples in k2 neighbors
                
                N_k2_classes_n = len(np.unique(y_k2_neighbors_n)) # Total number of classes in k2 neighbors
                
                overlap_n = N_k2_classes_n / J # Neighbor class diversity
                
                SC_prime_n = (k2 + 1) / (N_k2_min_n + 1e-6) # Minority class concentration score (avoid division by zero)

                won_n = overlap_n * SC_prime_n
                WON_values.append(won_n)
            
            if not WON_values:
                continue
                
            # Select neighbor x_k with highest WON value as synthesis target
            x_k_index_in_S_nn = np.argmax(WON_values)
            x_k = S_nn_X[x_k_index_in_S_nn]
            
            # Number of new samples to generate
            num_to_synthesize = N_oversample_per_instance + (1 if idx < remainder else 0)

            # SMOTE interpolation
            for _ in range(num_to_synthesize):
                gap = np.random.rand()
                x_gen = x_i + gap * (x_k - x_i) 
                
                X_synthetic.append(x_gen)
                y_synthetic.append(c)

    X_resampled = np.vstack([X, np.array(X_synthetic)])
    y_resampled = np.hstack([y, np.array(y_synthetic)])
    
    return X_resampled, y_resampled

def apply_smote_oversampling(X_train, y_train): 
    """Wrapper for MSMOTE call and logging"""
    print(f"Original training set distribution: {np.bincount(y_train)}")
    
    X_resampled, y_resampled = msmote_oversampling(X_train, y_train, k1=5, k2=9, balance_ratio=BEST_HPARAMS['balance_ratio_msmote']) 
    
    print(f"Training set distribution after MSMOTE: {np.bincount(y_resampled)}")
    return X_resampled, y_resampled

# Focal Loss function
class FocalLoss(nn.Module):
    """
    Focal Loss with class weights for handling class imbalance.
    L_f = - \alpha_t (1 - p_t)^\gamma \log(p_t)
    """
    def __init__(self, alpha=None, gamma=BEST_HPARAMS['gamma_focal_loss'], reduction='mean'): 
        super().__init__()
        self.alpha = alpha 
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        if self.alpha is not None and self.alpha.device != input.device:
             self.alpha = self.alpha.to(input.device)
             
        # Cross Entropy Loss (with class weights)
        ce_loss = F.cross_entropy(input, target, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_and_evaluate(model, X_train, y_train, X_val, y_val, raw_train_labels, num_classes, device, num_epochs=100, patience=10): 
    """Main loop for model training and validation evaluation"""
    
    batch_size = BEST_HPARAMS['batch_size']
    max_lr_sup = BEST_HPARAMS['max_lr_sup']
    weight_decay = BEST_HPARAMS['weight_decay']
    ssl_reg_weight = BEST_HPARAMS['ssl_reg_weight']
    gamma_focal_loss = BEST_HPARAMS['gamma_focal_loss']

    train_dataset = ProteinDataset(X_train, y_train)
    val_dataset = ProteinDataset(X_val, y_val)
    
    # Class weights and weighted sampler
    raw_class_weights_resampled = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_resampled = torch.FloatTensor(raw_class_weights_resampled).to(device)
    
    target_counts = Counter(y_train)
    total_samples = len(y_train)
    sample_weights = np.array([1.0 / target_counts[label] for label in y_train])
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total_samples,
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Focal Loss with class weights
    criterion = FocalLoss(alpha=class_weights_resampled, gamma=gamma_focal_loss).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr_sup, steps_per_epoch=len(train_loader), epochs=num_epochs) 
    
    best_f1 = -np.inf
    best_metrics = None
    
    early_stopping = EarlyStopping(patience=patience, verbose=False)
    
    for epoch in range(num_epochs):

        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            logits, reconstruction = model(X_batch) 
            
            class_loss = criterion(logits, y_batch)
            # Add reconstruction regularization term
            reconstruction_loss = F.mse_loss(reconstruction, X_batch)
            
            loss = class_loss + reconstruction_loss * ssl_reg_weight 
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
            scheduler.step() 
            
        # Validation phase
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits, _ = model(X_batch) 
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        val_loss = total_loss / len(train_loader)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # Calculate probabilities for validation set for MAP
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            val_logits, _ = model(X_val_tensor)
            val_probs = F.softmax(val_logits, dim=1)
            
            val_probs_np = val_probs.detach().cpu().numpy()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss: {val_loss:.4f}, F1-Macro: {f1_macro:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        early_stopping(f1_macro, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

        if f1_macro > best_f1:
            best_f1 = f1_macro
            # Calculate and store best performance metrics
            best_metrics = {
                'accuracy': accuracy_score(all_labels, all_preds),
                'f1_macro': f1_macro,
                'hamming_loss': hamming_loss(all_labels, all_preds),
                'map': average_precision_score(label_binarize(y_val, classes=range(num_classes)), 
                                                 val_probs_np, 
                                                 average='macro'),
                'report': classification_report(all_labels, all_preds, zero_division=0, output_dict=True),
                # Calculate accuracy for each class
                'class_accuracies': {c: accuracy_score(np.array(all_labels)[np.array(all_labels) == c], np.array(all_preds)[np.array(all_labels) == c]) for c in range(num_classes)}
            }
            
    if best_metrics is None:
        # If training stopped early but no improvement, return current metrics
        best_metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_macro': f1_macro,
            'hamming_loss': hamming_loss(all_labels, all_preds),
            'map': average_precision_score(label_binarize(y_val, classes=range(num_classes)), 
                                             val_probs_np, 
                                             average='macro'),
            'report': classification_report(all_labels, all_preds, zero_division=0, output_dict=True),
            'class_accuracies': {c: accuracy_score(np.array(all_labels)[np.array(all_labels) == c], np.array(all_preds)[np.array(all_labels) == c]) for c in range(num_classes)}
        }

    return best_metrics

# --- Main Execution Flow ---

if __name__ == '__main__':

    # Check if feature files exist
    if not os.path.exists(ESM_CACHE_PATH):
        print(f"ESM feature file '{ESM_CACHE_PATH}' not found")
        exit(1)
        
    if not os.path.exists(PROTT5_CACHE_PATH):
        print(f"ProtT5 feature file '{PROTT5_CACHE_PATH}' not found")

    # 1. Extract or load ESM-2 features
    esm_features = get_enhanced_esm_features(sequences, cache_path=ESM_CACHE_PATH)
    
    # 2. Extract or load ProtT5 features 
    prot_t5_features = np.empty((esm_features.shape[0], 0))

    try:
        # Load ProtT5 directly from cache
        print(f"Attempting to load ProtT5 features from cache: {PROTT5_CACHE_PATH}")
        with open(PROTT5_CACHE_PATH, 'rb') as f:
            prot_t5_features = pickle.load(f)

    except Exception as e:
        print(f"ProtT5 feature loading failed: {e}. Will use only ESM features.")
    
    # 3. Feature concatenation
    if prot_t5_features.shape[1] > 0 and esm_features.shape[0] == prot_t5_features.shape[0]:
        features = np.concatenate([esm_features, prot_t5_features], axis=1)
        print(f"Feature concatenation successful")
    
    num_classes = len(np.unique(labels))
    input_dim = features.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- K-fold Cross Validation Setup ---
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_results = []
    class_accuracies_list = {c: [] for c in range(num_classes)}

    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")
        
        X_train_original, X_val = features[train_idx], features[val_idx]
        y_train_original, y_val = labels[train_idx], labels[val_idx]
        
        # Feature standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_original) 
        X_val_scaled = scaler.transform(X_val)
        
        model = MambaClassifier(input_dim, num_classes).to(device)
        model = pretrain_ssl(model, X_train_scaled, input_dim, device, num_epochs=50) 
        
        # 5. Apply MSMOTE oversampling to training set
        X_train_resampled, y_train_resampled = apply_smote_oversampling(X_train_scaled, y_train_original) 
        
        print(f"Training set size - Original: {len(y_train_original)}, After oversampling: {len(y_train_resampled)}")
        
        # 6. Supervised learning training and evaluation
        metrics = train_and_evaluate(model, X_train_resampled, y_train_resampled, X_val_scaled, y_val, y_train_original, num_classes, device, num_epochs=100, patience=10) 
        cv_results.append(metrics)
        
        for c, acc in metrics['class_accuracies'].items():
            class_accuracies_list[c].append(acc)

    # 7. Statistics and result display
    def calculate_statistics(results, class_acc_list):
        """Calculate mean, standard deviation and 95% confidence interval for cross-validation results"""

        metrics_df = pd.DataFrame([{k: v for k, v in res.items() if k not in ['report', 'class_accuracies']} for res in results])
        overall_stats = {}
        
        for col in metrics_df.columns:
            mean = metrics_df[col].mean()
            std = metrics_df[col].std()
            n = len(metrics_df)
            
            try:
                # 95% CI (using t-distribution)
                ci = t.interval(0.95, n-1, loc=mean, scale=std/np.sqrt(n))
                ci = (max(ci[0], 0), min(ci[1], 1))
            except:
                ci = (mean - 1.96*std/np.sqrt(n), mean + 1.96*std/np.sqrt(n))
            
            overall_stats[col] = {'mean': mean, 'std': std, 'ci': ci}
        
        class_stats = {}
        for class_idx, acc_list in class_acc_list.items():
            if acc_list:
                acc_array = np.array(acc_list)
                mean = acc_array.mean()
                std = acc_array.std()
                n = len(acc_array)
                
                try:
                    ci = t.interval(0.95, n-1, loc=mean, scale=std/np.sqrt(n))
                    ci = (max(ci[0], 0), min(ci[1], 1))
                except:
                    ci = (mean - 1.96*std/np.sqrt(n), mean + 1.96*std/np.sqrt(n))
                
                class_stats[class_idx] = {'mean': mean, 'std': std, 'ci': ci}
        
        return overall_stats, class_stats

    overall_stats, class_stats = calculate_statistics(cv_results, class_accuracies_list)

    print("\n=== Final Evaluation Metrics (Cross-validation Mean ± Std [95% CI]) ===")
    print("Metric          | Mean    | Std      | 95% CI")
    print("---------------------------------------------------------")
    for metric, values in overall_stats.items():
        metric_name = metric.replace('_', ' ').title()
        if metric_name == 'Map': metric_name = 'MAP'
        print(f"{metric_name:<13} | {values['mean']:.4f}  | {values['std']:.4f}  | "
              f"[{values['ci'][0]:.4f}, {values['ci'][1]:.4f}]")

    print("\n=== Class Accuracy Statistics ===")
    print("Class ID        | Mean Acc | Std      | 95% CI")
    print("---------------------------------------------------------")
    for class_idx, values in class_stats.items():
        print(f"Class {class_idx:<6} | {values['mean']:.4f}  | {values['std']:.4f}  | "

              f"[{values['ci'][0]:.4f}, {values['ci'][1]:.4f}]")
