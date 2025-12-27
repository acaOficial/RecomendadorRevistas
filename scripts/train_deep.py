import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report
from data_manager import PROCESSED_DATA_PATH, MODELS_PATH
import os
import numpy as np

# --- CONFIGURACIÓN DE ÉLITE (GOD MODE) ---
MAX_WORDS = 15000   
MAX_LEN = 300       # Captura la semántica completa del abstract
BATCH_SIZE = 32     # Batch pequeño para mayor estabilidad en clases minoritarias
DEVICE = torch.device('cpu') 
EMBEDDING_DIM = 256 
EPOCHS = 50
PATIENCE = 10

# CAPA DE ATENCIÓN: Permite al modelo "enfocarse" en términos técnicos clave
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch, features, seq_len]
        weights = torch.tanh(self.attn(x.permute(0, 2, 1))) # [batch, seq_len, 1]
        weights = F.softmax(weights, dim=1)
        return torch.sum(x * weights.permute(0, 2, 1), dim=2)

class TextDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = str(self.texts[idx]).split()
        seq = [self.word_to_idx.get(w, 1) for w in tokens[:MAX_LEN]] 
        if len(seq) < MAX_LEN:
            seq += [0] * (MAX_LEN - len(seq))
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

class GodJournalCNN(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super(GodJournalCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0)
        self.spatial_dropout = nn.Dropout2d(0.3)
        
        # Convoluciones multiescala para captar desde términos simples hasta frases complejas
        self.convs = nn.ModuleList([
            nn.Conv1d(EMBEDDING_DIM, 128, kernel_size=k, padding=k//2) for k in [3, 5, 7]
        ])
        
        self.attention = Attention(128 * 3)
        self.pool_max = nn.AdaptiveMaxPool1d(1)
        
        # Clasificador con activación SiLU (Sigmoid Linear Unit)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 2, 512), 
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.spatial_dropout(x.unsqueeze(3)).squeeze(3)
        
        # Extraer características en paralelo
        conv_results = [F.leaky_relu(conv(x)) for conv in self.convs]
        combined = torch.cat(conv_results, dim=1) 
        
        # Dual Path: Atención (contexto) + MaxPool (palabras clave)
        attn_out = self.attention(combined)
        pool_out = self.pool_max(combined).squeeze(2)
        
        final_features = torch.cat([attn_out, pool_out], dim=1)
        return self.classifier(final_features)

def train_deep():
    print(f"\n{'='*60}\n EJECUTANDO 'GOD MODE' - CNN CON ATENCIÓN\n{'='*60}")
    
    df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "dataset_final.csv")).dropna()
    le = LabelEncoder()
    y = le.fit_transform(df['journal'])
    
    # Vocabulario extendido para no perder términos técnicos raros
    all_words = " ".join(df['cleaned_text']).split()
    vocab = {word: i+2 for i, word in enumerate(pd.Series(all_words).value_counts().index[:MAX_WORDS-2])}
    vocab["<PAD>"], vocab["<UNK>"] = 0, 1

    # División balanceada
    X_temp, X_test, y_temp, y_test = train_test_split(df['cleaned_text'].values, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, stratify=y_temp, random_state=42)

    train_loader = DataLoader(TextDataset(X_train, y_train, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(X_val, y_val, vocab), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TextDataset(X_test, y_test, vocab), batch_size=BATCH_SIZE)

    # Pesos y Modelo
    class_weights = torch.FloatTensor(compute_class_weight('balanced', classes=np.unique(y), y=y))
    model = GodJournalCNN(len(vocab), len(le.classes_))
    
    # Optimizador AdamW y Scheduler Cíclico
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # CrossEntropy con Label Smoothing para penalizar el exceso de confianza
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    best_f1 = 0
    patience_cnt = 0

    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        for texts, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(texts), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        
        model.eval()
        v_preds, v_labels = [], []
        with torch.no_grad():
            for texts, labels in val_loader:
                v_preds.extend(model(texts).argmax(1).numpy())
                v_labels.extend(labels.numpy())
        
        v_f1 = f1_score(v_labels, v_preds, average='weighted')
        v_acc = (np.array(v_preds) == np.array(v_labels)).mean()
        scheduler.step(v_f1)

        print(f"Epo {epoch+1:2d} | Val Acc: {100*v_acc:.1f}% | Val F1: {v_f1:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save({'model_state': model.state_dict(), 'vocab': vocab, 'classes': le.classes_}, 
                       os.path.join(MODELS_PATH, "deep_model_cnn.pth"))
            patience_cnt = 0
            print("  [✓] Nuevo récord guardado.")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print("\n[!] Early stopping activado.")
                break

    # --- EVALUACIÓN FINAL DEFINITIVA ---
    print(f"\n{'='*60}\n RESULTADOS FINALES EN TEST SET (MODELO DIOS)\n{'='*60}")
    ckpt = torch.load(os.path.join(MODELS_PATH, "deep_model_cnn.pth"))
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    
    final_preds = []
    with torch.no_grad():
        for texts, _ in test_loader:
            final_preds.extend(model(texts).argmax(1).numpy())

    print(classification_report(y_test, final_preds, target_names=le.classes_, digits=4))

if __name__ == "__main__":
    train_deep()