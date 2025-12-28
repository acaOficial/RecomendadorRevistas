import re
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import Dataset, DataLoader
from collections import Counter


# ==================================================
# Configuración del modelo
# ==================================================

MAX_LEN = 120
BATCH_SIZE = 64
EPOCHS = 10
EMBED_DIM = 128
HIDDEN_DIM = 128
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================================================
# Utilidades de texto
# ==================================================

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def build_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab


def encode(text, vocab):
    tokens = tokenize(text)
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens]
    ids = ids[:MAX_LEN]
    return ids + [0] * (MAX_LEN - len(ids))


# ==================================================
# Dataset PyTorch
# ==================================================

class TextDataset(Dataset):
    def __init__(self, df, vocab):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return (
            torch.tensor(encode(self.texts[idx], self.vocab), dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


# ==================================================
# Modelo LSTM bidireccional
# ==================================================

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(
            EMBED_DIM,
            HIDDEN_DIM,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(HIDDEN_DIM * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        _, (h, _) = self.lstm(x)
        h_final = torch.cat((h[-2], h[-1]), dim=1)
        h_final = self.dropout(h_final)
        return self.fc(h_final)


# ==================================================
# Ejecución principal del modelo LSTM
# ==================================================

def run_lstm_model():

    print(f"[INFO] Device seleccionado: {DEVICE}")

    # --------------------------------------------------
    # 1. Cargar datos
    # --------------------------------------------------

    print("\n[STEP 1] Cargando dataset...")
    df = pd.read_csv("Dataset/processed/dataset.csv")
    print(f"[INFO] Artículos totales: {len(df)}")

    df["text"] = (
        df["title"].fillna("") + " " +
        df["abstract"].fillna("") + " " +
        df["keywords"].fillna("")
    )

    labels = sorted(df["journal"].unique())
    label2idx = {l: i for i, l in enumerate(labels)}
    df["label"] = df["journal"].map(label2idx)

    print("[INFO] Clases:")
    for l, i in label2idx.items():
        print(f"  - {l}: {i}")

    # --------------------------------------------------
    # 2. Train / Test split
    # --------------------------------------------------

    print("\n[STEP 2] Dividiendo en train / test...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df["label"],
        random_state=42
    )

    print(f"[INFO] Train: {len(train_df)} | Test: {len(test_df)}")

    # --------------------------------------------------
    # 3. Vocabulario
    # --------------------------------------------------

    print("\n[STEP 3] Construyendo vocabulario...")
    vocab = build_vocab(train_df["text"])
    print(f"[INFO] Tamaño del vocabulario: {len(vocab)}")

    # --------------------------------------------------
    # 4. DataLoaders
    # --------------------------------------------------

    print("\n[STEP 4] Creando DataLoaders...")
    train_ds = TextDataset(train_df, vocab)
    test_ds = TextDataset(test_df, vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    print(f"[INFO] Batches train: {len(train_loader)} | test: {len(test_loader)}")

    # --------------------------------------------------
    # 5. Modelo
    # --------------------------------------------------

    print("\n[STEP 5] Inicializando modelo LSTM...")
    model = LSTMClassifier(len(vocab), len(labels)).to(DEVICE)
    print(model)

    # --------------------------------------------------
    # 6. Entrenamiento
    # --------------------------------------------------

    print("\n[STEP 6] Entrenando modelo...")

    class_counts = train_df["label"].value_counts().sort_index().values
    class_weights = torch.tensor(
        class_counts.sum() / class_counts,
        dtype=torch.float
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        print(f"\n[EPOCH {epoch+1}/{EPOCHS}]")

        for i, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 50 == 0 or i == len(train_loader):
                print(f"  Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        print(f"[INFO] Epoch {epoch+1} mean loss: {total_loss / len(train_loader):.4f}")

    # --------------------------------------------------
    # 7. Evaluación
    # --------------------------------------------------

    print("\n[STEP 7] Evaluando modelo...")
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            preds = model(x).argmax(dim=1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)

    print("\n=== RESULTADOS LSTM ===")
    print(classification_report(y_true, y_pred, target_names=labels))
    print("Macro-F1:", f1_score(y_true, y_pred, average="macro"))


# ==================================================
# Ejecución directa
# ==================================================

if __name__ == "__main__":
    run_lstm_model()
