import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import print_confusion_matrix, plot_confusion_matrix

def run_classical_models():
    df = pd.read_csv("Dataset/processed/dataset.csv")

    df["text"] = (
        df["title"].fillna("") + " " +
        df["abstract"].fillna("") + " " +
        df["keywords"].fillna("")
    )

    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=3
    )

    X = vectorizer.fit_transform(df["text"])
    y = df["journal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    models = {
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=42),
        "Logistic Regression": LogisticRegression(
            max_iter=300,
            class_weight="balanced",
            n_jobs=-1
        ),
        "Multinomial NB": MultinomialNB()
    }

    results = []

    for name, model in models.items():
        print(f"\nModelo: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(classification_report(y_test, y_pred))
        macro_f1 = f1_score(y_test, y_pred, average="macro")


        print_confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
        plot_confusion_matrix(
            y_test,
            y_pred,
            labels=sorted(y.unique()),
            title=f"Matriz de Confusión - {name}"
        )

        results.append((name, macro_f1))

    print("\nResumen Macro-F1:")
    for name, score in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{name}: {score:.4f}")

if __name__ == "__main__":
    run_classical_models()
