import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from data_manager import PROCESSED_DATA_PATH, MODELS_PATH

def train_classic():
    csv_path = os.path.join(PROCESSED_DATA_PATH, "dataset_final.csv")
    if not os.path.exists(csv_path):
        print("Error: No se encuentra el CSV. Ejecuta primero preprocess.py")
        return

    # Cargar y limpiar nulos
    df = pd.read_csv(csv_path).dropna()
    
    # Analizar distribución de clases
    print("\n=== DISTRIBUCIÓN DE CLASES ===")
    class_dist = df['journal'].value_counts()
    print(class_dist)
    print(f"\nTotal artículos: {len(df)}")
    
    # Calcular class weights para manejar desbalanceo
    classes = df['journal'].unique()
    class_weights = compute_class_weight('balanced', classes=classes, y=df['journal'])
    class_weight_dict = dict(zip(classes, class_weights))
    print(f"\nClass weights calculados para balanceo")
    
    # División estratificada para mantener proporciones de las revistas
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['journal'], 
        test_size=0.20, random_state=42, stratify=df['journal']
    )

    # MEJORA 1: Vectorización TF-IDF optimizada
    # - Incrementar max_features para capturar más vocabulario
    # - Usar trigramas (1,3) para capturar frases comunes
    # - sublinear_tf para normalizar frecuencias extremas
    print("\n=== VECTORIZACIÓN TF-IDF MEJORADA ===")
    vectorizer = TfidfVectorizer(
        max_features=10000,      # Más características
        ngram_range=(1, 3),      # Uni, bi y trigramas
        min_df=2,                # Ignorar términos muy raros
        max_df=0.95,             # Ignorar términos muy comunes
        sublinear_tf=True,       # Escala logarítmica de frecuencias
        use_idf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"Matriz TF-IDF: {X_train_tfidf.shape}")

    # MEJORA 2: Entrenar múltiples modelos con GridSearch
    print("\n=== ENTRENANDO MODELO 1: Linear SVM ===")
    svm_params = {'C': [0.1, 1, 10, 50]}
    svm_grid = GridSearchCV(
        LinearSVC(max_iter=3000, random_state=42, class_weight='balanced'),
        svm_params, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1
    )
    svm_grid.fit(X_train_tfidf, y_train)
    best_svm = svm_grid.best_estimator_
    print(f"Mejor SVM - C: {svm_grid.best_params_['C']}, Score: {svm_grid.best_score_:.4f}")
    
    print("\n=== ENTRENANDO MODELO 2: Logistic Regression ===")
    lr_params = {'C': [0.1, 1, 10, 50], 'solver': ['saga']}
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', n_jobs=-1),
        lr_params, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1
    )
    lr_grid.fit(X_train_tfidf, y_train)
    best_lr = lr_grid.best_estimator_
    print(f"Mejor LR - C: {lr_grid.best_params_['C']}, Score: {lr_grid.best_score_:.4f}")
    
    print("\n=== ENTRENANDO MODELO 3: Random Forest ===")
    rf_params = {'n_estimators': [100, 200], 'max_depth': [30, 50], 'min_samples_split': [2, 5]}
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
        rf_params, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1
    )
    rf_grid.fit(X_train_tfidf, y_train)
    best_rf = rf_grid.best_estimator_
    print(f"Mejor RF - Score: {rf_grid.best_score_:.4f}")
    
    # MEJORA 3: Ensemble con Voting Classifier (hard voting)
    print("\n=== CREANDO ENSEMBLE DE MODELOS ===")
    ensemble = VotingClassifier(
        estimators=[
            ('svm', best_svm),
            ('lr', best_lr),
            ('rf', best_rf)
        ],
        voting='hard',  # Hard voting (mayoría)
        n_jobs=-1
    )
    ensemble.fit(X_train_tfidf, y_train)

    # Evaluación de cada modelo
    print("\n=== COMPARACIÓN DE MODELOS ===")
    models = {
        'Linear SVM': best_svm,
        'Logistic Regression': best_lr,
        'Random Forest': best_rf,
        'Ensemble (Voting)': ensemble
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        preds = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        print(f"\n{name}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        if f1 > best_score:
            best_score = f1
            best_model = model
            best_name = name
    
    # Reporte detallado del mejor modelo
    print(f"\n{'='*60}")
    print(f"MEJOR MODELO: {best_name} (F1-Score: {best_score:.4f})")
    print(f"{'='*60}")
    preds = best_model.predict(X_test_tfidf)
    print("\nReporte de clasificación detallado:")
    print(classification_report(y_test, preds))

    # Guardar el mejor modelo y el ensemble
    joblib.dump(best_model, os.path.join(MODELS_PATH, "classic_model.pkl"))
    joblib.dump(ensemble, os.path.join(MODELS_PATH, "ensemble_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS_PATH, "tfidf_vectorizer.pkl"))
    print(f"\n✓ Mejor modelo guardado: {best_name}")
    print("✓ Ensemble guardado")
    print("✓ Vectorizador guardado")

if __name__ == "__main__":
    train_classic()