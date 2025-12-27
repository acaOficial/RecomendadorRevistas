import pandas as pd
import glob, json, os, re, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from data_manager import REVISTAS_ASIGNADAS, BASE_PATH, PROCESSED_DATA_PATH

def clean_text(text):
    if not text: return ""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # Limpieza: minúsculas y quitar caracteres no alfabéticos
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    tokens = nltk.word_tokenize(text)
    # Lematización y filtrado de palabras cortas/vacías
    cleaned = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(cleaned)

def run_preprocessing():
    all_articles = []
    
    for revista in REVISTAS_ASIGNADAS:
        path_pattern = os.path.join(BASE_PATH, revista, "*.json")
        files = glob.glob(path_pattern)
        print(f"-> Procesando: {revista} ({len(files)} archivos)")
        
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Manejamos si el archivo es un único objeto o una lista
                articles_list = data if isinstance(data, list) else [data]
                
                for art in articles_list:
                    # Extraer campos según tu JSON
                    titulo = art.get('title', '')
                    abstract = art.get('abstract', '')
                    # Unir lista de keywords en un solo string
                    kw_list = art.get('keywords', [])
                    keywords = " ".join(kw_list) if isinstance(kw_list, list) else str(kw_list)
                    
                    # El recomendador aprende de la unión de los tres campos
                    texto_completo = f"{titulo} {abstract} {keywords}"
                    
                    all_articles.append({
                        'journal': revista,
                        'cleaned_text': clean_text(texto_completo)
                    })

    df = pd.DataFrame(all_articles)
    output_file = os.path.join(PROCESSED_DATA_PATH, "dataset_final.csv")
    df.to_csv(output_file, index=False)
    print(f"\n¡Éxito! Dataset creado con {len(df)} artículos en: {output_file}")

if __name__ == "__main__":
    run_preprocessing()