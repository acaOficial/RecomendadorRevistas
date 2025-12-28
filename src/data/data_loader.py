import json
import os

def load_journal_folder(folder_path, journal_name):
    articles = []
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                for paper in data:
                    articles.append({
                        "title": paper.get("title", ""),
                        "abstract": paper.get("abstract", ""),
                        "keywords": " ".join(paper.get("keywords", [])),
                        "journal": journal_name
                    })
    return articles
