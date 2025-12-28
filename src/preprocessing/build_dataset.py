from src.data.data_loader import load_journal_folder
import pandas as pd
import os

def build_dataset():
    data = []

    data += load_journal_folder(
        "Dataset/raw/applied_ergonomics",
        "Applied Ergonomics"
    )
    data += load_journal_folder(
        "Dataset/raw/neural_networks",
        "Neural Networks"
    )
    data += load_journal_folder(
        "Dataset/raw/expert_systems",
        "Expert Systems"
    )
    data += load_journal_folder(
        "Dataset/raw/robotics_autonomous_systems",
        "Robotics and Autonomous Systems"
    )

    df = pd.DataFrame(data)

    output_dir = "Dataset/processed"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "dataset.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Dataset guardado correctamente en: {output_path}")
    print(f"Número total de artículos: {len(df)}")
    print(df["journal"].value_counts())

if __name__ == "__main__":
    build_dataset()
