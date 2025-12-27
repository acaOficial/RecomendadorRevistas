import os

# Revistas asignadas 1, 2, 3 y 6
REVISTAS_ASIGNADAS = [
    "1 Applied Ergonomics",
    "2 Neural Networks",
    "3 Expert Systems with Applications",
    "6 Robotics and Autonomous Systems"
]

BASE_PATH = "Dataset"
PROCESSED_DATA_PATH = "processed_data"
MODELS_PATH = "models"

# Crear carpetas si no existen
for path in [PROCESSED_DATA_PATH, MODELS_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)