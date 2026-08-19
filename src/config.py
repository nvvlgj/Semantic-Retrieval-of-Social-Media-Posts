from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data/raw/kansas_city.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data/processed/kansas_city_clean.csv"

TEXT_COLUMN = "content"

# Cleaning
REMOVE_URLS = True
REMOVE_EMPTY_POSTS = True

# Near duplicate detection
ENABLE_NEAR_DUPLICATE_REMOVAL = True
NEAR_DUPLICATE_THRESHOLD = 0.95

# Embedding

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 32

NORMALIZE_EMBEDDINGS = True

EMBEDDING_DIRECTORY = PROJECT_ROOT / "data" / "embeddings"

EMBEDDING_FILE = EMBEDDING_DIRECTORY / "embeddings.npy"

METADATA_FILE = EMBEDDING_DIRECTORY / "metadata.csv"

EMBEDDING_INFO_FILE = EMBEDDING_DIRECTORY / "embedding_info.json"

# Retrieval

TOP_K = 30

RESULT_DIRECTORY = PROJECT_ROOT / "results"

RESULT_FILE = RESULT_DIRECTORY / "top30_results.csv"

RETRIEVAL_INFO_FILE = RESULT_DIRECTORY / "retrieval_info.json"

EVALUATION_FILE = RESULT_DIRECTORY / "retrieval_evaluation.json" 