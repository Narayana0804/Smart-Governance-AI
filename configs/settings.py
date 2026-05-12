"""
Configuration settings for Smart Governance AI Platform.
Central config for all paths, model names, and hyperparameters.
"""

import os
from pathlib import Path

# ── Base Paths ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
MODELS_DIR = BASE_DIR / "models"

# ── Model Configuration ────────────────────────────────────────────────
# Embedding model (384-dim, fast, multilingual-capable)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Urgency classifier (XLM-RoBERTa for multilingual support)
URGENCY_MODEL_NAME = "xlm-roberta-base"
URGENCY_LABELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
URGENCY_NUM_LABELS = len(URGENCY_LABELS)
URGENCY_MODEL_DIR = MODELS_DIR / "urgency_model"

# Topic model
TOPIC_MODEL_PATH = MODELS_DIR / "topic_model.pkl"

# FAISS indices
FAISS_COMPLAINT_INDEX = VECTOR_DB_DIR / "complaint_index.faiss"
FAISS_NEWS_INDEX = VECTOR_DB_DIR / "news_index.faiss"

# ── Training Configuration ─────────────────────────────────────────────
TRAINING_CONFIG = {
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 2,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "max_seq_length": 256,
    "eval_steps": 100,
    "save_steps": 500,
}

# ── Preprocessing Configuration ────────────────────────────────────────
SUPPORTED_LANGUAGES = ["en", "hi", "te", "ta"]
MIN_TEXT_LENGTH = 10
MAX_TEXT_LENGTH = 512

# ── FAISS Configuration ────────────────────────────────────────────────
FAISS_TOP_K = 10
SIMILARITY_THRESHOLD = 0.75

# ── API Configuration ──────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_TITLE = "Smart Governance AI API"
API_VERSION = "1.0.0"

# ── Dashboard Configuration ────────────────────────────────────────────
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", 8501))

# ── Ensure directories exist ───────────────────────────────────────────
for d in [RAW_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, VECTOR_DB_DIR, MODELS_DIR, URGENCY_MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)
