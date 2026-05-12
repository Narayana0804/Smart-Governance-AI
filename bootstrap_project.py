
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from configs.settings import (
    RAW_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, VECTOR_DB_DIR, 
    MODELS_DIR, URGENCY_MODEL_DIR, TOPIC_MODEL_PATH,
    FAISS_COMPLAINT_INDEX, FAISS_NEWS_INDEX
)
from src.preprocessing import TextPreprocessor
from src.embeddings import EmbeddingGenerator
from src.vector_db import FAISSVectorDB

def bootstrap():
    logger.info("🚀 Starting Smart Governance AI Bootstrap...")
    
    # 1. Check for data
    complaints_raw = RAW_DIR / "complaints.csv"
    news_raw = RAW_DIR / "fake_news.csv"
    
    if not complaints_raw.exists():
        logger.error(f"❌ Complaints data missing at {complaints_raw}")
        logger.info("Please download a dataset (e.g., NYC 311) and save it as complaints.csv in data/raw/")
        return

    if not news_raw.exists():
        logger.warning(f"⚠️ News data missing at {news_raw}. Will skip news indexing.")

    preprocessor = TextPreprocessor()
    embedder = EmbeddingGenerator()
    
    # 2. Process Complaints
    logger.info("--- Processing Complaints ---")
    df_complaints = preprocessor.process_complaints(str(complaints_raw))
    # Take a sample for bootstrap to keep it fast
    if len(df_complaints) > 5000:
        df_complaints = df_complaints.sample(5000, random_state=42)
    
    # Generate synthetic urgency labels for bootstrap
    df_complaints = preprocessor.generate_synthetic_urgency_labels(df_complaints)
    
    complaints_processed_path = PROCESSED_DIR / "complaints_processed.csv"
    df_complaints.to_csv(complaints_processed_path, index=False)
    
    # 3. Generate Embeddings for Complaints
    logger.info("Generating embeddings for complaints...")
    complaint_embs = embedder.generate_and_save(df_complaints['clean_text'], "complaint_embeddings.npy")
    
    # 4. Build FAISS Index for Complaints
    logger.info("Building FAISS index for complaints...")
    complaint_db = FAISSVectorDB(dimension=embedder.dimension)
    complaint_db.add_documents(df_complaints['clean_text'].tolist(), complaint_embs)
    complaint_db.save_index(FAISS_COMPLAINT_INDEX)
    
    # 5. Process News (if exists)
    if news_raw.exists():
        logger.info("--- Processing News ---")
        df_news = preprocessor.process_news(str(news_raw))
        if len(df_news) > 2000:
            df_news = df_news.sample(2000, random_state=42)
            
        news_processed_path = PROCESSED_DIR / "news_processed.csv"
        df_news.to_csv(news_processed_path, index=False)
        
        logger.info("Generating embeddings for news...")
        news_embs = embedder.generate_and_save(df_news['clean_text'], "news_embeddings.npy")
        
        logger.info("Building FAISS index for news...")
        news_db = FAISSVectorDB(dimension=embedder.dimension)
        news_db.add_documents(df_news['clean_text'].tolist(), news_embs)
        news_db.save_index(FAISS_NEWS_INDEX)

    # 6. Initialize Models (Cold Start)
    logger.info("--- Initializing Models (Cold Start) ---")
    
    # Urgency Model - Save base model as "trained" to allow API to start
    try:
        from src.urgency_model import UrgencyClassifier
        classifier = UrgencyClassifier()
        classifier._load_base_model()
        classifier.save_model(URGENCY_MODEL_DIR)
        logger.info("✅ Urgency model initialized with base weights.")
    except Exception as e:
        logger.error(f"Failed to initialize urgency model: {e}")

    # Topic Model - Train a small BERTopic model
    try:
        from src.topic_modeling import TopicModeler
        topic_modeler = TopicModeler(min_topic_size=10)
        topic_modeler.fit_transform(df_complaints['clean_text'].tolist(), embeddings=complaint_embs)
        topic_modeler.save_model(TOPIC_MODEL_PATH)
        logger.info("✅ Topic model trained and saved.")
    except Exception as e:
        logger.error(f"Failed to train topic model: {e}")

    logger.info("=" * 50)
    logger.info("🎉 Bootstrap Complete! Your system is now ready.")
    logger.info("Restart your API and Dashboard to see the green checks.")
    logger.info("=" * 50)

if __name__ == "__main__":
    bootstrap()
