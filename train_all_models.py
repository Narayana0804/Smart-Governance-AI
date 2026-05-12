"""
Smart Governance AI — Full Training Pipeline
=============================================
Runs all training steps in sequence:
  1. Preprocess complaints data
  2. Preprocess fake news data
  3. Generate embeddings (sentence-transformers)
  4. Train BERTopic (topic categorization)
  5. Train Urgency classifier (XLM-RoBERTa fine-tuning)
  6. Build FAISS indices (complaints + news)

Usage:
    python train_all_models.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from configs.settings import (
    RAW_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, VECTOR_DB_DIR, MODELS_DIR,
    EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION,
    URGENCY_MODEL_DIR, TOPIC_MODEL_PATH,
    FAISS_COMPLAINT_INDEX, FAISS_NEWS_INDEX,
    TRAINING_CONFIG
)
from src.preprocessing import TextPreprocessor


# ══════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════

# Paths to raw data
COMPLAINTS_CSV = RAW_DIR / "archive (1)" / "rows.csv"
FAKE_NEWS_CSV = RAW_DIR / "archive (2)" / "Fake.csv"
TRUE_NEWS_CSV = RAW_DIR / "archive (2)" / "True.csv"

# Sampling limits (to keep training fast on CPU)
MAX_COMPLAINTS = 30000      # Sample from 383K rows with narratives
MAX_NEWS_ARTICLES = 20000   # Sample from ~44K articles (10K fake + 10K true)

# Output paths
COMPLAINTS_PROCESSED = PROCESSED_DIR / "complaints_processed.csv"
NEWS_PROCESSED = PROCESSED_DIR / "news_processed.csv"
COMPLAINT_EMBEDDINGS = EMBEDDINGS_DIR / "complaint_embeddings.npy"
NEWS_EMBEDDINGS = EMBEDDINGS_DIR / "news_embeddings.npy"


def banner(msg: str):
    """Print a visible step banner."""
    logger.info("=" * 70)
    logger.info(f"  {msg}")
    logger.info("=" * 70)


# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Preprocess Complaints Data
# ══════════════════════════════════════════════════════════════════════════

def step1_preprocess_complaints() -> pd.DataFrame:
    banner("STEP 1: Preprocessing Complaints Data")

    if COMPLAINTS_PROCESSED.exists():
        logger.info(f"Found existing processed file: {COMPLAINTS_PROCESSED}")
        df = pd.read_csv(COMPLAINTS_PROCESSED)
        logger.info(f"Loaded {len(df)} preprocessed complaints")
        return df

    logger.info(f"Loading raw complaints from: {COMPLAINTS_CSV}")
    df = pd.read_csv(
        COMPLAINTS_CSV,
        usecols=["Consumer complaint narrative", "Product"],
        low_memory=False
    )
    logger.info(f"Total rows: {len(df)}")

    # Drop rows without narrative text
    df = df.dropna(subset=["Consumer complaint narrative"])
    logger.info(f"Rows with narrative: {len(df)}")

    # Sample to keep training manageable
    if len(df) > MAX_COMPLAINTS:
        df = df.sample(n=MAX_COMPLAINTS, random_state=42)
        logger.info(f"Sampled to {MAX_COMPLAINTS} rows")

    # Rename columns
    df = df.rename(columns={
        "Consumer complaint narrative": "text",
        "Product": "category"
    })

    # Preprocess
    preprocessor = TextPreprocessor(min_length=20, max_length=512)
    df["clean_text"] = df["text"].astype(str).apply(preprocessor.clean_text)

    # Drop short/empty
    df = df[df["clean_text"].str.len() >= 20].copy()
    df = df.drop_duplicates(subset=["clean_text"])

    # Detect language
    logger.info("Detecting languages (this may take a few minutes)...")
    df["language"] = df["clean_text"].apply(preprocessor.detect_language)
    lang_dist = df["language"].value_counts().to_dict()
    logger.info(f"Language distribution: {lang_dist}")

    # Generate synthetic urgency labels
    df = preprocessor.generate_synthetic_urgency_labels(df)

    df = df.reset_index(drop=True)

    # Save
    df.to_csv(COMPLAINTS_PROCESSED, index=False)
    logger.info(f"✅ Saved {len(df)} processed complaints to {COMPLAINTS_PROCESSED}")

    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Preprocess Fake News Data
# ══════════════════════════════════════════════════════════════════════════

def step2_preprocess_news() -> pd.DataFrame:
    banner("STEP 2: Preprocessing Fake News Data")

    if NEWS_PROCESSED.exists():
        logger.info(f"Found existing processed file: {NEWS_PROCESSED}")
        df = pd.read_csv(NEWS_PROCESSED)
        logger.info(f"Loaded {len(df)} preprocessed news articles")
        return df

    # Load both CSVs
    logger.info(f"Loading fake news from: {FAKE_NEWS_CSV}")
    fake_df = pd.read_csv(FAKE_NEWS_CSV)
    fake_df["label"] = "FAKE"
    logger.info(f"Fake articles: {len(fake_df)}")

    logger.info(f"Loading true news from: {TRUE_NEWS_CSV}")
    true_df = pd.read_csv(TRUE_NEWS_CSV)
    true_df["label"] = "REAL"
    logger.info(f"True articles: {len(true_df)}")

    # Combine
    df = pd.concat([fake_df, true_df], ignore_index=True)
    logger.info(f"Combined: {len(df)} articles")

    # Sample
    half = MAX_NEWS_ARTICLES // 2
    if len(fake_df) > half:
        fake_sample = fake_df.sample(n=half, random_state=42)
    else:
        fake_sample = fake_df
    if len(true_df) > half:
        true_sample = true_df.sample(n=half, random_state=42)
    else:
        true_sample = true_df
    df = pd.concat([fake_sample, true_sample], ignore_index=True)
    logger.info(f"Sampled to {len(df)} articles")

    # Combine title + text for richer embeddings
    df["full_text"] = df["title"].fillna("") + ". " + df["text"].fillna("")

    # Preprocess
    preprocessor = TextPreprocessor(min_length=30, max_length=1024)
    df["clean_text"] = df["full_text"].astype(str).apply(preprocessor.clean_text)

    # Truncate to 512 chars for embedding
    df["clean_text"] = df["clean_text"].str[:512]

    df = df[df["clean_text"].str.len() >= 30].copy()
    df = df.drop_duplicates(subset=["clean_text"])
    df = df.reset_index(drop=True)

    # Save
    df.to_csv(NEWS_PROCESSED, index=False)
    logger.info(f"✅ Saved {len(df)} processed news articles to {NEWS_PROCESSED}")

    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 3: Generate Embeddings
# ══════════════════════════════════════════════════════════════════════════

def step3_generate_embeddings(complaints_df: pd.DataFrame, news_df: pd.DataFrame):
    banner("STEP 3: Generating Embeddings (Sentence-Transformers)")

    from src.embeddings import EmbeddingGenerator

    emb_gen = EmbeddingGenerator(model_name=EMBEDDING_MODEL_NAME)

    # Complaint embeddings
    if COMPLAINT_EMBEDDINGS.exists():
        logger.info(f"Loading existing complaint embeddings from {COMPLAINT_EMBEDDINGS}")
        complaint_embs = np.load(str(COMPLAINT_EMBEDDINGS))
        logger.info(f"Loaded complaint embeddings: {complaint_embs.shape}")
    else:
        logger.info(f"Generating embeddings for {len(complaints_df)} complaints...")
        complaint_embs = emb_gen.generate_embeddings(
            complaints_df["clean_text"].tolist(),
            batch_size=64,
            show_progress=True
        )
        np.save(str(COMPLAINT_EMBEDDINGS), complaint_embs)
        logger.info(f"✅ Complaint embeddings saved: {complaint_embs.shape}")

    # News embeddings
    if NEWS_EMBEDDINGS.exists():
        logger.info(f"Loading existing news embeddings from {NEWS_EMBEDDINGS}")
        news_embs = np.load(str(NEWS_EMBEDDINGS))
        logger.info(f"Loaded news embeddings: {news_embs.shape}")
    else:
        logger.info(f"Generating embeddings for {len(news_df)} news articles...")
        news_embs = emb_gen.generate_embeddings(
            news_df["clean_text"].tolist(),
            batch_size=64,
            show_progress=True
        )
        np.save(str(NEWS_EMBEDDINGS), news_embs)
        logger.info(f"✅ News embeddings saved: {news_embs.shape}")

    return complaint_embs, news_embs


# ══════════════════════════════════════════════════════════════════════════
# STEP 4: Train BERTopic (Topic Categorization)
# ══════════════════════════════════════════════════════════════════════════

def step4_train_topic_model(complaints_df: pd.DataFrame, complaint_embs: np.ndarray):
    banner("STEP 4: Training BERTopic (Topic Categorization)")

    topic_dir = str(TOPIC_MODEL_PATH).replace('.pkl', '')
    if Path(topic_dir).exists():
        logger.info(f"Topic model already exists at {topic_dir}, skipping.")
        return

    from src.topic_modeling import TopicModeler

    topic_modeler = TopicModeler(
        min_topic_size=15,
        umap_n_neighbors=15,
        umap_n_components=5,
    )

    texts = complaints_df["clean_text"].tolist()
    topics, probs = topic_modeler.fit_transform(texts, embeddings=complaint_embs)

    # Save model
    topic_modeler.save_model()
    logger.info("✅ BERTopic model trained and saved!")

    # Log discovered topics
    topic_info = topic_modeler.get_topic_info()
    logger.info(f"\nDiscovered Topics:\n{topic_info[['Topic', 'Count', 'Name']].head(20).to_string()}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 5: Train Urgency Classifier (XLM-RoBERTa)
# ══════════════════════════════════════════════════════════════════════════

def step5_train_urgency_model(complaints_df: pd.DataFrame):
    banner("STEP 5: Training Urgency Classifier (XLM-RoBERTa)")

    if (URGENCY_MODEL_DIR / "config.json").exists():
        logger.info(f"Urgency model already exists at {URGENCY_MODEL_DIR}, skipping.")
        return

    from src.urgency_model import UrgencyClassifier

    classifier = UrgencyClassifier()

    # Use a subset for training (XLM-RoBERTa fine-tuning is expensive on CPU)
    train_size = min(500, len(complaints_df))
    train_df = complaints_df.sample(n=train_size, random_state=42).copy()

    # Ensure balanced-ish classes
    urgency_dist = train_df["urgency"].value_counts()
    logger.info(f"Training urgency distribution:\n{urgency_dist}")

    results = classifier.train(
        train_df=train_df,
        text_col="clean_text",
        label_col="urgency",
        output_dir=URGENCY_MODEL_DIR
    )

    logger.info(f"✅ Urgency model trained! Results: {results}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 6: Build FAISS Indices
# ══════════════════════════════════════════════════════════════════════════

def step6_build_faiss_indices(
    complaints_df: pd.DataFrame, news_df: pd.DataFrame,
    complaint_embs: np.ndarray, news_embs: np.ndarray
):
    banner("STEP 6: Building FAISS Indices")

    from src.vector_db import FAISSVectorDB

    # Complaint index
    if not FAISS_COMPLAINT_INDEX.exists():
        logger.info(f"Building complaint FAISS index ({len(complaint_embs)} vectors)...")
        complaint_db = FAISSVectorDB(dimension=complaint_embs.shape[1])
        complaint_db.build_index(
            embeddings=complaint_embs,
            texts=complaints_df["clean_text"].tolist(),
            use_ivf=(len(complaint_embs) > 5000)
        )
        complaint_db.save_index(FAISS_COMPLAINT_INDEX)
        logger.info(f"✅ Complaint FAISS index saved ({complaint_db.index.ntotal} vectors)")
    else:
        logger.info(f"Complaint FAISS index already exists, skipping.")

    # News index
    if not FAISS_NEWS_INDEX.exists():
        logger.info(f"Building news FAISS index ({len(news_embs)} vectors)...")
        news_db = FAISSVectorDB(dimension=news_embs.shape[1])
        news_db.build_index(
            embeddings=news_embs,
            texts=news_df["clean_text"].tolist(),
            use_ivf=(len(news_embs) > 5000)
        )
        news_db.save_index(FAISS_NEWS_INDEX)
        logger.info(f"✅ News FAISS index saved ({news_db.index.ntotal} vectors)")
    else:
        logger.info(f"News FAISS index already exists, skipping.")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    start = time.time()
    banner("🚀 Smart Governance AI — Full Training Pipeline")

    # Verify data files exist
    missing = []
    if not COMPLAINTS_CSV.exists():
        missing.append(str(COMPLAINTS_CSV))
    if not FAKE_NEWS_CSV.exists():
        missing.append(str(FAKE_NEWS_CSV))
    if not TRUE_NEWS_CSV.exists():
        missing.append(str(TRUE_NEWS_CSV))

    if missing:
        logger.error(f"Missing data files: {missing}")
        logger.error("Please place the CSV files in data/raw/ as instructed.")
        sys.exit(1)

    # Step 1: Preprocess complaints
    complaints_df = step1_preprocess_complaints()

    # Step 2: Preprocess news
    news_df = step2_preprocess_news()

    # Step 3: Generate embeddings (requires sentence-transformers + torch)
    try:
        complaint_embs, news_embs = step3_generate_embeddings(complaints_df, news_df)
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        logger.error("Make sure torch/sentence-transformers are working (check K7 antivirus)")
        logger.info("Skipping embedding-dependent steps (BERTopic, FAISS).")
        logger.info("The API will still work with rule-based fallbacks.")
        elapsed = time.time() - start
        logger.info(f"\n⏱️  Pipeline finished in {elapsed/60:.1f} minutes (partial)")
        return

    # Step 4: Train BERTopic
    try:
        step4_train_topic_model(complaints_df, complaint_embs)
    except Exception as e:
        logger.error(f"❌ BERTopic training failed: {e}")
        logger.info("Topic prediction will use rule-based fallback.")

    # Step 5: Train urgency classifier
    try:
        step5_train_urgency_model(complaints_df)
    except Exception as e:
        logger.error(f"❌ Urgency model training failed: {e}")
        logger.info("Urgency prediction will use rule-based fallback.")

    # Step 6: Build FAISS indices
    try:
        step6_build_faiss_indices(complaints_df, news_df, complaint_embs, news_embs)
    except Exception as e:
        logger.error(f"❌ FAISS index building failed: {e}")
        logger.info("Similarity search will be unavailable.")

    elapsed = time.time() - start
    banner(f"✅ PIPELINE COMPLETE — {elapsed/60:.1f} minutes")
    logger.info("\nNext steps:")
    logger.info("  1. Restart the API:    python -m api.main")
    logger.info("  2. Restart dashboard:  streamlit run dashboard/app.py")
    logger.info("  3. All models should now load automatically!")


if __name__ == "__main__":
    main()
