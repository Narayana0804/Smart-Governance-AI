"""
FastAPI Backend for Smart Governance AI.
Exposes model inference APIs for complaint analysis and misinformation detection.

Endpoints:
  POST /predict-topic     → Complaint category prediction
  POST /predict-urgency   → Urgency level detection
  POST /similar-news      → Fake news similarity search
  POST /analyze-text      → Full pipeline analysis
  GET  /health            → Health check
  GET  /stats             → System statistics
"""

import sys
import os
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager
from loguru import logger

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import json
import uvicorn

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from configs.settings import (
    API_TITLE, API_VERSION, API_HOST, API_PORT,
    URGENCY_MODEL_DIR, TOPIC_MODEL_PATH,
    FAISS_COMPLAINT_INDEX, FAISS_NEWS_INDEX,
    PROCESSED_DIR, EMBEDDINGS_DIR
)
from src.preprocessing import TextPreprocessor
from src.embeddings import EmbeddingGenerator
from src.vector_db import FAISSVectorDB

# ── Pydantic Models ─────────────────────────────────────────────────────

class TextInput(BaseModel):
    text: str = Field(..., min_length=5, description="Input text for analysis")
    language: Optional[str] = Field(None, description="Language code (auto-detect if not provided)")

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of texts for analysis")

class TopicResponse(BaseModel):
    text: str
    topic_id: int
    topic_label: str
    confidence: float
    top_keywords: List[str]

class UrgencyResponse(BaseModel):
    text: str
    urgency: str
    confidence: float
    probabilities: dict

class SimilarityResponse(BaseModel):
    text: str
    is_suspicious: bool
    confidence: float
    num_similar: int
    similar_articles: list
    reason: str

class FullAnalysisResponse(BaseModel):
    text: str
    clean_text: str
    language: str
    topic: dict
    urgency: dict
    similarity: dict

class HealthResponse(BaseModel):
    status: str
    models_loaded: dict
    version: str

# ── Lifespan ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global embedding_generator, urgency_classifier, topic_modeler, complaint_db, news_db, models_loaded

    logger.info("=" * 60)
    logger.info("Starting Smart Governance AI API...")
    logger.info("=" * 60)

    # Load embedding model
    try:
        embedding_generator = EmbeddingGenerator()
        models_loaded["embeddings"] = True
        logger.info("✅ Embedding model loaded")
    except Exception as e:
        logger.warning(f"⚠️ Embedding model not loaded: {e}")

    # Load urgency model
    try:
        if URGENCY_MODEL_DIR.exists() and (URGENCY_MODEL_DIR / "config.json").exists():
            from src.urgency_model import UrgencyClassifier
            urgency_classifier = UrgencyClassifier()
            urgency_classifier.load_model()
            models_loaded["urgency"] = True
            logger.info("✅ Urgency model loaded")
        else:
            logger.info("ℹ️ Urgency model not found - will use rule-based fallback")
    except Exception as e:
        logger.warning(f"⚠️ Urgency model not loaded: {e}")

    # Load topic model
    try:
        topic_dir = str(TOPIC_MODEL_PATH).replace('.pkl', '')
        if Path(topic_dir).exists():
            from src.topic_modeling import TopicModeler
            topic_modeler = TopicModeler()
            topic_modeler.load_model()
            models_loaded["topic"] = True
            logger.info("✅ Topic model loaded")
        else:
            logger.info("ℹ️ Topic model not found - will use keyword-based fallback")
    except Exception as e:
        logger.warning(f"⚠️ Topic model not loaded: {e}")

    # Load FAISS indices
    try:
        if FAISS_COMPLAINT_INDEX.exists():
            complaint_db = FAISSVectorDB()
            complaint_db.load_index(FAISS_COMPLAINT_INDEX)
            models_loaded["faiss_complaints"] = True
            logger.info("✅ Complaint FAISS index loaded")
    except Exception as e:
        logger.warning(f"⚠️ Complaint FAISS index not loaded: {e}")

    try:
        if FAISS_NEWS_INDEX.exists():
            news_db = FAISSVectorDB()
            news_db.load_index(FAISS_NEWS_INDEX)
            models_loaded["faiss_news"] = True
            logger.info("✅ News FAISS index loaded")
    except Exception as e:
        logger.warning(f"⚠️ News FAISS index not loaded: {e}")

    logger.info(f"Models status: {models_loaded}")
    logger.info("API startup complete!")
    yield  # App runs here
    logger.info("Shutting down Smart Governance AI API...")


# ── Initialize App ──────────────────────────────────────────────────────

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="AI-Powered Smart Governance + Regional Language Misinformation Detection API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ────────────────────────────────────────────────────────

preprocessor = TextPreprocessor()
embedding_generator = None
urgency_classifier = None
topic_modeler = None
complaint_db = None
news_db = None

models_loaded = {
    "embeddings": False,
    "urgency": False,
    "topic": False,
    "faiss_complaints": False,
    "faiss_news": False
}



# ── Fallback Functions ──────────────────────────────────────────────────

def _rule_based_topic(text: str) -> dict:
    """Keyword-based topic detection fallback."""
    text_lower = text.lower()
    topics = {
        "Water Issues": ["water", "leak", "pipe", "supply", "drainage", "sewage", "flood", "पानी", "నీరు"],
        "Road Damage": ["road", "pothole", "crack", "pavement", "highway", "bridge", "सड़क", "రోడ్డు"],
        "Garbage & Sanitation": ["garbage", "waste", "trash", "clean", "sanitation", "dump", "कचरा", "చెత్త"],
        "Electricity": ["power", "electric", "light", "outage", "transformer", "wire", "बिजली", "విద్యుత్"],
        "Public Transport": ["bus", "metro", "train", "transport", "traffic", "signal", "यातायात"],
        "Noise & Pollution": ["noise", "pollution", "smoke", "dust", "air quality", "प्रदूषण"],
        "Housing": ["building", "construction", "housing", "colony", "apartment", "मकान"],
        "Healthcare": ["hospital", "clinic", "doctor", "health", "medicine", "अस्पताल"],
    }

    for topic, keywords in topics.items():
        if any(kw in text_lower for kw in keywords):
            return {"topic_id": list(topics.keys()).index(topic), "topic_label": topic, "confidence": 0.7, "top_keywords": keywords[:5]}

    return {"topic_id": -1, "topic_label": "General/Other", "confidence": 0.3, "top_keywords": []}

def _rule_based_urgency(text: str) -> dict:
    """Keyword-based urgency detection fallback."""
    text_lower = text.lower()

    critical = ['emergency', 'danger', 'death', 'fire', 'flood', 'collapse', 'urgent', 'आपातकालीन']
    high = ['broken', 'leak', 'overflow', 'no water', 'power cut', 'pothole', 'unsafe', 'टूटा']
    medium = ['repair', 'fix', 'complaint', 'issue', 'dirty', 'delay', 'मरम्मत']

    if any(kw in text_lower for kw in critical):
        return {"urgency": "CRITICAL", "confidence": 0.8, "probabilities": {"LOW": 0.05, "MEDIUM": 0.05, "HIGH": 0.1, "CRITICAL": 0.8}}
    elif any(kw in text_lower for kw in high):
        return {"urgency": "HIGH", "confidence": 0.75, "probabilities": {"LOW": 0.05, "MEDIUM": 0.1, "HIGH": 0.75, "CRITICAL": 0.1}}
    elif any(kw in text_lower for kw in medium):
        return {"urgency": "MEDIUM", "confidence": 0.7, "probabilities": {"LOW": 0.1, "MEDIUM": 0.7, "HIGH": 0.15, "CRITICAL": 0.05}}
    else:
        return {"urgency": "LOW", "confidence": 0.6, "probabilities": {"LOW": 0.6, "MEDIUM": 0.25, "HIGH": 0.1, "CRITICAL": 0.05}}


# ── API Endpoints ───────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "🏛️ Smart Governance AI API",
        "version": API_VERSION,
        "docs": "/docs",
        "endpoints": ["/predict-topic", "/predict-urgency", "/similar-news", "/analyze-text"]
    }

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="healthy",
        models_loaded=models_loaded,
        version=API_VERSION
    )

@app.get("/stats", tags=["System"])
async def system_stats():
    stats = {
        "models_loaded": models_loaded,
        "complaint_index_size": complaint_db.index.ntotal if complaint_db and complaint_db.index else 0,
        "news_index_size": news_db.index.ntotal if news_db and news_db.index else 0,
    }
    return stats

@app.post("/predict-topic", tags=["Complaint Intelligence"])
async def predict_topic(input_data: TextInput):
    """Predict complaint category/topic."""
    clean_text = preprocessor.clean_text(input_data.text)

    if topic_modeler and models_loaded["topic"]:
        try:
            if embedding_generator:
                emb = embedding_generator.generate_embeddings([clean_text], show_progress=False)
                topics, probs = topic_modeler.predict([clean_text], embeddings=emb)
            else:
                topics, probs = topic_modeler.predict([clean_text])

            topic_info = topic_modeler.get_topic_labels()
            topic_id = int(topics[0])
            return {
                "text": input_data.text,
                "clean_text": clean_text,
                "topic_id": topic_id,
                "topic_label": topic_info.get(topic_id, "Unknown"),
                "confidence": round(float(probs[0].max()) if hasattr(probs[0], 'max') else 0.8, 4),
            }
        except Exception as e:
            logger.error(f"Topic prediction error: {e}")

    # Fallback to rule-based
    result = _rule_based_topic(clean_text)
    result["text"] = input_data.text
    result["clean_text"] = clean_text
    result["method"] = "rule-based"
    return result

@app.post("/predict-urgency", tags=["Complaint Intelligence"])
async def predict_urgency(input_data: TextInput):
    """Predict complaint urgency level."""
    clean_text = preprocessor.clean_text(input_data.text)

    if urgency_classifier and models_loaded["urgency"]:
        try:
            predictions = urgency_classifier.predict([clean_text])
            pred = predictions[0]
            return {
                "text": input_data.text,
                "clean_text": clean_text,
                **pred
            }
        except Exception as e:
            logger.error(f"Urgency prediction error: {e}")

    # Fallback
    result = _rule_based_urgency(clean_text)
    result["text"] = input_data.text
    result["clean_text"] = clean_text
    result["method"] = "rule-based"
    return result

@app.post("/similar-news", tags=["Misinformation Detection"])
async def find_similar_news(input_data: TextInput):
    """Find semantically similar news articles (misinformation detection)."""
    clean_text = preprocessor.clean_text(input_data.text)

    if embedding_generator and news_db and models_loaded["faiss_news"]:
        try:
            emb = embedding_generator.generate_embeddings([clean_text], show_progress=False)
            result = news_db.detect_misinformation(clean_text, emb[0])
            result["text"] = input_data.text
            result["clean_text"] = clean_text
            return result
        except Exception as e:
            logger.error(f"Similarity search error: {e}")

    return {
        "text": input_data.text,
        "clean_text": clean_text,
        "is_suspicious": False,
        "confidence": 0.0,
        "num_similar": 0,
        "similar_articles": [],
        "reason": "News FAISS index not loaded. Please build index first.",
        "method": "unavailable"
    }

@app.post("/analyze-text", tags=["Full Pipeline"])
async def analyze_text(input_data: TextInput):
    """Full pipeline analysis: topic + urgency + similarity."""
    clean_text = preprocessor.clean_text(input_data.text)
    language = preprocessor.detect_language(clean_text)

    # Topic
    topic_result = _rule_based_topic(clean_text)
    if topic_modeler and models_loaded["topic"]:
        try:
            if embedding_generator:
                emb = embedding_generator.generate_embeddings([clean_text], show_progress=False)
                topics, probs = topic_modeler.predict([clean_text], embeddings=emb)
            else:
                topics, probs = topic_modeler.predict([clean_text])
            
            topic_info = topic_modeler.get_topic_labels()
            topic_result = {
                "topic_id": int(topics[0]),
                "topic_label": topic_info.get(int(topics[0]), "Unknown"),
                "confidence": round(float(probs[0].max()) if hasattr(probs[0], 'max') else 0.8, 4),
                "method": "bertopic"
            }
        except Exception:
            topic_result["method"] = "rule-based"

    # Urgency
    urgency_result = _rule_based_urgency(clean_text)
    if urgency_classifier and models_loaded["urgency"]:
        try:
            predictions = urgency_classifier.predict([clean_text])
            urgency_result = predictions[0]
            urgency_result["method"] = "xlm-roberta"
        except Exception:
            urgency_result["method"] = "rule-based"

    # Similarity
    similarity_result = {"is_suspicious": False, "num_similar": 0, "similar_articles": [], "method": "unavailable"}
    if embedding_generator and news_db and models_loaded["faiss_news"]:
        try:
            emb = embedding_generator.generate_embeddings([clean_text], show_progress=False)
            similarity_result = news_db.detect_misinformation(clean_text, emb[0])
            similarity_result["method"] = "faiss"
        except Exception:
            pass

    return {
        "text": input_data.text,
        "clean_text": clean_text,
        "language": language,
        "topic": topic_result,
        "urgency": urgency_result,
        "similarity": similarity_result,
    }

@app.post("/upload-csv", tags=["Data Management"])
async def upload_csv(file: UploadFile = File(...), data_type: str = "complaints"):
    """Upload and process a CSV file."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    content = await file.read()
    save_path = PROCESSED_DIR / f"uploaded_{data_type}.csv" if data_type else PROCESSED_DIR / file.filename

    # Save uploaded file
    raw_path = Path(PROCESSED_DIR).parent / "raw" / file.filename
    with open(raw_path, 'wb') as f:
        f.write(content)

    try:
        df = pd.read_csv(raw_path, low_memory=False)
        if data_type == "complaints":
            processed = preprocessor.process_complaints(str(raw_path))
        else:
            processed = preprocessor.process_news(str(raw_path))

        processed.to_csv(save_path, index=False)
        return {
            "message": f"File processed successfully",
            "original_rows": len(df),
            "processed_rows": len(processed),
            "columns": list(processed.columns),
            "saved_to": str(save_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


# ── Run Server ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )
