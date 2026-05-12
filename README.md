# 🏛️ Smart Governance AI

## AI-Powered Smart Governance + Regional Language Misinformation Detection Platform

A multi-module civic intelligence system featuring:
- **Complaint Intelligence Engine** - Auto-categorization, urgency detection, trend analysis
- **Misinformation Detector** - Semantic similarity-based fake news detection
- **Multilingual NLP** - Hindi, Telugu, Tamil, English support.

### Tech Stack
| Component | Technology |
|-----------|-----------|
| NLP Models | XLM-RoBERTa, BERTopic, Sentence Transformers |
| Vector Search | FAISS |
| Backend | FastAPI |
| Dashboard | Streamlit |
| Deployment | Docker |

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
python -m api.main

# Start Dashboard (new terminal)
streamlit run dashboard/app.py
```

### Project Structure
```
smart-governance-ai/
├── data/raw/              # Raw datasets
├── data/processed/        # Cleaned data
├── data/embeddings/       # Generated embeddings
├── data/vector_db/        # FAISS indices
├── notebooks/             # Jupyter/Colab notebooks
├── src/                   # Core source modules
│   ├── preprocessing.py   # Text cleaning & normalization
│   ├── embeddings.py      # Sentence Transformer embeddings
│   ├── topic_modeling.py  # BERTopic complaint clustering
│   ├── urgency_model.py   # XLM-RoBERTa urgency classifier
│   └── vector_db.py       # FAISS similarity search
├── api/main.py            # FastAPI backend
├── dashboard/app.py       # Streamlit dashboard
├── configs/settings.py    # Configuration
├── tests/                 # Test suite
├── Dockerfile             # Container setup
└── docker-compose.yml     # Multi-service orchestration
```

### API Endpoints
| Endpoint | Purpose |
|----------|---------|
| `POST /predict-topic` | Complaint categorization |
| `POST /predict-urgency` | Urgency level detection |
| `POST /similar-news` | Misinformation detection |
| `POST /analyze-text` | Full pipeline analysis |
| `GET /health` | System health check |
