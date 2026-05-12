"""
Embedding Generation Module for Smart Governance AI.
Generates semantic embeddings using Sentence Transformers.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    logger.warning("sentence-transformers not installed.")

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.settings import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION, EMBEDDINGS_DIR


class EmbeddingGenerator:
    """
    Generate semantic embeddings for text using Sentence Transformers.

    Models supported:
    - all-MiniLM-L6-v2 (fast, 384-dim)
    - paraphrase-multilingual-MiniLM-L12-v2 (multilingual, 384-dim)
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: str = "auto"):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers required: pip install sentence-transformers")

        self.model_name = model_name
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")

    def generate_embeddings(
        self,
        texts: Union[List[str], pd.Series],
        batch_size: int = 64,
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        texts = [str(t) if t and str(t).strip() else "empty" for t in texts]

        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        logger.info(f"Embeddings generated: shape={embeddings.shape}")
        return embeddings

    def save_embeddings(self, embeddings: np.ndarray, filename: str, save_dir: Path = EMBEDDINGS_DIR):
        """Save embeddings to .npy file."""
        save_path = save_dir / filename
        np.save(str(save_path), embeddings)
        logger.info(f"Embeddings saved to {save_path}")

    def load_embeddings(self, filename: str, load_dir: Path = EMBEDDINGS_DIR) -> np.ndarray:
        """Load embeddings from .npy file."""
        load_path = load_dir / filename
        embeddings = np.load(str(load_path))
        logger.info(f"Embeddings loaded from {load_path} (shape: {embeddings.shape})")
        return embeddings

    def generate_and_save(
        self,
        texts: Union[List[str], pd.Series],
        filename: str,
        batch_size: int = 64
    ) -> np.ndarray:
        """Generate embeddings and save to file."""
        embeddings = self.generate_embeddings(texts, batch_size=batch_size)
        self.save_embeddings(embeddings, filename)
        return embeddings
