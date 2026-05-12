"""
FAISS Vector Database Module for Smart Governance AI.
Enables fast semantic search for complaint similarity and fake news retrieval.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from loguru import logger

try:
    import faiss
except ImportError:
    faiss = None
    logger.warning("FAISS not installed. Install with: pip install faiss-cpu")

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.settings import (
    VECTOR_DB_DIR, FAISS_COMPLAINT_INDEX, FAISS_NEWS_INDEX,
    EMBEDDING_DIMENSION, FAISS_TOP_K, SIMILARITY_THRESHOLD
)


class FAISSVectorDB:
    """
    FAISS-based vector database for semantic similarity search.

    Features:
    - Similarity search for related complaints
    - Duplicate detection
    - Fake news / misinformation retrieval via semantic matching
    """

    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        if faiss is None:
            raise ImportError("FAISS required: pip install faiss-cpu")

        self.dimension = dimension
        self.index = None
        self.metadata = []  # Stores original texts/IDs for retrieval
        logger.info(f"FAISSVectorDB initialized (dimension={dimension})")

    def build_index(
        self,
        embeddings: np.ndarray,
        texts: Optional[List[str]] = None,
        ids: Optional[List[int]] = None,
        use_ivf: bool = False,
        nlist: int = 100
    ):
        """
        Build a FAISS index from embeddings.

        Args:
            embeddings: NumPy array of shape (n, dimension)
            texts: Original texts for retrieval
            ids: Document IDs
            use_ivf: Use IVF index for large datasets (faster but approximate)
            nlist: Number of clusters for IVF
        """
        embeddings = embeddings.astype('float32')
        n, d = embeddings.shape

        if d != self.dimension:
            logger.warning(f"Dimension mismatch: expected {self.dimension}, got {d}. Adjusting.")
            self.dimension = d

        if use_ivf and n > 1000:
            # IVF index for large datasets
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, min(nlist, n // 10))
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = 10
            logger.info(f"IVF index built with {n} vectors, {min(nlist, n // 10)} clusters")
        else:
            # Flat index (exact search)
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine for normalized vecs)
            self.index.add(embeddings)
            logger.info(f"Flat index built with {n} vectors")

        # Store metadata
        self.metadata = []
        for i in range(n):
            entry = {"id": ids[i] if ids else i}
            if texts:
                entry["text"] = texts[i] if i < len(texts) else ""
            self.metadata.append(entry)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = FAISS_TOP_K,
        threshold: float = SIMILARITY_THRESHOLD
    ) -> List[Dict]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector (1, dimension) or (dimension,)
            top_k: Number of results
            threshold: Minimum similarity score

        Returns:
            List of dicts with 'id', 'text', 'score'
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype('float32')
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        is_l2 = getattr(self.index, 'metric_type', None) == faiss.METRIC_L2

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            
            # Convert L2 squared distance to cosine similarity if vectors are normalized
            # L2^2 = 2 - 2 * cos_sim  =>  cos_sim = 1 - L2^2 / 2
            sim_score = float(score)
            if is_l2:
                sim_score = 1.0 - (sim_score / 2.0)
            
            if sim_score < threshold:
                continue
                
            result = {
                "id": int(idx),
                "score": round(sim_score, 4),
            }
            if idx < len(self.metadata):
                result.update(self.metadata[idx])
            results.append(result)

        return results

    def detect_duplicates(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        threshold: float = 0.95
    ) -> List[Tuple[int, int, float]]:
        """
        Detect duplicate/near-duplicate entries.

        Returns:
            List of (idx1, idx2, similarity_score) tuples
        """
        duplicates = []
        embeddings = embeddings.astype('float32')

        temp_index = faiss.IndexFlatIP(embeddings.shape[1])
        temp_index.add(embeddings)

        for i in range(len(embeddings)):
            scores, indices = temp_index.search(embeddings[i:i+1], 5)
            for score, j in zip(scores[0], indices[0]):
                if j > i and score >= threshold:
                    duplicates.append((i, int(j), round(float(score), 4)))

        logger.info(f"Found {len(duplicates)} duplicate pairs (threshold={threshold})")
        return duplicates

    def detect_misinformation(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.70
    ) -> Dict:
        """
        Detect potential misinformation by finding semantically similar articles.

        Returns:
            Dict with 'is_suspicious', 'confidence', 'similar_articles'
        """
        similar = self.search(query_embedding, top_k=top_k, threshold=threshold)

        if not similar:
            return {
                "is_suspicious": False,
                "confidence": 0.0,
                "similar_articles": [],
                "reason": "No similar articles found"
            }

        max_similarity = max(s['score'] for s in similar)
        avg_similarity = sum(s['score'] for s in similar) / len(similar)

        is_suspicious = max_similarity > 0.90 or (len(similar) >= 3 and avg_similarity > 0.85)

        return {
            "is_suspicious": is_suspicious,
            "confidence": round(max_similarity, 4),
            "num_similar": len(similar),
            "avg_similarity": round(avg_similarity, 4),
            "similar_articles": similar,
            "reason": "High semantic overlap detected" if is_suspicious else "Low similarity - likely original"
        }

    def save_index(self, path: Path):
        """Save the FAISS index and metadata."""
        if self.index is None:
            raise RuntimeError("No index to save.")

        path = Path(path)
        faiss.write_index(self.index, str(path))

        meta_path = path.with_suffix('.meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Index saved to {path}")

    def load_index(self, path: Path):
        """Load a FAISS index and metadata."""
        path = Path(path)
        self.index = faiss.read_index(str(path))

        meta_path = path.with_suffix('.meta.json')
        if meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

        logger.info(f"Index loaded from {path} ({self.index.ntotal} vectors)")
