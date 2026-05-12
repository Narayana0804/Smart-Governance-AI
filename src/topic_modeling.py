"""
Topic Modeling Module for Smart Governance AI.
Uses BERTopic (UMAP + HDBSCAN) for automatic complaint categorization.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from loguru import logger

try:
    from bertopic import BERTopic
except ImportError:
    BERTopic = None
    logger.warning("BERTopic not installed. Install with: pip install bertopic")

try:
    from umap import UMAP
except ImportError:
    UMAP = None

try:
    from hdbscan import HDBSCAN
except ImportError:
    HDBSCAN = None

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.settings import TOPIC_MODEL_PATH, EMBEDDINGS_DIR


class TopicModeler:
    """
    BERTopic-based topic modeling for civic complaint categorization.

    Pipeline: Embeddings → UMAP → HDBSCAN → BERTopic
    Expected topics: Water Issues, Garbage, Road Damage, Electricity, etc.
    """

    def __init__(
        self,
        n_topics: Optional[int] = None,
        min_topic_size: int = 15,
        umap_n_neighbors: int = 15,
        umap_n_components: int = 5,
        umap_min_dist: float = 0.0,
    ):
        if BERTopic is None:
            raise ImportError("BERTopic required: pip install bertopic")

        self.n_topics = n_topics
        self.min_topic_size = min_topic_size

        # Configure UMAP for dimensionality reduction
        umap_model = None
        if UMAP is not None:
            umap_model = UMAP(
                n_neighbors=umap_n_neighbors,
                n_components=umap_n_components,
                min_dist=umap_min_dist,
                metric='cosine',
                random_state=42
            )

        # Configure HDBSCAN for clustering
        hdbscan_model = None
        if HDBSCAN is not None:
            hdbscan_model = HDBSCAN(
                min_cluster_size=min_topic_size,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )

        # Initialize BERTopic
        self.model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            nr_topics=n_topics,
            top_n_words=10,
            verbose=True,
            calculate_probabilities=True
        )

        logger.info(f"TopicModeler initialized (min_topic_size={min_topic_size})")

    def fit_transform(
        self,
        texts: List[str],
        embeddings: Optional[np.ndarray] = None
    ) -> Tuple[List[int], np.ndarray]:
        """
        Fit the topic model and transform texts.

        Args:
            texts: List of text documents
            embeddings: Pre-computed embeddings (recommended)

        Returns:
            Tuple of (topic_assignments, probabilities)
        """
        logger.info(f"Fitting topic model on {len(texts)} documents...")
        topics, probs = self.model.fit_transform(texts, embeddings=embeddings)

        topic_info = self.model.get_topic_info()
        logger.info(f"Discovered {len(topic_info) - 1} topics (excluding outliers)")
        logger.info(f"\nTopic Distribution:\n{topic_info[['Topic', 'Count', 'Name']].to_string()}")

        return topics, probs

    def predict(self, texts: List[str], embeddings: Optional[np.ndarray] = None) -> Tuple[List[int], np.ndarray]:
        """Predict topics for new texts."""
        topics, probs = self.model.transform(texts, embeddings=embeddings)
        return topics, probs

    def get_topic_info(self) -> pd.DataFrame:
        """Get topic information."""
        return self.model.get_topic_info()

    def get_topic_labels(self) -> Dict[int, str]:
        """Get topic labels as a dictionary."""
        topic_info = self.model.get_topic_info()
        return dict(zip(topic_info['Topic'], topic_info['Name']))

    def save_model(self, path: Path = TOPIC_MODEL_PATH):
        """Save the topic model."""
        # BERTopic has its own save method
        save_dir = str(path).replace('.pkl', '')
        self.model.save(save_dir)
        logger.info(f"Topic model saved to {save_dir}")

    def load_model(self, path: Path = TOPIC_MODEL_PATH):
        """Load a saved topic model."""
        load_dir = str(path).replace('.pkl', '')
        self.model = BERTopic.load(load_dir)
        logger.info(f"Topic model loaded from {load_dir}")

    def visualize_topics(self):
        """Generate topic visualization (returns plotly figure)."""
        try:
            fig = self.model.visualize_topics()
            return fig
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return None

    def visualize_barchart(self, top_n_topics: int = 10):
        """Generate topic barchart visualization."""
        try:
            fig = self.model.visualize_barchart(top_n_topics=top_n_topics)
            return fig
        except Exception as e:
            logger.error(f"Barchart visualization failed: {e}")
            return None
