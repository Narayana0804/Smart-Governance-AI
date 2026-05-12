"""
Data Preprocessing Module for Smart Governance AI.
Handles text cleaning, normalization, and tokenization for multilingual civic data.
Supports: English, Hindi, Telugu, Tamil
"""

import re
import pandas as pd
import numpy as np
from typing import Optional, List
from loguru import logger

try:
    from langdetect import detect
except ImportError:
    detect = None


class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline for civic complaints,
    news articles, and social media data.
    """

    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    EMAIL_PATTERN = re.compile(r'\S+@\S+\.\S+')
    HTML_PATTERN = re.compile(r'<[^>]+>')
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    WHITESPACE_PATTERN = re.compile(r'\s+')

    SUPPORTED_LANGUAGES = ['en', 'hi', 'te', 'ta', 'mr', 'bn', 'gu', 'kn', 'ml']

    def __init__(self, min_length: int = 10, max_length: int = 512):
        self.min_length = min_length
        self.max_length = max_length
        logger.info(f"TextPreprocessor initialized (min_len={min_length}, max_len={max_length})")

    def clean_text(self, text: str) -> str:
        """Clean a single text string - remove URLs, emails, HTML, emojis, special chars."""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""

        text = self.URL_PATTERN.sub(' ', text)
        text = self.EMAIL_PATTERN.sub(' ', text)
        text = self.HTML_PATTERN.sub(' ', text)
        text = self.EMOJI_PATTERN.sub(' ', text)
        # Keep Unicode letters (for Indian languages) and basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\u0900-\u097F\u0C00-\u0C7F\u0B80-\u0BFF\u0980-\u09FF]', ' ', text)
        text = self.WHITESPACE_PATTERN.sub(' ', text)
        text = text.strip()
        return text

    def detect_language(self, text: str) -> str:
        """Detect the language of text. Returns ISO 639-1 code."""
        if detect is None:
            return "en"
        try:
            lang = detect(text)
            return lang if lang in self.SUPPORTED_LANGUAGES else "en"
        except Exception:
            return "unknown"

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: Optional[str] = None,
        detect_lang: bool = True,
        drop_duplicates: bool = True,
        drop_empty: bool = True
    ) -> pd.DataFrame:
        """
        Process an entire DataFrame of text data.

        Args:
            df: Input DataFrame
            text_column: Name of the text column
            label_column: Optional label column to preserve
            detect_lang: Whether to detect language
            drop_duplicates: Remove duplicate texts
            drop_empty: Remove empty/short texts

        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Processing DataFrame with {len(df)} rows...")
        result = df.copy()

        if text_column not in result.columns:
            raise ValueError(f"Column '{text_column}' not found. Available: {list(result.columns)}")

        result = result.rename(columns={text_column: 'text'})

        # Clean text
        logger.info("Cleaning text...")
        result['clean_text'] = result['text'].astype(str).apply(self.clean_text)

        # Drop empty/short texts
        if drop_empty:
            before = len(result)
            result = result[result['clean_text'].str.len() >= self.min_length]
            logger.info(f"Removed {before - len(result)} short/empty texts")

        # Truncate long texts
        result['clean_text'] = result['clean_text'].str[:self.max_length * 5]

        # Drop duplicates
        if drop_duplicates:
            before = len(result)
            result = result.drop_duplicates(subset=['clean_text'])
            logger.info(f"Removed {before - len(result)} duplicate texts")

        # Detect language
        if detect_lang:
            logger.info("Detecting languages...")
            result['language'] = result['clean_text'].apply(self.detect_language)
            lang_dist = result['language'].value_counts().to_dict()
            logger.info(f"Language distribution: {lang_dist}")

        # Handle label column
        if label_column and label_column in df.columns:
            result = result.rename(columns={label_column: 'label'})

        result = result.reset_index(drop=True)
        logger.info(f"Processing complete. {len(result)} rows remaining.")
        return result

    def process_complaints(self, filepath: str) -> pd.DataFrame:
        """Process a complaints CSV file. Auto-detects text/label columns."""
        df = pd.read_csv(filepath, low_memory=False)
        logger.info(f"Loaded complaints file: {filepath} ({len(df)} rows)")

        text_candidates = [
            'Descriptor', 'descriptor', 'Description', 'description',
            'Complaint Type', 'complaint_type', 'complaint',
            'text', 'Text', 'content', 'Content',
            'complaint_text', 'issue', 'Issue'
        ]

        text_col = None
        for col in text_candidates:
            if col in df.columns:
                text_col = col
                break

        if text_col is None:
            for col in df.columns:
                if df[col].dtype == 'object' and df[col].str.len().mean() > 20:
                    text_col = col
                    break

        if text_col is None:
            raise ValueError(f"Could not find text column. Columns available: {list(df.columns)}")

        logger.info(f"Using text column: '{text_col}'")

        label_candidates = ['Complaint Type', 'complaint_type', 'category', 'Category', 'label', 'Label']
        label_col = None
        for col in label_candidates:
            if col in df.columns and col != text_col:
                label_col = col
                break

        return self.process_dataframe(df, text_col, label_col)

    def process_news(self, filepath: str) -> pd.DataFrame:
        """Process a fake news CSV file."""
        df = pd.read_csv(filepath, low_memory=False)
        logger.info(f"Loaded news file: {filepath} ({len(df)} rows)")

        text_candidates = [
            'text', 'Text', 'title', 'Title', 'content', 'Content',
            'article', 'Article', 'news_text', 'headline', 'Headline',
            'statement', 'Statement'
        ]

        text_col = None
        for col in text_candidates:
            if col in df.columns:
                text_col = col
                break

        if text_col is None:
            for col in df.columns:
                if df[col].dtype == 'object' and df[col].str.len().mean() > 30:
                    text_col = col
                    break

        if text_col is None:
            raise ValueError(f"Could not find text column. Columns: {list(df.columns)}")

        label_candidates = ['label', 'Label', 'class', 'Class', 'is_fake', 'fake', 'target']
        label_col = None
        for col in label_candidates:
            if col in df.columns:
                label_col = col
                break

        return self.process_dataframe(df, text_col, label_col)

    def generate_synthetic_urgency_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic urgency labels based on keyword matching.
        Used when no labeled urgency dataset is available.
        """
        critical_keywords = [
            'emergency', 'danger', 'life threatening', 'death', 'collapsed',
            'fire', 'flood', 'accident', 'urgent', 'immediate',
            'hazardous', 'toxic', 'explosion', 'electrocution',
            'आपातकालीन', 'खतरनाक', 'तुरंत', 'अत्यवसर', 'ప్రమాదకరమైన',
        ]
        high_keywords = [
            'broken', 'leak', 'overflow', 'sewage', 'blockage',
            'power cut', 'no water', 'pothole', 'dangerous',
            'unsafe', 'crack', 'damage', 'contaminated',
            'टूटा', 'रिसाव', 'बहाव',
        ]
        medium_keywords = [
            'repair', 'fix', 'complaint', 'issue', 'problem',
            'noise', 'smell', 'dirty', 'maintenance', 'delay',
            'irregular', 'poor', 'bad condition',
            'मरम्मत', 'समस्या', 'शिकायत',
        ]

        def classify_urgency(text: str) -> str:
            text_lower = text.lower()
            if any(kw in text_lower for kw in critical_keywords):
                return "CRITICAL"
            elif any(kw in text_lower for kw in high_keywords):
                return "HIGH"
            elif any(kw in text_lower for kw in medium_keywords):
                return "MEDIUM"
            else:
                return "LOW"

        df = df.copy()
        df['urgency'] = df['clean_text'].apply(classify_urgency)
        dist = df['urgency'].value_counts().to_dict()
        logger.info(f"Urgency distribution: {dist}")
        return df
