"""
Urgency Detection Model for Smart Governance AI.
Fine-tunes XLM-RoBERTa for multi-class urgency classification.
Classes: LOW, MEDIUM, HIGH, CRITICAL
"""

import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from loguru import logger

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback
    )
    from datasets import Dataset
except ImportError:
    logger.warning("transformers/datasets not installed.")

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.settings import (
    URGENCY_MODEL_NAME, URGENCY_LABELS, URGENCY_NUM_LABELS,
    URGENCY_MODEL_DIR, TRAINING_CONFIG
)


class UrgencyClassifier:
    """
    XLM-RoBERTa-based urgency classifier for civic complaints.
    Supports multilingual input (English, Hindi, Telugu, Tamil).
    """

    def __init__(self, model_name: str = URGENCY_MODEL_NAME, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(URGENCY_LABELS)
        self.tokenizer = None
        self.model = None
        self.is_trained = False

        logger.info(f"UrgencyClassifier initialized (model={model_name}, device={self.device})")

    def _load_base_model(self):
        """Load the base pre-trained model and tokenizer."""
        logger.info(f"Loading base model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=URGENCY_NUM_LABELS,
            problem_type="single_label_classification"
        )
        self.model.to(self.device)
        logger.info("Base model loaded.")

    def _tokenize_function(self, examples):
        """Tokenize examples for the model."""
        return self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=TRAINING_CONFIG['max_seq_length']
        )

    def prepare_dataset(self, df: pd.DataFrame, text_col: str = 'clean_text', label_col: str = 'urgency') -> Dataset:
        """
        Prepare a HuggingFace Dataset from DataFrame.

        Args:
            df: DataFrame with text and label columns
            text_col: Name of text column
            label_col: Name of label column

        Returns:
            HuggingFace Dataset
        """
        data = df[[text_col, label_col]].copy()
        data = data.rename(columns={text_col: 'text', label_col: 'label_str'})
        data['label'] = self.label_encoder.transform(data['label_str'])
        data = data.drop('label_str', axis=1)

        dataset = Dataset.from_pandas(data)
        dataset = dataset.map(self._tokenize_function, batched=True)
        dataset = dataset.remove_columns(['text'])
        dataset.set_format('torch')

        logger.info(f"Dataset prepared: {len(dataset)} samples")
        return dataset

    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        return {"accuracy": acc, "f1": f1}

    def train(
        self,
        train_df: pd.DataFrame,
        eval_df: Optional[pd.DataFrame] = None,
        text_col: str = 'clean_text',
        label_col: str = 'urgency',
        output_dir: Path = URGENCY_MODEL_DIR,
    ):
        """
        Fine-tune the urgency classification model.

        Args:
            train_df: Training DataFrame
            eval_df: Evaluation DataFrame (optional, will split if not provided)
            text_col: Text column name
            label_col: Label column name
            output_dir: Directory to save model
        """
        self._load_base_model()

        # Split if no eval set provided
        if eval_df is None:
            from sklearn.model_selection import train_test_split
            train_df, eval_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df[label_col])

        train_dataset = self.prepare_dataset(train_df, text_col, label_col)
        eval_dataset = self.prepare_dataset(eval_df, text_col, label_col)

        training_args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=TRAINING_CONFIG['num_epochs'],
            per_device_train_batch_size=TRAINING_CONFIG['batch_size'],
            per_device_eval_batch_size=TRAINING_CONFIG['batch_size'],
            warmup_steps=TRAINING_CONFIG['warmup_steps'],
            weight_decay=TRAINING_CONFIG['weight_decay'],
            learning_rate=TRAINING_CONFIG['learning_rate'],
            eval_strategy="steps",
            eval_steps=TRAINING_CONFIG['eval_steps'],
            save_steps=TRAINING_CONFIG['save_steps'],
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_dir=str(output_dir / "logs"),
            logging_steps=50,
            report_to="none",
            fp16=self.device == "cuda",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        logger.info("Starting training...")
        trainer.train()

        # Evaluate
        results = trainer.evaluate()
        logger.info(f"Evaluation results: {results}")

        # Save
        self.save_model(output_dir)
        self.is_trained = True

        return results

    def predict(self, texts: List[str]) -> List[Dict]:
        """
        Predict urgency for a list of texts.

        Returns:
            List of dicts with 'label', 'confidence', and 'probabilities'
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.model.eval()
        results = []

        for text in texts:
            inputs = self.tokenizer(
                text, return_tensors="pt",
                padding=True, truncation=True,
                max_length=TRAINING_CONFIG['max_seq_length']
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]

            pred_idx = probs.argmax().item()
            pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
            confidence = probs[pred_idx].item()

            results.append({
                'urgency': pred_label,
                'confidence': round(confidence, 4),
                'probabilities': {
                    label: round(probs[i].item(), 4)
                    for i, label in enumerate(URGENCY_LABELS)
                }
            })

        return results

    def save_model(self, path: Path = URGENCY_MODEL_DIR):
        """Save the trained model, tokenizer, and label encoder."""
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        joblib.dump(self.label_encoder, str(path / "label_encoder.pkl"))
        logger.info(f"Model saved to {path}")

    def load_model(self, path: Path = URGENCY_MODEL_DIR):
        """Load a saved model."""
        logger.info(f"Loading model from {path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path))
        self.model.to(self.device)
        self.model.eval()

        le_path = path / "label_encoder.pkl"
        if le_path.exists():
            self.label_encoder = joblib.load(str(le_path))

        self.is_trained = True
        logger.info("Model loaded successfully.")
