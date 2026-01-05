"""
ML Baseline Model Module
========================

This module implements the grounded baseline using:
1. Transformer-based text embeddings (Sentence-BERT)
2. Supervised regression models for each OCEAN trait

Supports multiple embedding models and regression algorithms.
Reports Pearson correlation, MAE, and R².
"""

import os
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OCEAN traits
OCEAN_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


@dataclass
class MLConfig:
    """Configuration for ML baseline model."""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    regressor_type: str = "ridge"  # ridge, svr, mlp, linear
    cv_folds: int = 5
    random_seed: int = 42
    batch_size: int = 32
    device: str = "auto"
    
    # Ridge parameters
    ridge_alpha: float = 1.0
    
    # SVR parameters
    svr_kernel: str = "rbf"
    svr_C: float = 1.0
    svr_epsilon: float = 0.1
    
    # MLP parameters
    mlp_hidden_layers: Tuple = (256, 128, 64)
    mlp_activation: str = "relu"
    mlp_max_iter: int = 500
    mlp_learning_rate: float = 0.001


class TextEmbedder:
    """
    Text embedding using Sentence Transformers.
    
    Generates dense vector representations of text that capture semantic meaning.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "auto"):
        """
        Initialize the text embedder.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to use (auto, cpu, cuda, mps)
        """
        self.model_name = model_name
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def embed(
        self, 
        texts: List[str], 
        batch_size: int = 32, 
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.model.encode([text], convert_to_numpy=True)[0]


class TraitRegressor:
    """
    Regression model for a single OCEAN trait.
    
    Supports multiple regression algorithms with hyperparameter tuning.
    """
    
    def __init__(self, trait_name: str, config: MLConfig):
        """
        Initialize the trait regressor.
        
        Args:
            trait_name: Name of the OCEAN trait
            config: ML configuration
        """
        self.trait_name = trait_name
        self.config = config
        self.scaler = StandardScaler()
        self.model = self._create_model()
        self.is_fitted = False
        self.cv_scores = None
    
    def _create_model(self):
        """Create the regression model based on config."""
        if self.config.regressor_type == "ridge":
            return Ridge(
                alpha=self.config.ridge_alpha,
                random_state=self.config.random_seed
            )
        elif self.config.regressor_type == "svr":
            return SVR(
                kernel=self.config.svr_kernel,
                C=self.config.svr_C,
                epsilon=self.config.svr_epsilon
            )
        elif self.config.regressor_type == "mlp":
            return MLPRegressor(
                hidden_layer_sizes=self.config.mlp_hidden_layers,
                activation=self.config.mlp_activation,
                max_iter=self.config.mlp_max_iter,
                learning_rate_init=self.config.mlp_learning_rate,
                random_state=self.config.random_seed,
                early_stopping=True,
                validation_fraction=0.1
            )
        elif self.config.regressor_type == "linear":
            return LinearRegression()
        else:
            raise ValueError(f"Unknown regressor type: {self.config.regressor_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, cross_validate: bool = True):
        """
        Fit the regression model.
        
        Args:
            X: Feature matrix (embeddings)
            y: Target values (trait scores)
            cross_validate: Whether to perform cross-validation
        """
        logger.info(f"Training regressor for {self.trait_name}...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Cross-validation
        if cross_validate:
            kfold = KFold(n_splits=self.config.cv_folds, shuffle=True, 
                         random_state=self.config.random_seed)
            self.cv_scores = cross_val_score(
                self.model, X_scaled, y, cv=kfold, scoring='r2'
            )
            logger.info(f"CV R² scores for {self.trait_name}: {self.cv_scores.mean():.4f} (+/- {self.cv_scores.std():.4f})")
        
        # Fit on full data
        self.model.fit(X_scaled, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict trait scores.
        
        Args:
            X: Feature matrix (embeddings)
            
        Returns:
            Predicted trait scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Clip to valid range [0, 1]
        predictions = np.clip(predictions, 0, 1)
        
        return predictions


class MLBaselineModel:
    """
    Complete ML baseline model for personality trait prediction.
    
    Combines:
    - Text embeddings from Sentence Transformers
    - Per-trait regression models
    - Cross-validation and evaluation
    """
    
    def __init__(self, config: MLConfig = None):
        """
        Initialize the ML baseline model.
        
        Args:
            config: ML configuration
        """
        self.config = config or MLConfig()
        self.embedder = TextEmbedder(
            self.config.embedding_model, 
            self.config.device
        )
        self.regressors: Dict[str, TraitRegressor] = {}
        self.is_fitted = False
        self.train_embeddings = None
        self.train_distribution = {}
    
    def fit(
        self, 
        texts: List[str], 
        labels: Dict[str, np.ndarray],
        cross_validate: bool = True
    ):
        """
        Fit the model on training data.
        
        Args:
            texts: List of training texts
            labels: Dictionary mapping trait names to score arrays
            cross_validate: Whether to perform cross-validation
        """
        logger.info(f"Fitting ML baseline model on {len(texts)} samples...")
        
        # Generate embeddings
        embeddings = self.embedder.embed(texts, batch_size=self.config.batch_size)
        self.train_embeddings = embeddings
        
        # Store training distribution for percentile calculation
        for trait in OCEAN_TRAITS:
            if trait in labels:
                self.train_distribution[trait] = np.sort(labels[trait])
        
        # Train per-trait regressors
        for trait in OCEAN_TRAITS:
            if trait in labels:
                self.regressors[trait] = TraitRegressor(trait, self.config)
                self.regressors[trait].fit(embeddings, labels[trait], cross_validate)
        
        self.is_fitted = True
        logger.info("ML baseline model training complete")
    
    def predict(
        self, 
        texts: List[str],
        return_embeddings: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Predict trait scores for texts.
        
        Args:
            texts: List of texts to predict
            return_embeddings: Whether to return embeddings
            
        Returns:
            Dictionary mapping trait names to predicted score arrays
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Generate embeddings
        embeddings = self.embedder.embed(texts, batch_size=self.config.batch_size, show_progress=False)
        
        # Predict each trait
        predictions = {}
        for trait, regressor in self.regressors.items():
            predictions[trait] = regressor.predict(embeddings)
        
        if return_embeddings:
            return predictions, embeddings
        
        return predictions
    
    def predict_single(self, text: str) -> Dict[str, float]:
        """
        Predict trait scores for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping trait names to predicted scores
        """
        predictions = self.predict([text])
        return {trait: float(scores[0]) for trait, scores in predictions.items()}
    
    def get_percentile(self, trait: str, score: float) -> float:
        """
        Calculate percentile for a score based on training distribution.
        
        Args:
            trait: Trait name
            score: Predicted score
            
        Returns:
            Percentile (0-100)
        """
        if trait not in self.train_distribution:
            return 50.0
        
        distribution = self.train_distribution[trait]
        percentile = np.searchsorted(distribution, score) / len(distribution) * 100
        return float(percentile)
    
    def evaluate(
        self, 
        texts: List[str], 
        labels: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on test data.
        
        Args:
            texts: Test texts
            labels: Ground truth labels
            
        Returns:
            Dictionary of metrics per trait
        """
        logger.info(f"Evaluating on {len(texts)} samples...")
        
        predictions = self.predict(texts)
        metrics = {}
        
        for trait in self.regressors.keys():
            if trait in labels:
                y_true = labels[trait]
                y_pred = predictions[trait]
                
                # Calculate metrics
                pearson_corr, pearson_p = pearsonr(y_true, y_pred)
                spearman_corr, spearman_p = spearmanr(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                
                metrics[trait] = {
                    "pearson_correlation": float(pearson_corr),
                    "pearson_p_value": float(pearson_p),
                    "spearman_correlation": float(spearman_corr),
                    "spearman_p_value": float(spearman_p),
                    "mae": float(mae),
                    "r2": float(r2),
                    "rmse": float(rmse)
                }
        
        # Calculate average metrics
        avg_metrics = {}
        for metric_name in ["pearson_correlation", "mae", "r2", "rmse"]:
            values = [m[metric_name] for m in metrics.values()]
            avg_metrics[metric_name] = float(np.mean(values))
        metrics["average"] = avg_metrics
        
        return metrics
    
    def save(self, filepath: str):
        """Save model to disk."""
        logger.info(f"Saving model to {filepath}")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model components
        model_data = {
            "config": self.config,
            "regressors": self.regressors,
            "train_distribution": self.train_distribution,
            "embedding_model": self.config.embedding_model
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str):
        """Load model from disk."""
        logger.info(f"Loading model from {filepath}")
        
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        
        self.config = model_data["config"]
        self.regressors = model_data["regressors"]
        self.train_distribution = model_data["train_distribution"]
        
        # Reload embedder with saved model name
        self.embedder = TextEmbedder(
            model_data["embedding_model"],
            self.config.device
        )
        
        self.is_fitted = True


def train_ml_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame = None,
    config: MLConfig = None,
    text_col: str = "text"
) -> Tuple[MLBaselineModel, Dict]:
    """
    Convenience function to train ML baseline model.
    
    Args:
        train_df: Training data
        val_df: Validation data (optional)
        config: ML configuration
        text_col: Name of text column
        
    Returns:
        Trained model and evaluation metrics
    """
    config = config or MLConfig()
    model = MLBaselineModel(config)
    
    # Prepare training data
    texts = train_df[text_col].tolist()
    labels = {trait: train_df[trait].values for trait in OCEAN_TRAITS if trait in train_df.columns}
    
    # Train
    model.fit(texts, labels)
    
    # Evaluate on validation set
    metrics = {}
    if val_df is not None:
        val_texts = val_df[text_col].tolist()
        val_labels = {trait: val_df[trait].values for trait in OCEAN_TRAITS if trait in val_df.columns}
        metrics = model.evaluate(val_texts, val_labels)
    
    return model, metrics


if __name__ == "__main__":
    # Test ML baseline
    from .data_loader import load_data, DataConfig
    
    # Load data
    data_config = DataConfig(min_samples=1000)
    train_df, val_df, test_df, analysis = load_data(data_config)
    
    # Train model
    ml_config = MLConfig(regressor_type="ridge")
    model, val_metrics = train_ml_baseline(train_df, val_df, ml_config)
    
    print("\n=== Validation Metrics ===")
    for trait, metrics in val_metrics.items():
        if trait != "average":
            print(f"\n{trait.capitalize()}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
    
    print(f"\n=== Average Metrics ===")
    for metric_name, value in val_metrics["average"].items():
        print(f"  {metric_name}: {value:.4f}")
    
    # Test single prediction
    sample_text = "I love exploring new ideas and meeting new people. I'm always curious about the world."
    prediction = model.predict_single(sample_text)
    print(f"\n=== Sample Prediction ===")
    print(f"Text: {sample_text[:100]}...")
    for trait, score in prediction.items():
        percentile = model.get_percentile(trait, score)
        print(f"  {trait}: {score:.3f} ({percentile:.1f}th percentile)")
