"""
Unified Inference Pipeline
==========================

This module provides a unified interface for personality prediction:
1. Single text prediction
2. Batch prediction
3. Complete analysis with explainability
4. Easy-to-use API for deployment
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

from .data_loader import TextPreprocessor, DataConfig
from .ml_baseline import MLBaselineModel, MLConfig
from .llm_inference import LLMInferenceEngine, LLMConfig, create_llm_engine, MockLLMEngine
from .ensemble import EnsembleModel, EnsembleConfig, PersonalityPrediction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OCEAN traits
OCEAN_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


@dataclass
class PipelineConfig:
    """Configuration for the inference pipeline."""
    # ML model settings
    ml_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ml_regressor_type: str = "ridge"
    
    # LLM settings
    llm_model: str = "gemini-1.5-flash"
    llm_api_key: str = ""
    use_mock_llm: bool = False  # Use mock LLM for testing
    
    # Ensemble settings
    ml_weight: float = 0.6
    llm_weight: float = 0.4
    calibration_enabled: bool = True
    
    # Category thresholds
    low_threshold: float = 33.0
    high_threshold: float = 67.0
    
    # Text preprocessing
    min_text_length: int = 50
    max_text_length: int = 2048


class PersonalityPredictor:
    """
    Main personality prediction pipeline.
    
    Combines:
    - ML baseline (transformer embeddings + regression)
    - LLM inference (Gemini-based analysis)
    - Ensemble combination and calibration
    
    Provides:
    - Continuous OCEAN scores (0-1)
    - Percentiles
    - Category labels (Low/Medium/High)
    - Evidence sentences
    - Justifications
    """
    
    def __init__(self, config: PipelineConfig = None):
        """
        Initialize the personality predictor.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        
        # Initialize components (lazy loading)
        self._ml_model: Optional[MLBaselineModel] = None
        self._llm_engine: Optional[LLMInferenceEngine] = None
        self._ensemble: Optional[EnsembleModel] = None
        self._preprocessor: Optional[TextPreprocessor] = None
        
        self.is_trained = False
    
    @property
    def preprocessor(self) -> TextPreprocessor:
        """Get or create text preprocessor."""
        if self._preprocessor is None:
            data_config = DataConfig(
                min_text_length=self.config.min_text_length,
                max_text_length=self.config.max_text_length
            )
            self._preprocessor = TextPreprocessor(data_config)
        return self._preprocessor
    
    @property
    def ml_model(self) -> MLBaselineModel:
        """Get or create ML model."""
        if self._ml_model is None:
            ml_config = MLConfig(
                embedding_model=self.config.ml_embedding_model,
                regressor_type=self.config.ml_regressor_type
            )
            self._ml_model = MLBaselineModel(ml_config)
        return self._ml_model
    
    @property
    def llm_engine(self) -> Union[LLMInferenceEngine, MockLLMEngine]:
        """Get or create LLM engine."""
        if self._llm_engine is None:
            if self.config.use_mock_llm:
                self._llm_engine = MockLLMEngine()
            else:
                llm_config = LLMConfig(
                    api_key=self.config.llm_api_key or os.getenv("GEMINI_API_KEY", ""),
                    model=self.config.llm_model
                )
                self._llm_engine = create_llm_engine(llm_config)
        return self._llm_engine
    
    @property
    def ensemble(self) -> EnsembleModel:
        """Get or create ensemble model."""
        if self._ensemble is None:
            ensemble_config = EnsembleConfig(
                ml_weight=self.config.ml_weight,
                llm_weight=self.config.llm_weight,
                calibration_enabled=self.config.calibration_enabled,
                low_threshold=self.config.low_threshold,
                high_threshold=self.config.high_threshold
            )
            self._ensemble = EnsembleModel(ensemble_config)
        return self._ensemble
    
    def train(
        self,
        train_texts: List[str],
        train_labels: Dict[str, np.ndarray],
        val_texts: List[str] = None,
        val_labels: Dict[str, np.ndarray] = None,
        use_llm_for_training: bool = False
    ):
        """
        Train the prediction pipeline.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels per trait
            val_texts: Validation texts
            val_labels: Validation labels per trait
            use_llm_for_training: Whether to use LLM predictions for ensemble training
        """
        logger.info(f"Training pipeline on {len(train_texts)} samples...")
        
        # Train ML model
        logger.info("Training ML baseline...")
        self.ml_model.fit(train_texts, train_labels)
        
        # Get ML predictions on training data for ensemble calibration
        ml_predictions = self.ml_model.predict(train_texts)
        
        # Get LLM predictions (or use mock)
        if use_llm_for_training:
            logger.info("Getting LLM predictions for ensemble training...")
            llm_results = self.llm_engine.predict_batch(train_texts)
            llm_predictions = self.llm_engine.get_scores_array(llm_results)
        else:
            # Use ML predictions with noise as proxy for LLM
            logger.info("Using ML predictions as LLM proxy for training...")
            llm_predictions = {
                trait: np.clip(ml_predictions[trait] + np.random.randn(len(train_texts)) * 0.05, 0, 1)
                for trait in ml_predictions
            }
        
        # Train ensemble
        logger.info("Training ensemble...")
        self.ensemble.fit(ml_predictions, llm_predictions, train_labels)
        
        self.is_trained = True
        logger.info("Training complete!")
        
        # Evaluate on validation set if provided
        if val_texts is not None and val_labels is not None:
            logger.info("Evaluating on validation set...")
            val_predictions = self.predict_batch(val_texts, include_llm=use_llm_for_training)
            
            # Calculate simple metrics
            for trait in OCEAN_TRAITS:
                if trait in val_labels:
                    pred_scores = np.array([p.scores[trait] for p in val_predictions])
                    true_scores = val_labels[trait]
                    from scipy.stats import pearsonr
                    r, _ = pearsonr(true_scores, pred_scores)
                    mae = np.mean(np.abs(pred_scores - true_scores))
                    logger.info(f"  {trait}: Pearson r = {r:.4f}, MAE = {mae:.4f}")
    
    def predict(
        self,
        text: str,
        include_llm: bool = True,
        include_evidence: bool = True
    ) -> PersonalityPrediction:
        """
        Predict personality traits for a single text.
        
        Args:
            text: Input text
            include_llm: Whether to use LLM for prediction
            include_evidence: Whether to include evidence extraction
            
        Returns:
            PersonalityPrediction with scores, percentiles, categories, and evidence
        """
        # Preprocess text
        processed_text = self.preprocessor.preprocess(text)
        if processed_text is None:
            logger.warning("Text too short after preprocessing, using original")
            processed_text = text
        
        # Get ML prediction
        ml_scores = self.ml_model.predict_single(processed_text)
        
        # Get LLM prediction
        if include_llm:
            llm_result = self.llm_engine.predict(processed_text)
            llm_scores = llm_result.scores
            evidence = llm_result.evidence if include_evidence else {}
            justifications = llm_result.justifications if include_evidence else {}
        else:
            llm_scores = ml_scores  # Use ML scores as fallback
            evidence = {}
            justifications = {}
        
        # Combine with ensemble
        prediction = self.ensemble.predict(
            processed_text,
            ml_scores,
            llm_scores,
            evidence,
            justifications
        )
        
        return prediction
    
    def predict_batch(
        self,
        texts: List[str],
        include_llm: bool = True,
        include_evidence: bool = True,
        show_progress: bool = True
    ) -> List[PersonalityPrediction]:
        """
        Predict personality traits for multiple texts.
        
        Args:
            texts: List of input texts
            include_llm: Whether to use LLM for prediction
            include_evidence: Whether to include evidence extraction
            show_progress: Whether to show progress bar
            
        Returns:
            List of PersonalityPrediction objects
        """
        from tqdm import tqdm
        
        # Preprocess texts
        processed_texts = []
        for text in texts:
            processed = self.preprocessor.preprocess(text)
            processed_texts.append(processed if processed else text)
        
        # Get ML predictions
        ml_predictions = self.ml_model.predict(processed_texts)
        
        # Get LLM predictions
        if include_llm:
            llm_results = self.llm_engine.predict_batch(processed_texts, show_progress)
            llm_predictions = self.llm_engine.get_scores_array(llm_results)
            evidences = [r.evidence for r in llm_results]
            justifications = [r.justifications for r in llm_results]
        else:
            llm_predictions = ml_predictions
            evidences = [{}] * len(texts)
            justifications = [{}] * len(texts)
        
        # Combine with ensemble
        predictions = self.ensemble.predict_batch(
            processed_texts,
            ml_predictions,
            llm_predictions,
            evidences if include_evidence else None,
            justifications if include_evidence else None
        )
        
        return predictions
    
    def analyze(self, text: str) -> Dict:
        """
        Perform complete personality analysis with explanation.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with complete analysis
        """
        prediction = self.predict(text, include_llm=True, include_evidence=True)
        
        analysis = {
            "input_text": text[:500] + "..." if len(text) > 500 else text,
            "text_length": len(text),
            "traits": {}
        }
        
        for trait in OCEAN_TRAITS:
            if trait in prediction.scores:
                analysis["traits"][trait] = {
                    "score": round(prediction.scores[trait], 3),
                    "percentile": round(prediction.percentiles.get(trait, 50), 1),
                    "category": prediction.categories.get(trait, "Medium"),
                    "confidence": round(prediction.confidence.get(trait, 0.5), 3),
                    "evidence": prediction.evidence.get(trait, []),
                    "justification": prediction.justifications.get(trait, ""),
                    "ml_score": round(prediction.ml_scores.get(trait, 0.5), 3),
                    "llm_score": round(prediction.llm_scores.get(trait, 0.5), 3)
                }
        
        return analysis
    
    def save(self, model_dir: str):
        """
        Save the trained pipeline.
        
        Args:
            model_dir: Directory to save models
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save ML model
        self.ml_model.save(os.path.join(model_dir, "ml_model.pkl"))
        
        # Save ensemble
        self.ensemble.save(os.path.join(model_dir, "ensemble.pkl"))
        
        # Save config
        config_dict = {
            "ml_embedding_model": self.config.ml_embedding_model,
            "ml_regressor_type": self.config.ml_regressor_type,
            "llm_model": self.config.llm_model,
            "ml_weight": self.config.ml_weight,
            "llm_weight": self.config.llm_weight,
            "calibration_enabled": self.config.calibration_enabled,
            "low_threshold": self.config.low_threshold,
            "high_threshold": self.config.high_threshold
        }
        
        with open(os.path.join(model_dir, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Saved pipeline to {model_dir}")
    
    def load(self, model_dir: str):
        """
        Load a trained pipeline.
        
        Args:
            model_dir: Directory with saved models
        """
        # Load config
        with open(os.path.join(model_dir, "config.json"), 'r') as f:
            config_dict = json.load(f)
        
        self.config = PipelineConfig(**config_dict)
        
        # Reset components
        self._ml_model = None
        self._ensemble = None
        
        # Load ML model
        self.ml_model.load(os.path.join(model_dir, "ml_model.pkl"))
        
        # Load ensemble
        self.ensemble.load(os.path.join(model_dir, "ensemble.pkl"))
        
        self.is_trained = True
        logger.info(f"Loaded pipeline from {model_dir}")


def create_predictor(
    api_key: str = None,
    use_mock_llm: bool = False,
    ml_weight: float = 0.6,
    llm_weight: float = 0.4
) -> PersonalityPredictor:
    """
    Convenience function to create a personality predictor.
    
    Args:
        api_key: Gemini API key
        use_mock_llm: Whether to use mock LLM
        ml_weight: Weight for ML predictions
        llm_weight: Weight for LLM predictions
        
    Returns:
        Configured PersonalityPredictor
    """
    config = PipelineConfig(
        llm_api_key=api_key or os.getenv("GEMINI_API_KEY", ""),
        use_mock_llm=use_mock_llm,
        ml_weight=ml_weight,
        llm_weight=llm_weight
    )
    
    return PersonalityPredictor(config)


if __name__ == "__main__":
    # Test pipeline
    from .data_loader import load_data, DataConfig
    
    # Load data
    data_config = DataConfig(min_samples=1000)
    train_df, val_df, test_df, analysis = load_data(data_config)
    
    print(f"Loaded {len(train_df)} training samples")
    
    # Create predictor
    predictor = create_predictor(use_mock_llm=True)  # Use mock for testing
    
    # Train
    train_texts = train_df["text"].tolist()
    train_labels = {trait: train_df[trait].values for trait in OCEAN_TRAITS if trait in train_df.columns}
    
    val_texts = val_df["text"].tolist()
    val_labels = {trait: val_df[trait].values for trait in OCEAN_TRAITS if trait in val_df.columns}
    
    predictor.train(train_texts, train_labels, val_texts, val_labels)
    
    # Test prediction
    sample_text = """
    I love exploring new ideas and learning about different cultures. 
    Reading philosophy and discussing abstract concepts brings me joy.
    I'm quite organized in my work and always plan ahead.
    Meeting new people energizes me, and I enjoy social gatherings.
    I try to help others whenever I can and value kindness.
    Sometimes I worry about things, but I try to stay calm.
    """
    
    print("\n" + "=" * 60)
    print("SAMPLE ANALYSIS")
    print("=" * 60)
    
    prediction = predictor.predict(sample_text)
    print(prediction.summary())
    
    # Full analysis
    analysis = predictor.analyze(sample_text)
    print("\nFull Analysis (JSON):")
    print(json.dumps(analysis, indent=2))
