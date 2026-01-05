"""
Ensemble and Calibration Module
===============================

This module implements:
1. Weighted ensembling of ML and LLM predictions
2. Score calibration using isotonic regression
3. Percentile calculation
4. Category assignment (Low/Medium/High)

The ensemble approach combines the strengths of both methods:
- ML: Consistent, fast, trained on labeled data
- LLM: Contextually aware, extracts evidence
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import pickle

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import percentileofscore
from scipy.optimize import minimize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OCEAN traits
OCEAN_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

# Category labels
CATEGORY_LOW = "Low"
CATEGORY_MEDIUM = "Medium"
CATEGORY_HIGH = "High"


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model."""
    # Combination method
    method: str = "weighted_average"  # weighted_average, stacking, learned
    
    # Default weights
    ml_weight: float = 0.6
    llm_weight: float = 0.4
    
    # Calibration settings
    calibration_enabled: bool = True
    calibration_method: str = "isotonic"  # isotonic, platt
    
    # Category thresholds (percentiles)
    low_threshold: float = 33.0
    high_threshold: float = 67.0
    
    # Learn weights from validation data
    learn_weights: bool = True


@dataclass
class PersonalityPrediction:
    """
    Complete personality prediction result.
    
    Contains:
    - Raw and calibrated scores
    - Percentiles
    - Category labels
    - Evidence and justifications
    - Component predictions (ML and LLM)
    """
    # Core predictions
    scores: Dict[str, float]  # Calibrated scores (0-1)
    percentiles: Dict[str, float]  # Percentile ranks (0-100)
    categories: Dict[str, str]  # Category labels
    
    # Explainability
    evidence: Dict[str, List[str]]
    justifications: Dict[str, str]
    
    # Component predictions
    ml_scores: Dict[str, float] = field(default_factory=dict)
    llm_scores: Dict[str, float] = field(default_factory=dict)
    
    # Confidence
    confidence: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    text_length: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "scores": self.scores,
            "percentiles": self.percentiles,
            "categories": self.categories,
            "evidence": self.evidence,
            "justifications": self.justifications,
            "ml_scores": self.ml_scores,
            "llm_scores": self.llm_scores,
            "confidence": self.confidence,
            "text_length": self.text_length
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["=" * 50, "PERSONALITY ANALYSIS RESULTS", "=" * 50, ""]
        
        for trait in OCEAN_TRAITS:
            if trait in self.scores:
                score = self.scores[trait]
                percentile = self.percentiles.get(trait, 50)
                category = self.categories.get(trait, "Medium")
                
                lines.append(f"{trait.upper()}")
                lines.append(f"  Score: {score:.3f} | Percentile: {percentile:.1f} | Category: {category}")
                
                if trait in self.evidence and self.evidence[trait]:
                    lines.append(f"  Evidence:")
                    for ev in self.evidence[trait][:2]:
                        lines.append(f"    - \"{ev[:80]}...\"" if len(ev) > 80 else f"    - \"{ev}\"")
                
                if trait in self.justifications:
                    lines.append(f"  Justification: {self.justifications[trait]}")
                
                lines.append("")
        
        return "\n".join(lines)


class ScoreCalibrator:
    """
    Calibrates raw prediction scores to improve reliability.
    
    Uses isotonic regression or Platt scaling to map raw scores
    to better-calibrated probabilities.
    """
    
    def __init__(self, method: str = "isotonic"):
        """
        Initialize the calibrator.
        
        Args:
            method: Calibration method (isotonic, platt)
        """
        self.method = method
        self.calibrators: Dict[str, Any] = {}
        self.is_fitted = False
    
    def fit(self, predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]):
        """
        Fit calibration models.
        
        Args:
            predictions: Raw predictions per trait
            targets: True labels per trait
        """
        logger.info("Fitting score calibrators...")
        
        for trait in OCEAN_TRAITS:
            if trait in predictions and trait in targets:
                y_pred = predictions[trait]
                y_true = targets[trait]
                
                if self.method == "isotonic":
                    calibrator = IsotonicRegression(out_of_bounds="clip")
                    calibrator.fit(y_pred, y_true)
                else:
                    # Simple linear calibration as fallback
                    calibrator = LinearCalibrator()
                    calibrator.fit(y_pred, y_true)
                
                self.calibrators[trait] = calibrator
        
        self.is_fitted = True
    
    def calibrate(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Calibrate prediction scores.
        
        Args:
            scores: Raw scores per trait
            
        Returns:
            Calibrated scores
        """
        if not self.is_fitted:
            return scores
        
        calibrated = {}
        for trait, score in scores.items():
            if trait in self.calibrators:
                cal_score = self.calibrators[trait].predict([score])[0]
                calibrated[trait] = float(np.clip(cal_score, 0, 1))
            else:
                calibrated[trait] = score
        
        return calibrated
    
    def calibrate_batch(self, predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calibrate batch predictions.
        
        Args:
            predictions: Raw predictions per trait
            
        Returns:
            Calibrated predictions
        """
        if not self.is_fitted:
            return predictions
        
        calibrated = {}
        for trait, scores in predictions.items():
            if trait in self.calibrators:
                cal_scores = self.calibrators[trait].predict(scores)
                calibrated[trait] = np.clip(cal_scores, 0, 1)
            else:
                calibrated[trait] = scores
        
        return calibrated


class LinearCalibrator:
    """Simple linear calibration as fallback."""
    
    def __init__(self):
        self.slope = 1.0
        self.intercept = 0.0
    
    def fit(self, y_pred, y_true):
        """Fit linear calibration."""
        if len(y_pred) > 1:
            A = np.vstack([y_pred, np.ones(len(y_pred))]).T
            self.slope, self.intercept = np.linalg.lstsq(A, y_true, rcond=None)[0]
    
    def predict(self, y_pred):
        """Apply linear calibration."""
        return np.array(y_pred) * self.slope + self.intercept


class PercentileCalculator:
    """
    Calculates percentile ranks based on reference distribution.
    """
    
    def __init__(self):
        self.distributions: Dict[str, np.ndarray] = {}
    
    def fit(self, scores: Dict[str, np.ndarray]):
        """
        Fit percentile calculator with reference distributions.
        
        Args:
            scores: Reference score distributions per trait
        """
        for trait, values in scores.items():
            self.distributions[trait] = np.sort(values)
    
    def get_percentile(self, trait: str, score: float) -> float:
        """
        Calculate percentile for a score.
        
        Args:
            trait: Trait name
            score: Score value
            
        Returns:
            Percentile (0-100)
        """
        if trait not in self.distributions:
            return 50.0
        
        distribution = self.distributions[trait]
        percentile = percentileofscore(distribution, score, kind='mean')
        return float(percentile)
    
    def get_percentiles(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate percentiles for all traits."""
        return {trait: self.get_percentile(trait, score) 
                for trait, score in scores.items()}


class EnsembleModel:
    """
    Ensemble model combining ML and LLM predictions.
    
    Features:
    - Weighted averaging with learned or fixed weights
    - Score calibration
    - Percentile calculation
    - Category assignment
    """
    
    def __init__(self, config: EnsembleConfig = None):
        """
        Initialize the ensemble model.
        
        Args:
            config: Ensemble configuration
        """
        self.config = config or EnsembleConfig()
        self.weights: Dict[str, Dict[str, float]] = {}  # Per-trait weights
        self.calibrator = ScoreCalibrator(self.config.calibration_method)
        self.percentile_calculator = PercentileCalculator()
        self.is_fitted = False
        
        # Initialize default weights
        for trait in OCEAN_TRAITS:
            self.weights[trait] = {
                "ml": self.config.ml_weight,
                "llm": self.config.llm_weight
            }
    
    def _learn_weights(
        self,
        ml_predictions: Dict[str, np.ndarray],
        llm_predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray]
    ):
        """
        Learn optimal weights from validation data.
        
        Uses grid search to find weights that minimize MAE.
        """
        logger.info("Learning optimal ensemble weights...")
        
        for trait in OCEAN_TRAITS:
            if trait not in ml_predictions or trait not in targets:
                continue
            
            ml_pred = ml_predictions[trait]
            llm_pred = llm_predictions[trait]
            y_true = targets[trait]
            
            best_weight = 0.5
            best_mae = float('inf')
            
            # Grid search over weights
            for ml_w in np.arange(0.0, 1.05, 0.05):
                llm_w = 1.0 - ml_w
                combined = ml_w * ml_pred + llm_w * llm_pred
                mae = np.mean(np.abs(combined - y_true))
                
                if mae < best_mae:
                    best_mae = mae
                    best_weight = ml_w
            
            self.weights[trait] = {
                "ml": best_weight,
                "llm": 1.0 - best_weight
            }
            
            logger.info(f"  {trait}: ML={best_weight:.2f}, LLM={1-best_weight:.2f} (MAE={best_mae:.4f})")
    
    def fit(
        self,
        ml_predictions: Dict[str, np.ndarray],
        llm_predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray]
    ):
        """
        Fit the ensemble model.
        
        Args:
            ml_predictions: ML model predictions per trait
            llm_predictions: LLM predictions per trait
            targets: True labels per trait
        """
        logger.info("Fitting ensemble model...")
        
        # Learn weights if enabled
        if self.config.learn_weights:
            self._learn_weights(ml_predictions, llm_predictions, targets)
        
        # Combine predictions for calibration fitting
        combined = self.combine_predictions_batch(ml_predictions, llm_predictions)
        
        # Fit calibrator
        if self.config.calibration_enabled:
            self.calibrator.fit(combined, targets)
        
        # Fit percentile calculator
        self.percentile_calculator.fit(targets)
        
        self.is_fitted = True
    
    def combine_predictions(
        self,
        ml_scores: Dict[str, float],
        llm_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Combine ML and LLM predictions.
        
        Args:
            ml_scores: ML predictions
            llm_scores: LLM predictions
            
        Returns:
            Combined scores
        """
        combined = {}
        
        for trait in OCEAN_TRAITS:
            ml_score = ml_scores.get(trait, 0.5)
            llm_score = llm_scores.get(trait, 0.5)
            
            weights = self.weights.get(trait, {"ml": 0.5, "llm": 0.5})
            combined[trait] = weights["ml"] * ml_score + weights["llm"] * llm_score
        
        return combined
    
    def combine_predictions_batch(
        self,
        ml_predictions: Dict[str, np.ndarray],
        llm_predictions: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Combine batch predictions.
        
        Args:
            ml_predictions: ML predictions per trait
            llm_predictions: LLM predictions per trait
            
        Returns:
            Combined predictions
        """
        combined = {}
        
        for trait in OCEAN_TRAITS:
            if trait in ml_predictions and trait in llm_predictions:
                weights = self.weights.get(trait, {"ml": 0.5, "llm": 0.5})
                combined[trait] = (
                    weights["ml"] * ml_predictions[trait] + 
                    weights["llm"] * llm_predictions[trait]
                )
        
        return combined
    
    def get_category(self, percentile: float) -> str:
        """
        Assign category based on percentile.
        
        Args:
            percentile: Percentile value (0-100)
            
        Returns:
            Category label
        """
        if percentile < self.config.low_threshold:
            return CATEGORY_LOW
        elif percentile > self.config.high_threshold:
            return CATEGORY_HIGH
        else:
            return CATEGORY_MEDIUM
    
    def predict(
        self,
        text: str,
        ml_scores: Dict[str, float],
        llm_scores: Dict[str, float],
        evidence: Dict[str, List[str]] = None,
        justifications: Dict[str, str] = None
    ) -> PersonalityPrediction:
        """
        Generate complete personality prediction.
        
        Args:
            text: Input text
            ml_scores: ML model predictions
            llm_scores: LLM predictions
            evidence: Evidence sentences from LLM
            justifications: Justifications from LLM
            
        Returns:
            Complete PersonalityPrediction
        """
        # Combine predictions
        combined_scores = self.combine_predictions(ml_scores, llm_scores)
        
        # Calibrate
        if self.config.calibration_enabled and self.calibrator.is_fitted:
            calibrated_scores = self.calibrator.calibrate(combined_scores)
        else:
            calibrated_scores = combined_scores
        
        # Calculate percentiles
        percentiles = self.percentile_calculator.get_percentiles(calibrated_scores)
        
        # Assign categories
        categories = {trait: self.get_category(pct) 
                     for trait, pct in percentiles.items()}
        
        # Calculate confidence (based on agreement between ML and LLM)
        confidence = {}
        for trait in OCEAN_TRAITS:
            if trait in ml_scores and trait in llm_scores:
                diff = abs(ml_scores[trait] - llm_scores[trait])
                confidence[trait] = max(0.0, 1.0 - diff * 2)  # Higher agreement = higher confidence
        
        return PersonalityPrediction(
            scores=calibrated_scores,
            percentiles=percentiles,
            categories=categories,
            evidence=evidence or {},
            justifications=justifications or {},
            ml_scores=ml_scores,
            llm_scores=llm_scores,
            confidence=confidence,
            text_length=len(text)
        )
    
    def predict_batch(
        self,
        texts: List[str],
        ml_predictions: Dict[str, np.ndarray],
        llm_predictions: Dict[str, np.ndarray],
        evidences: List[Dict[str, List[str]]] = None,
        justifications: List[Dict[str, str]] = None
    ) -> List[PersonalityPrediction]:
        """
        Generate predictions for multiple texts.
        
        Args:
            texts: Input texts
            ml_predictions: ML predictions per trait
            llm_predictions: LLM predictions per trait
            evidences: Evidence sentences per sample
            justifications: Justifications per sample
            
        Returns:
            List of PersonalityPrediction objects
        """
        n_samples = len(texts)
        evidences = evidences or [{} for _ in range(n_samples)]
        justifications = justifications or [{} for _ in range(n_samples)]
        
        predictions = []
        for i in range(n_samples):
            ml_scores = {trait: ml_predictions[trait][i] 
                        for trait in ml_predictions}
            llm_scores = {trait: llm_predictions[trait][i] 
                         for trait in llm_predictions}
            
            pred = self.predict(
                texts[i],
                ml_scores,
                llm_scores,
                evidences[i],
                justifications[i]
            )
            predictions.append(pred)
        
        return predictions
    
    def save(self, filepath: str):
        """Save ensemble model to disk."""
        model_data = {
            "config": self.config,
            "weights": self.weights,
            "calibrator": self.calibrator,
            "percentile_calculator": self.percentile_calculator
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str):
        """Load ensemble model from disk."""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        
        self.config = model_data["config"]
        self.weights = model_data["weights"]
        self.calibrator = model_data["calibrator"]
        self.percentile_calculator = model_data["percentile_calculator"]
        self.is_fitted = True


def create_ensemble(
    config: EnsembleConfig = None,
    train_labels: Dict[str, np.ndarray] = None
) -> EnsembleModel:
    """
    Factory function to create ensemble model.
    
    Args:
        config: Ensemble configuration
        train_labels: Training labels for percentile fitting
        
    Returns:
        Configured EnsembleModel
    """
    ensemble = EnsembleModel(config)
    
    if train_labels:
        ensemble.percentile_calculator.fit(train_labels)
    
    return ensemble


if __name__ == "__main__":
    # Test ensemble
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 100
    ml_predictions = {trait: np.random.rand(n_samples) for trait in OCEAN_TRAITS}
    llm_predictions = {trait: np.random.rand(n_samples) for trait in OCEAN_TRAITS}
    targets = {trait: np.random.rand(n_samples) for trait in OCEAN_TRAITS}
    texts = [f"Sample text {i}" for i in range(n_samples)]
    
    # Create and fit ensemble
    config = EnsembleConfig(learn_weights=True, calibration_enabled=True)
    ensemble = EnsembleModel(config)
    ensemble.fit(ml_predictions, llm_predictions, targets)
    
    # Test single prediction
    ml_scores = {trait: 0.6 for trait in OCEAN_TRAITS}
    llm_scores = {trait: 0.7 for trait in OCEAN_TRAITS}
    evidence = {trait: ["Sample evidence"] for trait in OCEAN_TRAITS}
    justifications = {trait: "Sample justification" for trait in OCEAN_TRAITS}
    
    prediction = ensemble.predict(
        "Sample text for prediction",
        ml_scores,
        llm_scores,
        evidence,
        justifications
    )
    
    print(prediction.summary())
