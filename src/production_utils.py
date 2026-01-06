"""
Production Safety Utilities
===========================

ADDITIVE MODULE - Does NOT modify existing behavior.

This module provides production-safe helpers for the personality detection system:
1. Percentile clamping to avoid extreme 0.0/100.0 values
2. Confidence estimation based on text length and prediction variance
3. Text validation with graceful error handling
4. Reference distribution storage for stable percentile computation

These utilities are designed to be called from the web backend layer,
preserving the existing ML pipeline behavior for CLI scripts.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# OCEAN traits constant (matches existing definition in other modules)
OCEAN_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


# =============================================================================
# Configuration Constants
# =============================================================================

# Minimum/maximum percentile bounds to avoid extreme values
# WHY: 0.0 and 100.0 percentiles imply certainty we cannot guarantee
PERCENTILE_MIN = 1.0
PERCENTILE_MAX = 99.0

# Text length thresholds for validation and confidence
# WHY: Very short texts produce unreliable predictions
MIN_TEXT_LENGTH = 50  # Minimum characters required
OPTIMAL_TEXT_LENGTH = 200  # Length at which confidence is maximized
MAX_TEXT_LENGTH_FOR_CONFIDENCE = 2000  # Beyond this, no additional confidence boost

# Confidence calculation parameters
# WHY: These weights balance text length contribution vs prediction stability
TEXT_LENGTH_CONFIDENCE_WEIGHT = 0.3
PREDICTION_STABILITY_WEIGHT = 0.7


# =============================================================================
# Percentile Safety Functions
# =============================================================================

def clamp_percentile(percentile: float) -> float:
    """
    Clamp a percentile value to avoid extreme 0.0 or 100.0 values.
    
    WHY: Extreme percentiles (0.0/100.0) imply absolute certainty that our
    model cannot guarantee. Clamping to [1.0, 99.0] maintains statistical
    humility while preserving the meaningful range for users.
    
    Args:
        percentile: Raw percentile value (0-100)
        
    Returns:
        Clamped percentile value within [PERCENTILE_MIN, PERCENTILE_MAX]
    """
    return float(np.clip(percentile, PERCENTILE_MIN, PERCENTILE_MAX))


def clamp_percentiles(percentiles: Dict[str, float]) -> Dict[str, float]:
    """
    Clamp all percentiles in a dictionary to safe bounds.
    
    WHY: Batch operation for processing all OCEAN traits at once.
    Maintains backward compatibility by returning a new dict.
    
    Args:
        percentiles: Dictionary of trait -> percentile mappings
        
    Returns:
        New dictionary with all percentiles clamped to safe bounds
    """
    return {trait: clamp_percentile(pct) for trait, pct in percentiles.items()}


def compute_safe_percentile(
    score: float,
    reference_distribution: np.ndarray,
    trait: str = "unknown"
) -> float:
    """
    Compute percentile against a reference distribution with safety bounds.
    
    WHY: Percentiles should only be computed against stored training data
    distributions, not ad-hoc. This ensures consistency across predictions
    and prevents gaming/drift over time.
    
    Args:
        score: The raw score to compute percentile for
        reference_distribution: Sorted array of training scores for this trait
        trait: Trait name for logging (optional)
        
    Returns:
        Safe-bounded percentile value
    """
    from scipy.stats import percentileofscore
    
    if reference_distribution is None or len(reference_distribution) == 0:
        # WHY: If no reference data, return neutral 50th percentile
        # rather than failing or returning extreme values
        return 50.0
    
    raw_percentile = percentileofscore(reference_distribution, score, kind='mean')
    return clamp_percentile(raw_percentile)


# =============================================================================
# Reference Distribution Storage
# =============================================================================

@dataclass
class ReferenceDistributions:
    """
    Stores reference score distributions from training data.
    
    WHY: Percentiles must be computed against a stable reference population.
    Storing training distributions ensures consistent percentile computation
    regardless of the specific text being analyzed.
    """
    distributions: Dict[str, np.ndarray] = None
    
    def __post_init__(self):
        if self.distributions is None:
            self.distributions = {}
    
    def store_from_training(self, trait_scores: Dict[str, np.ndarray]):
        """
        Store sorted distributions from training data.
        
        WHY: Called once after training to capture reference population.
        
        Args:
            trait_scores: Dictionary mapping trait names to score arrays
        """
        for trait, scores in trait_scores.items():
            self.distributions[trait] = np.sort(scores)
    
    def get_percentile(self, trait: str, score: float) -> float:
        """
        Get safe-bounded percentile for a score against stored distribution.
        
        Args:
            trait: Trait name
            score: Score value
            
        Returns:
            Safe-bounded percentile
        """
        if trait not in self.distributions:
            return 50.0  # Neutral fallback
        
        return compute_safe_percentile(score, self.distributions[trait], trait)
    
    def get_all_percentiles(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Compute safe percentiles for all traits.
        
        Args:
            scores: Dictionary of trait -> score mappings
            
        Returns:
            Dictionary of trait -> percentile mappings (all clamped)
        """
        return {trait: self.get_percentile(trait, score) 
                for trait, score in scores.items()}


# =============================================================================
# Confidence Estimation
# =============================================================================

def estimate_text_length_confidence(text_length: int) -> float:
    """
    Estimate confidence component based on text length.
    
    WHY: Longer texts provide more linguistic signal for personality detection.
    Very short texts are inherently unreliable. This function provides a
    normalized [0,1] confidence factor based on text length.
    
    Args:
        text_length: Number of characters in the input text
        
    Returns:
        Confidence factor from 0.0 to 1.0
    """
    if text_length < MIN_TEXT_LENGTH:
        # WHY: Below minimum, confidence drops sharply
        return max(0.1, text_length / MIN_TEXT_LENGTH * 0.5)
    
    if text_length >= MAX_TEXT_LENGTH_FOR_CONFIDENCE:
        # WHY: Beyond a certain length, additional text doesn't help much
        return 1.0
    
    # WHY: Logarithmic scaling - diminishing returns for very long texts
    normalized = (text_length - MIN_TEXT_LENGTH) / (OPTIMAL_TEXT_LENGTH - MIN_TEXT_LENGTH)
    return float(np.clip(0.5 + 0.5 * np.tanh(normalized - 0.5), 0.5, 1.0))


def estimate_prediction_stability_confidence(
    ml_score: float,
    llm_score: float,
    ensemble_score: float
) -> float:
    """
    Estimate confidence based on agreement between ML and LLM predictions.
    
    WHY: When ML and LLM predictions agree, we have higher confidence.
    Large disagreements suggest the text may be ambiguous or out-of-distribution.
    
    Args:
        ml_score: ML model prediction (0-1)
        llm_score: LLM prediction (0-1)
        ensemble_score: Final ensemble score (0-1)
        
    Returns:
        Confidence factor from 0.0 to 1.0
    """
    # WHY: Measure disagreement as absolute difference
    disagreement = abs(ml_score - llm_score)
    
    # WHY: Convert disagreement to confidence using exponential decay
    # Small disagreements (<0.1) give high confidence
    # Large disagreements (>0.3) significantly reduce confidence
    stability = np.exp(-disagreement * 3)
    
    # WHY: Also penalize extreme scores slightly (they're often less reliable)
    extremity_penalty = 1.0 - 0.1 * (abs(ensemble_score - 0.5) * 2) ** 2
    
    return float(np.clip(stability * extremity_penalty, 0.3, 1.0))


def estimate_trait_confidence(
    trait: str,
    ml_score: float,
    llm_score: float,
    ensemble_score: float,
    text_length: int
) -> float:
    """
    Compute overall confidence for a single trait prediction.
    
    WHY: Combines text length and prediction stability factors into
    a single confidence score that users can interpret.
    
    Args:
        trait: Trait name (for potential trait-specific adjustments)
        ml_score: ML model prediction
        llm_score: LLM prediction
        ensemble_score: Final ensemble score
        text_length: Input text length in characters
        
    Returns:
        Overall confidence score from 0.0 to 1.0
    """
    text_conf = estimate_text_length_confidence(text_length)
    stability_conf = estimate_prediction_stability_confidence(
        ml_score, llm_score, ensemble_score
    )
    
    # WHY: Weighted combination - stability matters more than length
    # once minimum length is met
    confidence = (
        TEXT_LENGTH_CONFIDENCE_WEIGHT * text_conf +
        PREDICTION_STABILITY_WEIGHT * stability_conf
    )
    
    return round(float(np.clip(confidence, 0.1, 1.0)), 3)


def estimate_all_confidences(
    ml_scores: Dict[str, float],
    llm_scores: Dict[str, float],
    ensemble_scores: Dict[str, float],
    text_length: int
) -> Dict[str, float]:
    """
    Compute confidence scores for all traits.
    
    WHY: Batch operation for processing all OCEAN traits efficiently.
    
    Args:
        ml_scores: ML predictions per trait
        llm_scores: LLM predictions per trait
        ensemble_scores: Ensemble predictions per trait
        text_length: Input text length
        
    Returns:
        Dictionary of trait -> confidence mappings
    """
    confidences = {}
    for trait in OCEAN_TRAITS:
        if trait in ensemble_scores:
            ml = ml_scores.get(trait, ensemble_scores[trait])
            llm = llm_scores.get(trait, ensemble_scores[trait])
            ensemble = ensemble_scores[trait]
            confidences[trait] = estimate_trait_confidence(
                trait, ml, llm, ensemble, text_length
            )
    return confidences


# =============================================================================
# Text Validation
# =============================================================================

@dataclass
class TextValidationResult:
    """
    Result of text validation.
    
    WHY: Structured result allows graceful error handling with detailed
    feedback for users about why their input was rejected.
    """
    is_valid: bool
    cleaned_text: str
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    character_count: int = 0
    word_count: int = 0


def validate_text_for_prediction(
    text: str,
    min_length: int = MIN_TEXT_LENGTH,
    max_length: int = 10000
) -> TextValidationResult:
    """
    Validate and clean input text for personality prediction.
    
    WHY: Input validation is critical for production. This function:
    - Prevents empty/too-short inputs that produce unreliable results
    - Cleans whitespace to get accurate length measurements
    - Provides helpful error messages for users
    - Warns about suboptimal but acceptable inputs
    
    Args:
        text: Raw input text
        min_length: Minimum required character count
        max_length: Maximum allowed character count
        
    Returns:
        TextValidationResult with validation status and details
    """
    # WHY: Handle None input gracefully
    if text is None:
        return TextValidationResult(
            is_valid=False,
            cleaned_text="",
            error_message="Text input is required. Please provide text to analyze.",
            character_count=0,
            word_count=0
        )
    
    # WHY: Clean whitespace for accurate length measurement
    cleaned = text.strip()
    cleaned = ' '.join(cleaned.split())  # Normalize internal whitespace
    
    char_count = len(cleaned)
    word_count = len(cleaned.split()) if cleaned else 0
    
    # WHY: Check minimum length with helpful error message
    if char_count < min_length:
        return TextValidationResult(
            is_valid=False,
            cleaned_text=cleaned,
            error_message=f"Text is too short ({char_count} characters). "
                         f"Please provide at least {min_length} characters for accurate analysis.",
            character_count=char_count,
            word_count=word_count
        )
    
    # WHY: Check maximum length to prevent abuse/memory issues
    if char_count > max_length:
        return TextValidationResult(
            is_valid=False,
            cleaned_text=cleaned[:max_length],
            error_message=f"Text is too long ({char_count} characters). "
                         f"Maximum allowed is {max_length} characters.",
            character_count=char_count,
            word_count=word_count
        )
    
    # WHY: Warn about suboptimal (but valid) input lengths
    warning = None
    if char_count < OPTIMAL_TEXT_LENGTH:
        warning = (f"Text is relatively short ({char_count} characters). "
                  f"Longer texts (200+ characters) typically produce more accurate results.")
    
    return TextValidationResult(
        is_valid=True,
        cleaned_text=cleaned,
        warning_message=warning,
        character_count=char_count,
        word_count=word_count
    )


# =============================================================================
# Response Schema Helpers
# =============================================================================

def ensure_complete_response(
    scores: Dict[str, float],
    percentiles: Dict[str, float],
    categories: Dict[str, str],
    evidence: Optional[Dict[str, List[str]]] = None,
    confidences: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Ensure response contains all required fields with safe defaults.
    
    WHY: Production responses must have a consistent schema. This function
    fills in missing fields with safe defaults to prevent frontend crashes.
    
    Args:
        scores: Trait scores (may be incomplete)
        percentiles: Percentile values (may be incomplete)
        categories: Category labels (may be incomplete)
        evidence: Evidence sentences (may be None or incomplete)
        confidences: Confidence scores (may be None or incomplete)
        
    Returns:
        Complete response dictionary with all traits populated
    """
    complete = {
        "scores": {},
        "percentiles": {},
        "categories": {},
        "evidence": {},
        "confidences": {}
    }
    
    for trait in OCEAN_TRAITS:
        # WHY: Default score is 0.5 (neutral), clamped percentile is 50.0
        complete["scores"][trait] = round(scores.get(trait, 0.5), 4)
        complete["percentiles"][trait] = clamp_percentile(percentiles.get(trait, 50.0))
        complete["categories"][trait] = categories.get(trait, "Medium")
        
        # WHY: Empty list is safer than None for frontend iteration
        if evidence:
            complete["evidence"][trait] = evidence.get(trait, [])
        else:
            complete["evidence"][trait] = []
        
        # WHY: Default confidence of 0.5 indicates uncertainty
        if confidences:
            complete["confidences"][trait] = round(confidences.get(trait, 0.5), 3)
        else:
            complete["confidences"][trait] = 0.5
    
    return complete


def safe_get_evidence(evidence: Optional[Dict], trait: str, max_items: int = 5) -> List[str]:
    """
    Safely extract evidence for a trait with bounds checking.
    
    WHY: Evidence may be None, empty, or have unexpected structure.
    This helper provides safe extraction with consistent return type.
    
    Args:
        evidence: Evidence dictionary (may be None)
        trait: Trait name to extract
        max_items: Maximum number of evidence items to return
        
    Returns:
        List of evidence strings (may be empty, never None)
    """
    if evidence is None:
        return []
    
    trait_evidence = evidence.get(trait)
    
    if trait_evidence is None:
        return []
    
    if not isinstance(trait_evidence, list):
        # WHY: Handle unexpected types gracefully
        return [str(trait_evidence)] if trait_evidence else []
    
    # WHY: Limit items to prevent overly large responses
    return trait_evidence[:max_items]
