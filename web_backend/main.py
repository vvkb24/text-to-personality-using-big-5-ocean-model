"""
FastAPI Backend for Personality Detection System
================================================

This backend integrates with the existing ML pipeline in src/pipeline.py
and exposes REST API endpoints for personality prediction.

PRODUCTION HARDENING (Jan 6, 2026):
- Added production_utils for percentile clamping and confidence estimation
- ML pipeline loaded once at startup (singleton pattern)
- Strict request/response schema enforcement
- Text validation with graceful error handling
- Normalized endpoints (/predict and /predict/ both work)
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Add parent directory to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import create_predictor, PersonalityPredictor
from src.data_loader import PersonalityDataLoader, DataConfig
# WHY: Import production safety utilities for percentile clamping, confidence, and validation
from src.production_utils import (
    clamp_percentiles,
    estimate_all_confidences,
    validate_text_for_prediction,
    ensure_complete_response,
    safe_get_evidence,
    ReferenceDistributions,
    PERCENTILE_MIN,
    PERCENTILE_MAX,
    MIN_TEXT_LENGTH
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global predictor instance (singleton - loaded once at startup)
# WHY: Loading ML model on every request is expensive; singleton ensures single load
predictor: Optional[PersonalityPredictor] = None

# WHY: Store reference distributions for stable percentile computation across requests
reference_distributions: Optional[ReferenceDistributions] = None

# OCEAN traits
OCEAN_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


# ============================================================================
# Pydantic Models - STRICT SCHEMA ENFORCEMENT
# ============================================================================

class PredictRequest(BaseModel):
    """
    Request model for personality prediction.
    
    WHY: Strict schema ensures frontend and backend contracts match exactly.
    Validation happens at the API boundary before any processing.
    """
    text: str = Field(
        ...,
        min_length=MIN_TEXT_LENGTH,
        description=f"Text to analyze for personality traits (minimum {MIN_TEXT_LENGTH} characters)"
    )
    
    @validator('text')
    def validate_text_length(cls, v):
        """
        WHY: Double validation - Pydantic min_length + custom validator
        for better error messages and whitespace handling.
        """
        validation = validate_text_for_prediction(v)
        if not validation.is_valid:
            raise ValueError(validation.error_message)
        return validation.cleaned_text


class TraitScore(BaseModel):
    """
    Individual trait score details.
    
    WHY: Strict typing with bounds ensures frontend can rely on these ranges.
    """
    score: float = Field(..., ge=0.0, le=1.0, description="Raw score (0-1)")
    # WHY: Percentile bounds match PERCENTILE_MIN/MAX from production_utils
    percentile: float = Field(..., ge=PERCENTILE_MIN, le=PERCENTILE_MAX, 
                              description=f"Percentile ranking ({PERCENTILE_MIN}-{PERCENTILE_MAX})")
    category: str = Field(..., description="Category (Low/Medium/High)")
    # WHY: Added confidence field for per-trait confidence scores
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, 
                             description="Prediction confidence (0-1)")


class PredictResponse(BaseModel):
    """
    Response model for personality prediction.
    
    WHY: Complete schema with all fields ensures frontend never receives
    unexpected structure. All fields have defaults or are required.
    """
    scores: Dict[str, float] = Field(..., description="Raw scores for each OCEAN trait")
    percentiles: Dict[str, float] = Field(..., description="Percentile rankings (clamped to safe bounds)")
    categories: Dict[str, str] = Field(..., description="Categories (Low/Medium/High)")
    evidence: Dict[str, List[str]] = Field(default_factory=dict, description="Evidence sentences")
    traits: Dict[str, TraitScore] = Field(..., description="Combined trait information")
    # WHY: Added confidences field for per-trait confidence estimation
    confidences: Dict[str, float] = Field(default_factory=dict, description="Per-trait confidence scores")
    text_length: int = Field(..., description="Length of analyzed text")
    # WHY: Added warning field for suboptimal input feedback
    warning: Optional[str] = Field(default=None, description="Warning message for suboptimal input")
    model_info: Dict[str, Any] = Field(..., description="Model information")


class HealthResponse(BaseModel):
    """
    Health check response.
    
    WHY: Explicit health schema allows monitoring systems to parse response.
    """
    status: str
    model_loaded: bool
    version: str
    # WHY: Added details field for richer health diagnostics
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional health details")


class ErrorResponse(BaseModel):
    """
    Error response model.
    
    WHY: Consistent error schema helps frontend display meaningful messages.
    """
    detail: str
    error_type: str
    # WHY: Added field to indicate if error is recoverable
    recoverable: bool = Field(default=True, description="Whether the error can be resolved by user action")


# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager - load model ONCE at startup.
    
    WHY: Loading ML models is expensive (several seconds). Doing this once
    at startup ensures fast response times for all subsequent requests.
    The singleton pattern prevents memory bloat from multiple model copies.
    """
    global predictor, reference_distributions
    
    logger.info("Starting Personality Detection API...")
    logger.info("Loading ML model and training on synthetic data...")
    
    try:
        # Check for API key
        api_key = os.getenv("GEMINI_API_KEY")
        use_mock = not api_key or api_key == "your_api_key_here"
        
        if use_mock:
            logger.warning("GEMINI_API_KEY not found or invalid. Using mock LLM.")
        
        # WHY: Create predictor once - this instance serves all requests
        predictor = create_predictor(
            api_key=api_key if not use_mock else None,
            use_mock_llm=use_mock
        )
        
        # Train on synthetic data for demo
        logger.info("Generating synthetic training data...")
        loader = PersonalityDataLoader(DataConfig(min_samples=500))
        df = loader.create_synthetic_dataset(500)
        
        train_texts = df["text"].tolist()
        train_labels = {trait: df[trait].values for trait in OCEAN_TRAITS}
        
        logger.info("Training model...")
        predictor.train(train_texts, train_labels)
        
        # WHY: Store reference distributions for stable percentile computation
        # Percentiles should be computed against training data, not ad-hoc
        reference_distributions = ReferenceDistributions()
        reference_distributions.store_from_training(train_labels)
        logger.info("Stored reference distributions for percentile computation")
        
        logger.info("Model loaded and ready!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        predictor = None
        reference_distributions = None
    
    yield
    
    # Cleanup
    logger.info("Shutting down Personality Detection API...")
    predictor = None
    reference_distributions = None


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Personality Detection API",
    description="""
## Big Five (OCEAN) Personality Trait Detection

This API analyzes text to predict personality traits based on the Big Five model:

- **O**penness: Creativity, curiosity, openness to experience
- **C**onscientiousness: Organization, dependability, self-discipline  
- **E**xtraversion: Sociability, assertiveness, positive emotions
- **A**greeableness: Cooperation, trust, empathy
- **N**euroticism: Emotional instability, anxiety, moodiness

### Usage

Send a POST request to `/predict` with a JSON body containing the text to analyze.

### Disclaimer

Personality predictions are probabilistic estimates based on text analysis.
They are not clinical diagnoses and should not be used for medical purposes.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the API status and whether the model is loaded.
    WHY: Essential for container orchestration and load balancer health probes.
    """
    return HealthResponse(
        status="healthy" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        version="1.0.0",
        # WHY: Additional details help with debugging in production
        details={
            "reference_distributions_loaded": reference_distributions is not None,
            "traits_supported": OCEAN_TRAITS
        }
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    },
    tags=["Prediction"]
)
@app.post(
    "/predict/",
    response_model=PredictResponse,
    include_in_schema=False,  # WHY: Hide duplicate from docs but handle trailing slash
    tags=["Prediction"]
)
async def predict_personality(request: PredictRequest):
    """
    Predict personality traits from text.
    
    Analyzes the provided text and returns Big Five (OCEAN) personality scores,
    percentiles, categories, and evidence sentences.
    
    **Minimum text length:** 50 characters
    
    **Returns:**
    - `scores`: Raw scores (0-1) for each trait
    - `percentiles`: Percentile rankings (clamped to 1-99 to avoid extreme values)
    - `categories`: Low/Medium/High classification
    - `evidence`: Supporting sentences from the text
    - `confidences`: Per-trait confidence scores based on text length and prediction stability
    - `traits`: Combined trait information
    
    WHY: Dual endpoint registration handles both /predict and /predict/ 
    to prevent 307 redirects that can cause CORS issues.
    """
    global predictor, reference_distributions
    
    # WHY: Check model loading status before processing
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        # WHY: Re-validate text to get warning message (Pydantic already validated)
        validation = validate_text_for_prediction(request.text)
        warning_message = validation.warning_message
        
        # Get prediction from ML pipeline
        result = predictor.predict(request.text)
        
        # WHY: Clamp percentiles to avoid extreme 0.0/100.0 values
        # These imply certainty we cannot guarantee
        safe_percentiles = clamp_percentiles(result.percentiles)
        
        # WHY: Compute confidence scores based on text length and prediction agreement
        ml_scores = result.ml_scores if hasattr(result, 'ml_scores') else result.scores
        llm_scores = result.llm_scores if hasattr(result, 'llm_scores') else result.scores
        confidences = estimate_all_confidences(
            ml_scores=ml_scores,
            llm_scores=llm_scores,
            ensemble_scores=result.scores,
            text_length=len(request.text)
        )
        
        # Build combined traits dictionary with all required fields
        traits = {}
        for trait in OCEAN_TRAITS:
            traits[trait] = TraitScore(
                score=round(result.scores.get(trait, 0.5), 4),
                percentile=round(safe_percentiles.get(trait, 50.0), 2),
                category=result.categories.get(trait, "Medium"),
                # WHY: Include confidence in trait-level response
                confidence=round(confidences.get(trait, 0.5), 3)
            )
        
        # WHY: Safely extract evidence, handling None and missing traits
        safe_evidence = {}
        raw_evidence = result.evidence if hasattr(result, 'evidence') and result.evidence else {}
        for trait in OCEAN_TRAITS:
            safe_evidence[trait] = safe_get_evidence(raw_evidence, trait, max_items=5)
        
        # Build response with all fields populated
        response = PredictResponse(
            scores={k: round(v, 4) for k, v in result.scores.items()},
            percentiles={k: round(v, 2) for k, v in safe_percentiles.items()},
            categories=result.categories,
            evidence=safe_evidence,
            traits=traits,
            confidences=confidences,
            text_length=len(request.text),
            warning=warning_message,
            model_info={
                "model_type": "ensemble",
                "ml_model": "sentence-transformers/all-MiniLM-L6-v2",
                "calibrated": True,
                # WHY: Include percentile bounds in model info for frontend reference
                "percentile_bounds": {"min": PERCENTILE_MIN, "max": PERCENTILE_MAX}
            }
        )
        
        logger.info(f"Prediction completed for text of length {len(request.text)}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Personality Detection API",
        "version": "1.0.0",
        "description": "Big Five (OCEAN) personality trait detection from text",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
