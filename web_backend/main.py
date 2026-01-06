"""
FastAPI Backend for Personality Detection System
================================================

This backend integrates with the existing ML pipeline in src/pipeline.py
and exposes REST API endpoints for personality prediction.
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global predictor instance
predictor: Optional[PersonalityPredictor] = None

# OCEAN traits
OCEAN_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


# ============================================================================
# Pydantic Models
# ============================================================================

class PredictRequest(BaseModel):
    """Request model for personality prediction."""
    text: str = Field(
        ...,
        min_length=50,
        description="Text to analyze for personality traits (minimum 50 characters)"
    )
    
    @validator('text')
    def validate_text_length(cls, v):
        if len(v.strip()) < 50:
            raise ValueError('Text must be at least 50 characters long for accurate analysis')
        return v.strip()


class TraitScore(BaseModel):
    """Individual trait score details."""
    score: float = Field(..., ge=0.0, le=1.0, description="Raw score (0-1)")
    percentile: float = Field(..., ge=0.0, le=100.0, description="Percentile ranking")
    category: str = Field(..., description="Category (Low/Medium/High)")


class PredictResponse(BaseModel):
    """Response model for personality prediction."""
    scores: Dict[str, float] = Field(..., description="Raw scores for each OCEAN trait")
    percentiles: Dict[str, float] = Field(..., description="Percentile rankings")
    categories: Dict[str, str] = Field(..., description="Categories (Low/Medium/High)")
    evidence: Dict[str, List[str]] = Field(default_factory=dict, description="Evidence sentences")
    traits: Dict[str, TraitScore] = Field(..., description="Combined trait information")
    text_length: int = Field(..., description="Length of analyzed text")
    model_info: Dict[str, Any] = Field(..., description="Model information")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str
    error_type: str


# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - load model on startup."""
    global predictor
    
    logger.info("Starting Personality Detection API...")
    logger.info("Loading ML model and training on synthetic data...")
    
    try:
        # Check for API key
        api_key = os.getenv("GEMINI_API_KEY")
        use_mock = not api_key or api_key == "your_api_key_here"
        
        if use_mock:
            logger.warning("GEMINI_API_KEY not found or invalid. Using mock LLM.")
        
        # Create predictor
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
        
        logger.info("Model loaded and ready!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        predictor = None
    
    yield
    
    # Cleanup
    logger.info("Shutting down Personality Detection API...")
    predictor = None


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
    """
    return HealthResponse(
        status="healthy" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        version="1.0.0"
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
async def predict_personality(request: PredictRequest):
    """
    Predict personality traits from text.
    
    Analyzes the provided text and returns Big Five (OCEAN) personality scores,
    percentiles, categories, and evidence sentences.
    
    **Minimum text length:** 50 characters
    
    **Returns:**
    - `scores`: Raw scores (0-1) for each trait
    - `percentiles`: Percentile rankings (0-100)
    - `categories`: Low/Medium/High classification
    - `evidence`: Supporting sentences from the text
    - `traits`: Combined trait information
    """
    global predictor
    
    # Check if model is loaded
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        # Get prediction
        result = predictor.predict(request.text)
        
        # Build combined traits dictionary
        traits = {}
        for trait in OCEAN_TRAITS:
            traits[trait] = TraitScore(
                score=round(result.scores.get(trait, 0.5), 4),
                percentile=round(result.percentiles.get(trait, 50.0), 2),
                category=result.categories.get(trait, "Medium")
            )
        
        # Build response
        response = PredictResponse(
            scores={k: round(v, 4) for k, v in result.scores.items()},
            percentiles={k: round(v, 2) for k, v in result.percentiles.items()},
            categories=result.categories,
            evidence=result.evidence if hasattr(result, 'evidence') and result.evidence else {},
            traits=traits,
            text_length=len(request.text),
            model_info={
                "model_type": "ensemble",
                "ml_model": "sentence-transformers/all-MiniLM-L6-v2",
                "calibrated": True
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
