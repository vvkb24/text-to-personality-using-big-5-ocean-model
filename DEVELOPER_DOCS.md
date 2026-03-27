# DEVELOPER DOCUMENTATION - Complete Reference

**Project**: Text-to-Personality Detection using Big Five (OCEAN) Model  
**Version**: 1.1.0 (Production Hardened)  
**Last Updated**: March 27, 2026  
**Status**: Production Ready

---

## Quick Navigation

| Document | Purpose |
|----------|---------|
| **README.md** | User-facing overview, quickstart guide |
| **ARCHITECTURE.md** | System design, component interactions, data flow |
| **IMPLEMENTATION_GUIDE.md** | Technical deep dives, code examples, testing |
| **This Document** | Integration overview, common tasks, troubleshooting |

---

## System Overview

### What This System Does

This system predicts **Big Five personality traits** (OCEAN model) from English text using a hybrid approach:

1. **ML Baseline**: Fast, consistent predictions using Sentence-BERT embeddings + Ridge regression
2. **LLM Engine**: Contextual analysis using Google Gemini API
3. **Ensemble**: Weighted combination with calibration for reliable probability estimates

**Output for each trait**:
- Score: 0-1 range (calibrated)
- Percentile: 1-99 range (clamped for safety)
- Category: Low/Medium/High
- Evidence: Supporting sentences from input text
- Confidence: Prediction confidence score (0-1)

### Architecture at a Glance

```
Text Input → [ML Baseline + LLM Engine] → Ensemble Combiner → Calibrated Output
                                           ↓
                          Score Normalization & Percentiles
                          ↓
                   Category Assignment & Confidence Estimation
                          ↓
                          JSON Response
```

---

## Core Components

### 1. **ML Baseline Model** (`src/ml_baseline.py`)

**What**: Fast, learnable model combining embeddings + regression

**Key Classes**:
- `TextEmbedder`: Sentence-BERT (all-MiniLM-L6-v2) → 384D vectors
- `TraitRegressor`: Ridge regression per OCEAN trait
- `MLBaselineModel`: Unified interface

**Training**: 5-fold cross-validation, saves checkpoints

**Inference**: ~50ms per text (embeddings) + ~1ms (5× predictions)

### 2. **LLM Inference Engine** (`src/llm_inference.py`)

**What**: Google Gemini-based contextual analysis

**Key Classes**:
- `LLMInferenceEngine`: Gemini API wrapper with rate limiting
- `MockLLMEngine`: Deterministic fallback for testing
- `TRAIT_DESCRIPTIONS`: Psychological prompts

**Features**:
- Retry logic: 3 attempts with exponential backoff
- Rate limiting: 15 requests/minute
- Evidence extraction: Supporting sentences from text
- Graceful degradation: Falls back to mock if API unavailable

**Latency**: 2-5 seconds per text (network dependent)

### 3. **Ensemble Combiner** (`src/ensemble.py`)

**What**: Fuses ML and LLM predictions with calibration

**Key Steps**:
1. **Weighted Average**: `0.6 × ML + 0.4 × LLM`
2. **Isotonic Calibration**: Maps raw scores to probabilities
3. **Percentile Calculation**: Relative ranking in training distribution
4. **Category Assignment**: Low/Medium/High based on thresholds
5. **Confidence Estimation**: Based on text length + prediction agreement

**Percentile Clamping** (v1.1.0): Avoids extreme 0.0/100.0 values → [1.0, 99.0]

### 4. **Production Utilities** (`src/production_utils.py`)

**What**: Safety guardrails for deployment

**Key Functions**:
- `clamp_percentiles`: Ensure [1, 99] bounds
- `estimate_all_confidences`: Per-trait confidence
- `validate_text_for_prediction`: Input validation
- `ensure_complete_response`: Response integrity checks

---

## Deployment Architecture

### Backend (FastAPI)

```
├─ Singleton Pattern: ML model loaded once at startup
├─ Request Validation: Pydantic schemas enforce contracts
├─ Response Validation: Safety checks before returning JSON
├─ CORS Middleware: Frontend access control
└─ Error Handling: Graceful degradation on failures
```

**Endpoints**:
- `GET /health`: Model status check
- `POST /predict`: Text → personality prediction
- `GET /docs`: Swagger UI

**Key Features**:
- Request timeout: 60 seconds
- Rate limiting: 15 API calls/minute (Gemini quota)
- Error responses: User-friendly messages

### Frontend (React)

```
├─ Vite: Build tool + dev server
├─ React 18: Component-based UI
├─ Tailwind CSS: Utility-first styling
├─ Recharts: Visualization library
├─ Axios: HTTP client with safety wrapper
└─ Safety Guards: Defensive rendering for API changes
```

**Components**:
- Text input field with validation
- Results visualization (radar charts, percentiles)
- Evidence display per trait
- Confidence indicators

---

## Common Development Tasks

### Running the System

**Option 1: Quick Demo (Fastest)**
```bash
python demo.py
```
- Generates synthetic training data
- Trains model in memory
- Tests on 5 examples + interactive mode
- Perfect for verification

**Option 2: Full Backend + Frontend**
```bash
# Terminal 1: Backend
python -m uvicorn web_backend.main:app --host 127.0.0.1 --port 8080

# Terminal 2: Frontend
cd web_frontend
npm install  # First time only
npm run dev
```
- Backend on http://localhost:8080
- Frontend on http://localhost:3000
- Full production-like setup

**Option 3: Training with Custom Data**
```bash
python train.py \
  --data-source huggingface \
  --dataset-name "Fatima0923/Automated-Personality-Prediction" \
  --use-llm \
  --learn-weights \
  --run-ablation
```
- Uses HuggingFace dataset
- Integrates Gemini API
- Learns optimal ensemble weights
- Generates ablation study report

### Making Code Changes

**1. Modifying Ensemble Weights**
```python
# In pipeline.py
config = PipelineConfig(
    ml_weight=0.7,      # Increased ML influence
    llm_weight=0.3      # Decreased LLM influence
)
```

**2. Changing Embedding Model**
```python
# In pipeline.py
config = PipelineConfig(
    ml_embedding_model="sentence-transformers/all-mpnet-base-v2"
)
```

**3. Adding New Trait**
```python
# In any module using OCEAN_TRAITS
OCEAN_TRAITS.append("my_new_trait")

# In llm_inference.py TRAIT_DESCRIPTIONS
TRAIT_DESCRIPTIONS["my_new_trait"] = {
    "name": "My New Trait",
    "description": "...",
    "high_characteristics": [...],
    "low_characteristics": [...]
}
```

**4. Adjusting Thresholds**
```python
# In pipeline.py
config = PipelineConfig(
    low_threshold=30.0,     # Changed from 33
    high_threshold=70.0     # Changed from 67
)
```

### Testing Changes

**Quick Validation**
```python
from src.pipeline import create_predictor

predictor = create_predictor(use_mock_llm=True)

# Synthetic training
predictor.train(
    ["I love new ideas"] * 100,
    {trait: np.random.uniform(0, 1, 100) for trait in OCEAN_TRAITS}
)

# Test prediction
prediction = predictor.predict("Your test text here...")
print(prediction.scores)
print(prediction.confidence)
```

**Full Integration Test**
```bash
pytest tests/ -v --cov=src/
```

---

## Troubleshooting Guide

### Issue: "No module named 'torch'"

**Cause**: PyTorch not installed  
**Solution**:
```bash
pip install torch torchvision torchaudio
```

### Issue: Gemini API fails with timeout

**Cause**: API rate limiting or network issues  
**Solution**:
```python
# Use mock LLM for testing
predictor = create_predictor(use_mock_llm=True)

# Or increase timeout
config.llm_timeout = 120
```

### Issue: Predictions are always "Medium"

**Cause**: Model not properly trained or ensemble weights wrong  
**Solution**:
```python
# Check if trained
print(predictor.is_trained)  # Should be True

# Check ensemble weights
print(predictor.ensemble.ml_weight)

# Verify reference distributions exist
print(len(predictor.ensemble.reference_distributions))
```

### Issue: Port 8080 or 3000 already in use

**Solution**:
```bash
# Use different port
python -m uvicorn web_backend.main:app --port 8081

# Or find and kill process
lsof -ti:8080 | xargs kill -9
```

### Issue: Text validation errors

**Cause**: Input text too short or too long  
**Solution**:
```python
# Minimum 50 characters
text = "I like exploring new ideas and concepts " * 3  # ~120 chars

# Automatically truncated at 2048 characters
```

---

## Performance Characteristics

### Latency Breakdown

| Phase | Time | Notes |
|-------|------|-------|
| Text embedding | ~50ms | SBERT forward pass |
| ML predictions | ~1ms | 5 Ridge models |
| LLM API call | 2-5s | Gemini network latency |
| Ensemble combination | <1ms | Arithmetic operations |
| Calibration + percentiles | <1ms | Lookups and arithmetic |
| **Total ML-only** | **~50ms** | Fast for batch processing |
| **Total with LLM** | **2-5.5s** | Dominated by API |

### Memory Footprint

| Component | Size |
|-----------|------|
| SBERT model | ~200MB |
| Ridge models (5×) | ~2MB |
| Isotonic regressors (5×) | <1MB |
| **Total** | **~202MB** |

### Throughput

- **ML-only**: ~20 requests/second (GPU), ~2 req/s (CPU)
- **With LLM**: ~0.3 requests/second (API rate-limited to 15/min)

---

## Configuration Reference

### Environment Variables

```bash
# Required for LLM
export GEMINI_API_KEY="your-api-key"

# Optional: Override settings
export PERSONALITY_MIN_TEXT_LENGTH=50
export PERSONALITY_MAX_TEXT_LENGTH=2048
```

### Runtime Configuration

```python
from src.pipeline import PipelineConfig, PersonalityPredictor

config = PipelineConfig(
    # ML settings
    ml_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    ml_regressor_type="ridge",  # Options: ridge, svr, mlp, linear
    
    # LLM settings
    llm_model="gemini-1.5-flash",
    llm_api_key=os.getenv("GEMINI_API_KEY"),
    use_mock_llm=False,  # Set True for testing
    
    # Ensemble settings
    ml_weight=0.6,
    llm_weight=0.4,
    calibration_enabled=True,
    
    # Thresholds
    low_threshold=33.0,
    high_threshold=67.0,
    
    # Text constraints
    min_text_length=50,
    max_text_length=2048
)

predictor = PersonalityPredictor(config)
```

---

## File Organization

```
src/
├── pipeline.py              # Main entry point (PersonalityPredictor)
├── ml_baseline.py           # ML model implementation
├── llm_inference.py         # Gemini API wrapper
├── ensemble.py              # Fusion and calibration
├── production_utils.py      # Safety guardrails (v1.1.0 new)
├── data_loader.py           # Dataset handling
├── evaluation.py            # Performance metrics
├── ablation.py              # Component analysis
└── utils.py                 # Helpers

web_backend/
├── main.py                  # FastAPI app
└── requirements.txt

web_frontend/
├── src/
│   ├── App.jsx              # Main React component
│   └── components/          # Reusable components
├── package.json
└── vite.config.js

config/
└── settings.yaml            # Global config

models/                       # Saved checkpoints
results/                      # Training outputs
```

---

## Best Practices

### ✅ Do

- Use mock LLM for testing (`use_mock_llm=True`)
- Validate input text length (50-2048 chars)
- Cache embeddings for batch operations
- Use GPU if available (auto-detected)
- Check `predictor.is_trained` before inference
- Monitor API rate limits (15 req/min Gemini quota)
- Use singleton pattern in production (model loads once)

### ❌ Don't

- Don't rely on percentiles 0.0 or 100.0 (clamped away)
- Don't make simultaneous API calls without rate limiting
- Don't assume LLM always succeeds (has fallback)
- Don't process text >2048 chars without truncation
- Don't create new predictor instance per request
- Don't ignore confidence scores (indicates reliability)
- Don't use real API key in test scripts

---

## Extending the System

### Adding New Regression Algorithm

1. **Edit `MLConfig`** in `src/ml_baseline.py`:
```python
@dataclass
class MLConfig:
    regressor_type: str = "ridge"  # Add "xgboost" option
```

2. **Edit `TraitRegressor._create_model()`**:
```python
elif self.config.regressor_type == "xgboost":
    from xgboost import XGBRegressor
    return XGBRegressor(...)
```

3. **Update dependencies**: Add to `requirements.txt`

### Adding New LLM Provider

1. **Create engine class** in `src/llm_inference.py`:
```python
class OpenAIInferenceEngine(LLMInferenceEngine):
    def predict_trait(self, text: str, trait: str) -> LLMPrediction:
        # OpenAI implementation
        pass
```

2. **Update `create_llm_engine()`** factory:
```python
if config.provider == "openai":
    return OpenAIInferenceEngine(config)
```

### Adding New Trait Category

1. **Update OCEAN_TRAITS** in all modules:
```python
OCEAN_TRAITS = [..., "my_new_trait"]
```

2. **Add TRAIT_DESCRIPTIONS** in `src/llm_inference.py`

3. **System auto-scales** (5 traits → 6 traits, etc.)

---

## Performance Optimization

### For ML-Only (No LLM)

```python
# Skip LLM inference entirely
prediction = predictor.predict_ml_only(text)  # ~50ms
```

### For Batch Processing

```python
# Batch embeddings for efficiency
texts = [text1, text2, ...]
predictions = predictor.predict_batch(texts)  # Much faster per text
```

### For GPU Acceleration

```python
# Auto-detected, but can force
config = PipelineConfig()
# Device automatically: CUDA > MPS > CPU
print(predictor.ml_model.embedder.device)  # Check which one
```

### For Testing

```python
# No API calls needed
predictor = create_predictor(use_mock_llm=True)  # Instant results
```

---

## Monitoring & Debugging

### Enable Detailed Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Now see detailed execution traces
predictor.predict("Your text...")
```

### Check Model State

```python
# Before/after training
print(f"Trained: {predictor.is_trained}")
print(f"ML fitted: {predictor.ml_model.is_fitted}")
print(f"Ensemble weights: {predictor.ensemble.ml_weight} + {predictor.ensemble.llm_weight}")

# Check reference distributions
for trait in OCEAN_TRAITS:
    dist = predictor.ensemble.reference_distributions[trait]
    print(f"{trait}: mean={dist.mean():.3f}, std={dist.std():.3f}")
```

### Validate Responses

```python
prediction = predictor.predict(text)

# Sanity checks
assert all(0 <= prediction.scores[t] <= 1 for t in OCEAN_TRAITS)
assert all(1 <= prediction.percentiles[t] <= 99 for t in OCEAN_TRAITS)
assert all(prediction.categories[t] in ["Low", "Medium", "High"] for t in OCEAN_TRAITS)
assert all(0 <= prediction.confidence[t] <= 1 for t in OCEAN_TRAITS)
```

---

## Deployment Checklist

- [ ] Set `GEMINI_API_KEY` environment variable
- [ ] Test with mock LLM first
- [ ] Run full integration tests
- [ ] Load test with expected traffic
- [ ] Configure CORS for frontend domain
- [ ] Set up logging and monitoring
- [ ] Configure alerts for API failures
- [ ] Plan for rate limiting (15 req/min)
- [ ] Test graceful degradation scenarios
- [ ] Document API endpoints for clients

---

## Key Decision Rationale

### Why Ensemble vs Single Model?

- **ML alone**: Fast & consistent, but narrow context
- **LLM alone**: Better reasoning, but expensive & variable
- **Ensemble**: Combines strengths, resilient to either failing

### Why 0.6 ML / 0.4 LLM Weights?

- ML is trained on labeled data (more grounded)
- LLM is variable but contextually aware
- 60/40 balance gives more weight to stable baseline

### Why Percentile Clamping [1, 99]?

- Avoids false certainty (0.0 and 100.0 are extreme)
- Maintains statistical humility
- Users less likely to misinterpret results

### Why Isotonic Calibration?

- Ridge predictions often uncalibrated
- Non-parametric (no sigmoid assumptions)
- Preserves monotonicity

### Why Singleton Pattern?

- Model loading takes ~2 seconds
- On 100 concurrent requests: 200s latency (bad!)
- Load once, use many times (fast!)

---

## Getting Help

### Resources

1. **Architecture Details**: See `ARCHITECTURE.md`
2. **Implementation Examples**: See `IMPLEMENTATION_GUIDE.md`
3. **Quick Start**: See `README.md`
4. **API Documentation**: Visit `/docs` when backend running

### Debugging Steps

1. Enable debug logging
2. Check `predictor.is_trained`
3. Verify input text length
4. Test with `use_mock_llm=True`
5. Check API rate limiting
6. Validate response schema
7. Review logs for errors

---

**End of Developer Documentation**

*For questions or contributions, open an issue on GitHub.*
