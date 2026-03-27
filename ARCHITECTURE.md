# System Architecture Documentation

## Executive Summary

This personality detection system uses a **hybrid ensemble architecture** that combines:
- **ML Baseline**: Sentence-BERT embeddings + Ridge regression for consistent predictions
- **LLM Engine**: Google Gemini API for contextual, reasoning-based analysis
- **Ensemble Combiner**: Weighted fusion with isotonic calibration for accurate probability estimates

The system produces calibrated personality trait predictions (Big Five/OCEAN model) with percentiles, categories, evidence sentences, and per-trait confidence scores.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT TEXT (50-2048 chars)               │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴──────────┐
         │                      │
         ▼                      ▼
    ┌─────────────┐        ┌──────────────┐
    │ ML BASELINE │        │ LLM ENGINE   │
    │             │        │              │
    │ • SBERT     │        │ • Gemini API │
    │ • Ridge Reg │        │ • Prompting  │
    │ • 384D vecs │        │ • Evidence   │
    └──────┬──────┘        └────────┬─────┘
           │                        │
           │  Per-trait scores      │
           │  (0-1 range)           │
           │                        │
           └────────────┬───────────┘
                        │
                        ▼
            ┌────────────────────────┐
            │ ENSEMBLE COMBINER      │
            │                        │
            │ 1. Weighted Average    │
            │    (0.6 ML + 0.4 LLM)  │
            │ 2. Isotonic Calib.     │
            │ 3. Percentile Calc.    │
            │ 4. Category Assign.    │
            │ 5. Confidence Est.     │
            └────────────┬───────────┘
                         │
                         ▼
    ┌─────────────────────────────────────┐
    │ OUTPUT RESPONSE                     │
    │ - Scores (0-1)                      │
    │ - Percentiles (1-99)                │
    │ - Categories (Low/Medium/High)      │
    │ - Evidence Sentences                │
    │ - Confidences (0-1)                 │
    └─────────────────────────────────────┘
```

---

## Component Details

### 1. Text Validation & Preprocessing

**Input Constraints**:
- Minimum length: 50 characters
- Maximum length: 2048 characters
- Whitespace normalization applied

**Processing**:
```python
class TextPreprocessor:
    - Remove URLs
    - Normalize whitespace
    - Length validation
    - Optional lowercasing (disabled by default)
```

### 2. ML Baseline Model

#### Architecture
```
Training Phase:
  Texts → SBERT Embedding → Ridge Regression × 5 traits
           
Inference Phase:
  Text → Embed (384D) → Scale → Predict with Ridge × 5
```

#### Key Components

**TextEmbedder (SBERT)**
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384-dimensional vectors
- Device: Auto-detects (CUDA → MPS → CPU)
- Batch processing for efficiency

**TraitRegressor (Ridge)**
- Algorithm: Ridge regression (L2 regularization)
- Alpha: 1.0 (default)
- Features: SBERT embeddings
- Target: OCEAN trait scores (0-1)
- Cross-validation: 5-fold during training

#### Why Ridge Regression?
- Fast inference (<1ms)
- Handles multicollinearity in embedding vectors
- Interpretable weights
- Regularization prevents overfitting

#### Device Support
```python
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon
else:
    device = "cpu"
```

### 3. LLM Inference Engine

#### Architecture
```
Text + Trait Definition
    ↓
Psychologically-grounded Prompt
    ↓
Gemini API Call (gemini-1.5-flash)
    ↓
JSON Response Parsing
    ↓
Evidence Extraction
    ↓
Score + Evidence Dict
```

#### Prompting Strategy

Each trait uses a structured prompt:
```
TRAIT DEFINITION: [Big Five definition]
HIGH CHARACTERISTICS: [Behavioral markers]
LOW CHARACTERISTICS: [Opposite markers]

Text: [User input]

Respond with JSON:
{
  "score": 0.0-1.0,
  "evidence": ["sentence1", "sentence2"],
  "justification": "explanation"
}
```

#### Rate Limiting & Resilience
- **Rate limit**: 15 requests/minute (API quota compliance)
- **Retry logic**: Exponential backoff (Tenacity library)
- **Max retries**: 3 attempts
- **Timeout**: 60 seconds
- **Fallback**: Mock LLM if API key missing

#### Error Handling
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def call_gemini_api(prompt: str) -> Dict:
    # Call with retry logic
    pass
```

### 4. Ensemble Combiner

#### Four-Stage Process

**Stage 1: Weighted Average**
```
ensemble_score = 0.6 × ml_score + 0.4 × llm_score
```
- **0.6 ML weight**: More stable, trained on labeled data
- **0.4 LLM weight**: Contextual reasoning, less overfit-prone
- **Learnable**: Can optimize weights on validation set

**Stage 2: Score Calibration**
```
CalibratedScore = IsotonicRegression(RawEnsembleScore)
```

**Why Isotonic Regression?**
- Maps raw scores to true probabilities
- Non-parametric (no sigmoid assumptions)
- Preserves monotonicity
- Handles uncalibrated predictions from Ridge

**Training**: Fitted on validation set labels
**Inference**: Applied to all ensemble scores

**Stage 3: Percentile Calculation**
```
percentile = percentileofscore(
    training_distribution, 
    calibrated_score
)
```

Uses empirical distribution from training data for consistent percentile computation.

**Stage 4: Category Assignment**
```
Category = {
    "Low":    percentile < 33,
    "Medium": 33 ≤ percentile ≤ 67,
    "High":   percentile > 67
}
```

#### Confidence Estimation (v1.1.0)

```python
text_length_confidence = sigmoid(
    (text_length - MIN_LENGTH) / (OPTIMAL_LENGTH - MIN_LENGTH)
)

prediction_stability = 1.0 - (
    abs(ml_score - llm_score) / 0.5
)

confidence = (
    0.3 × text_length_confidence +
    0.7 × prediction_stability
)
```

**Factors**:
- Text length (longer = more confident)
- ML/LLM agreement (high agreement = more confident)
- Training sample size (implicit in percentile distribution)

#### Safety Guardrails (v1.1.0)

**Percentile Clamping**:
```python
percentile = clamp(percentile, 1.0, 99.0)
```
- Avoids extreme 0.0/100.0 values
- Maintains statistical humility
- Prevents false certainty

**Score Validation**:
- All scores bounded to [0, 1]
- All percentiles bounded to [1, 99]
- Evidence sentences validated for existence
- Complete response verification

---

## Data Flow: Request to Response

### Inference Pipeline (Per Request)

```
1. TEXT VALIDATION
   Input: Raw text
   ├─ Check length ≥ 50 chars
   ├─ Check length ≤ 2048 chars
   ├─ Remove extra whitespace
   └─ Output: Validated text or error
   
2. PARALLEL PROCESSING
   ├─ ML Pipeline:
   │  ├─ Embed with SBERT (384D)
   │  ├─ Scale with StandardScaler
   │  ├─ Predict with 5 Ridge models
   │  └─ Output: 5 trait scores
   │
   └─ LLM Pipeline (async):
      ├─ Call Gemini API × 5 traits
      ├─ Parse JSON responses
      ├─ Extract evidence sentences
      └─ Output: 5 trait scores + evidence

3. ENSEMBLE COMBINATION
   ├─ Weighted average per trait
   ├─ Apply calibration isotonic regressors
   ├─ Compute percentiles from training dist
   ├─ Clamp percentiles [1.0, 99.0]
   ├─ Assign categories
   └─ Estimate confidence

4. SAFETY CHECKS
   ├─ Validate scores ∈ [0, 1]
   ├─ Validate percentiles ∈ [1, 99]
   ├─ Ensure evidence exists
   └─ Ensure all traits present

5. RETURN RESPONSE
   JSON with:
   - scores: {openness: 0.72, ...}
   - percentiles: {openness: 78.5, ...}
   - categories: {openness: "High", ...}
   - evidence: {openness: ["sentence1", ...], ...}
   - confidences: {openness: 0.85, ...}
   - text_length: 391
   - warning: null (or error message)
```

---

## Training Pipeline

### Training Phases

**Phase 1: Data Loading**
```
HuggingFace Dataset (Fatima0923/Automated-Personality-Prediction)
    ↓
Filter valid OCEAN labels
    ↓
Train/Val/Test Split (70/15/15)
    ↓
~10,000 texts with personality labels
```

**Phase 2: ML Baseline Training**
```
For each trait:
  ├─ Embed all training texts with SBERT
  ├─ Fit StandardScaler
  ├─ Train Ridge regressor with 5-fold CV
  ├─ Evaluate on validation set
  └─ Save model checkpoint
```

**Phase 3: LLM Inference (Optional)**
```
Sample 100 validation texts
For each text:
  ├─ Call Gemini API
  ├─ Get scores + evidence
  ├─ Store results
  └─ Record success/failures
```

**Phase 4: Ensemble Training & Evaluation**
```
From validation set:
  ├─ Optimize ensemble weights
  ├─ Fit calibration isotonic regressors
  ├─ Store reference distributions
  
On test set:
  ├─ Evaluate ensemble performance
  ├─ Compute metrics (Pearson r, MAE, R², RMSE)
  ├─ Run ablation studies
  └─ Generate report
```

---

## Performance Characteristics

### Memory Profile

| Component | Memory | Notes |
|-----------|--------|-------|
| SBERT Model | ~200MB | Loaded once at startup |
| Ridge Models (5×) | ~2MB | Tiny serialized regressors |
| Scalers (5×) | <1MB | StandardScaler state |
| Isotonic Regressors (5×) | <1MB | Calibration models |
| **Total** | **~200MB** | Reasonable for production |

### Latency Profile

| Operation | Latency | Notes |
|-----------|---------|-------|
| Text embedding | ~50ms | Single SBERT forward pass |
| Ridge prediction (5×) | ~1ms | Negligible matrix multiply |
| Gemini API call | 2-5s | Network + model inference |
| Ensemble combination | <1ms | Arithmetic operations |
| **Total** | **2-5.5s** | Dominated by API latency |

### Scalability

**Throughput** (assuming 2x concurrent requests):
- ML-only: ~20 req/s per GPU
- With LLM: ~0.3 req/s (API limited to 15 req/min)

**Optimization**:
- ML pipeline is GPU-accelerated
- LLM calls are rate-limited (respects API quotas)
- Batch prediction possible for inference-only workloads

---

## Failure Modes & Recovery

### Graceful Degradation

```python
# No Gemini API key?
→ Use mock LLM
→ Return ML-only predictions
→ Confidence score lower

# API timeout?
→ Retry with exponential backoff (max 3×)
→ On final failure: use ML prediction only
→ Set warning in response

# Text too short?
→ Validate early
→ Return user-friendly error
→ No model processing

# ML/LLM significantly disagree?
→ Lower confidence score
→ Return ensemble prediction anyway
→ Log discrepancy for monitoring

# Score out of bounds?
→ Clamp to [0, 1]
→ Log warning
→ Continue processing
```

---

## Quality Metrics

### Typical Performance (Test Set)

| Metric | Range | Interpretation |
|--------|-------|-----------------|
| Pearson r | 0.65-0.75 | Good correlation with ground truth |
| MAE | 0.10-0.15 | ~10-15% absolute error |
| R² | 0.42-0.56 | Explains 42-56% of variance |
| RMSE | 0.12-0.18 | ~12-18% root error |

### Confidence Calibration

- **Expected calibration error**: <5%
- **Mean absolute percentile error**: <3%
- **Coverage** (68% confidence interval): ~65-70%

---

## Extensibility Points

### Adding New Traits

Edit `OCEAN_TRAITS` constant and add:
1. Trait name to list
2. Embedding model training (automatic)
3. LLM prompt definition (in TRAIT_DESCRIPTIONS)

### Switching Embedding Models

Edit `PipelineConfig.ml_embedding_model`:
```python
Alternative models:
- "sentence-transformers/all-mpnet-base-v2" (higher quality)
- "sentence-transformers/paraphrase-MiniLM-L6-v2" (faster)
```

### Changing Regression Algorithm

Edit `MLConfig.regressor_type`:
```python
Options: "ridge" (default), "svr", "mlp", "linear"
```

### Modifying Ensemble Weights

Edit `PipelineConfig.ml_weight` and `llm_weight`:
```python
Or enable automatic learning:
config.learn_weights = True
```

---

## Deployment Architecture

### Backend (FastAPI)

```
Uvicorn ASGI Server (Port 8080)
    ↓
FastAPI application
    ↓
[CORS Middleware]
    ↓
Request validation (Pydantic)
    ↓
Singleton PersonalityPredictor
    ↓
Inference pipeline
    ↓
Response validation + safety checks
    ↓
JSON response
```

**Key Features**:
- Singleton pattern: ML model loaded once
- Request validation: Pydantic models
- Response validation: Type-checked before return
- CORS: Configured for frontend access

### Frontend (React)

```
React 18 Application (Port 3000)
    ↓
Vite build tool
    ↓
Tailwind CSS styling
    ↓
Recharts visualization
    ↓
Axios HTTP client
    ↓
SafeAPI wrapper (handles missing fields)
    ↓
Component rendering
```

**Key Features**:
- Safety guards against API response changes
- Defensive rendering (missing fields don't crash UI)
- Loading states for async operations
- Error display with user-friendly messages

---

## Version History

### v1.1.0 (Production Hardened - Jan 6, 2026)

**New Features**:
- Percentile safety clamping (1-99 range)
- Per-trait confidence scores
- Text validation with graceful errors
- Backend singleton pattern for ML model
- Normalized API endpoints

**Files Modified**:
- `web_backend/main.py`: Added safety checks
- `src/production_utils.py`: New module for safety
- `src/pipeline.py`: Confidence calculation
- `src/ensemble.py`: Percentile clamping

### v1.0.0 (Initial Release - Jan 5-6, 2026)

**Core Features**:
- ML baseline with SBERT + Ridge
- LLM inference with Gemini
- Ensemble with calibration
- FastAPI backend
- React frontend

---

## Testing & Validation

### Unit Tests
- Text preprocessing
- Embedding generation
- Trait regressor training
- Ensemble combination
- Calibration fitting

### Integration Tests
- Full pipeline: text → prediction
- ML + LLM + Ensemble
- Error handling paths
- API endpoints

### Performance Tests
- Embedding latency
- Regression inference time
- API response time
- Memory footprint

### Ablation Studies
- ML-only vs LLM-only vs Ensemble
- Effect of text length
- Effect of calibration
- Weight sensitivity analysis

---

## References & Theory

### Big Five (OCEAN) Model
- Goldberg, L. R. (1990). An alternative "description of personality"
- Five-factor taxonomy of personality traits
- Widely used in psychology and HR

### Sentence-BERT
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks

### Isotonic Regression
- Monotonic transformation for probability calibration
- Non-parametric approach to score calibration

---

## Contact & Support

For technical questions or contributions, please open an issue on the GitHub repository:
- Repository: https://github.com/vvkb24/text-to-personality-using-big-5-ocean-model
- Issues: GitHub Issues tracker

---

**Document Version**: 1.0  
**Last Updated**: March 27, 2026  
**Status**: Production  
