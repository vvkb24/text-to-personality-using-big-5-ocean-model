# Technical Implementation Guide

## Getting Started for Developers

This guide covers implementation details, key decisions, and how to extend the system.

---

## Project Structure Deep Dive

```
personality-detection/
│
├── src/                                  # Core ML pipeline
│   ├── __init__.py
│   ├── pipeline.py                       # Main inference interface
│   │   └── PersonalityPredictor: Unified entry point
│   ├── data_loader.py                    # Data handling
│   │   ├── DataConfig: Configuration
│   │   ├── TextPreprocessor: Text cleaning
│   │   └── PersonalityDataLoader: Dataset loading
│   ├── ml_baseline.py                    # ML model
│   │   ├── TextEmbedder: SBERT embeddings
│   │   ├── TraitRegressor: Ridge regression per trait
│   │   └── MLBaselineModel: Unified ML interface
│   ├── llm_inference.py                  # LLM engine
│   │   ├── LLMInferenceEngine: Gemini API wrapper
│   │   ├── MockLLMEngine: Testing fallback
│   │   └── TRAIT_DESCRIPTIONS: Psychological prompts
│   ├── ensemble.py                       # Fusion & calibration
│   │   ├── EnsembleModel: Weighted combination
│   │   ├── PersonalityPrediction: Output container
│   │   └── Calibration (isotonic regression)
│   ├── production_utils.py               # Safety guardrails (v1.1.0)
│   │   ├── clamp_percentiles: Avoid 0/100
│   │   ├── estimate_all_confidences: Per-trait confidence
│   │   ├── validate_text_for_prediction: Input validation
│   │   └── ReferenceDistributions: Stable percentiles
│   ├── evaluation.py                     # Metrics & reporting
│   │   └── PersonalityEvaluator: Performance metrics
│   ├── ablation.py                       # Ablation studies
│   │   └── AblationStudies: Component analysis
│   └── utils.py                          # Helper functions
│
├── web_backend/                          # FastAPI REST API
│   ├── main.py                           # API endpoints
│   │   ├── Singleton predictor pattern
│   │   ├── Request/response validation
│   │   └── Production safety checks
│   └── requirements.txt                  # Backend dependencies
│
├── web_frontend/                         # React UI
│   ├── src/
│   │   ├── App.jsx                       # Main component
│   │   └── components/                   # React components
│   ├── package.json                      # NPM dependencies
│   ├── vite.config.js                    # Build config
│   └── tailwind.config.js                # CSS config
│
├── config/
│   └── settings.yaml                     # Global configuration
│
├── models/                               # Saved model checkpoints
│   ├── ml_embedder.pkl
│   ├── trait_regressors/
│   ├── isotonic_calibrators/
│   └── reference_distributions.pkl
│
├── results/                              # Training outputs
│   ├── metrics.json
│   ├── predictions.csv
│   ├── ablation_report.html
│   └── plots/
│
├── train.py                              # Training entry point
├── demo.py                               # Quick demo
├── example_inference.py                  # API examples
├── analyze_essay.py                      # Custom text analysis
└── requirements.txt                      # Python dependencies
```

---

## Code Flow: From Input to Output

### 1. Creating a Predictor

```python
from src.pipeline import create_predictor

# Option A: With real Gemini API
predictor = create_predictor(
    api_key="sk-...",           # Gemini API key
    use_mock_llm=False,         # Use real API
    ml_weight=0.6,              # ML/LLM fusion weights
    llm_weight=0.4
)

# Option B: For testing (mock LLM)
predictor = create_predictor(
    api_key="",                 # Empty key
    use_mock_llm=True,          # Use deterministic mock
)
```

**What happens internally**:
```python
class PersonalityPredictor:
    def __init__(self, config):
        # Lazy initialization (components created on first use)
        self._ml_model = None           # TextEmbedder + TraitRegressors
        self._llm_engine = None         # Gemini API wrapper
        self._ensemble = None           # EnsembleModel
        self._preprocessor = None       # TextPreprocessor
        
        self.is_trained = False
        self.reference_distributions = None
```

### 2. Training Phase

```python
train_texts = [...]  # List of text strings
train_labels = {     # Dict: trait -> numpy array of scores
    "openness": np.array([0.7, 0.6, 0.8, ...]),
    "conscientiousness": np.array([...]),
    ...
}

predictor.train(train_texts, train_labels)
```

**Training pipeline**:
```
1. Initialize TextPreprocessor
   └─ Preprocess all texts

2. Create TextEmbedder
   └─ Load SBERT model (download if needed)

3. For each trait:
   ├─ Embed all training texts
   ├─ Fit StandardScaler
   ├─ Train Ridge regressor with 5-fold CV
   └─ Save checkpoint

4. Create EnsembleModel
   ├─ Learn optimal weights on validation set
   ├─ Fit isotonic regressors for calibration
   └─ Store reference distributions

5. Save all components to models/ directory
```

### 3. Inference Phase (Single Text)

```python
text = "I love exploring new ideas and meeting different people..."
prediction = predictor.predict(text)

print(prediction.scores)        # {"openness": 0.72, ...}
print(prediction.percentiles)   # {"openness": 78.5, ...}
print(prediction.categories)    # {"openness": "High", ...}
print(prediction.confidence)    # {"openness": 0.85, ...}
```

**Inference pipeline (detailed)**:

```
INPUT: text (raw string)
    ↓
TEXT VALIDATION
├─ Check: 50 ≤ len ≤ 2048
├─ Clean: strip whitespace
├─ Validate: non-empty, valid characters
└─ Output: validated_text or raise ValueError

PARALLEL PROCESSING
├─ ML Pipeline:
│  ├─ embed = SBERT.encode(text)           # 384D vector
│  ├─ for trait in OCEAN_TRAITS:
│  │  ├─ X_scaled = scaler[trait].transform([embed])
│  │  └─ ml_score[trait] = Ridge[trait].predict(X_scaled)[0]
│  └─ Output: Dict[trait → float]
│
└─ LLM Pipeline (async):
   ├─ for trait in OCEAN_TRAITS:
   │  ├─ prompt = build_prompt(text, trait)
   │  ├─ response = call_gemini_api(prompt)   # Retry logic
   │  ├─ parsed = parse_json_response(response)
   │  ├─ llm_score[trait] = parsed["score"]
   │  └─ evidence[trait] = parsed["evidence"]
   └─ Output: Dict[trait → score], Dict[trait → evidence]

ENSEMBLE COMBINATION
├─ for trait in OCEAN_TRAITS:
│  ├─ raw_score = 0.6 × ml_score + 0.4 × llm_score
│  ├─ cal_score = isotonic_reg[trait].transform([raw_score])
│  ├─ percentile = percentileofscore(train_dist, cal_score)
│  ├─ percentile = clamp(percentile, 1.0, 99.0)
│  ├─ category = categorize(percentile)
│  ├─ confidence = estimate_confidence(text, ml, llm)
│  └─ Store all values
│
└─ Output: PersonalityPrediction object

VALIDATION & SAFETY
├─ Assert all scores ∈ [0, 1]
├─ Assert all percentiles ∈ [1, 99]
├─ Assert evidence not empty
├─ Assert all traits present
└─ Raise exception if any check fails

RETURN
└─ PersonalityPrediction with scores, percentiles, categories, evidence, confidence
```

### 4. Backend API Endpoint

```python
# FastAPI endpoint
@app.post("/predict")
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Receives text, calls singleton predictor, returns response.
    """
    # 1. Pydantic validation (automatic)
    text = request.text  # Already validated by Pydantic
    
    # 2. Call predictor
    prediction = predictor.predict(text)
    
    # 3. Apply production safety checks
    safe_prediction = ensure_complete_response(prediction)
    safe_prediction.percentiles = clamp_percentiles(safe_prediction.percentiles)
    safe_prediction.confidences = estimate_all_confidences(...)
    
    # 4. Convert to response
    response = build_response(safe_prediction)
    
    # 5. Return JSON
    return response
```

---

## Key Classes & Interfaces

### TextEmbedder

```python
class TextEmbedder:
    """Sentence transformer embedding layer."""
    
    def __init__(self, model_name: str, device: str = "auto"):
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = 384  # For MiniLM-L6-v2
    
    def embed(self, texts: List[str], batch_size=32) -> np.ndarray:
        """Embed multiple texts efficiently."""
        # Batch processing for memory efficiency
        # Returns: (n_texts, 384)
        pass
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed one text."""
        # Returns: (384,)
        pass
```

### TraitRegressor

```python
class TraitRegressor:
    """Regression model for a single OCEAN trait."""
    
    def __init__(self, trait_name: str, config: MLConfig):
        self.trait_name = trait_name
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=config.ridge_alpha)
        
    def fit(self, X: np.ndarray, y: np.ndarray, cross_validate=True):
        """
        Train on embeddings X and labels y.
        X shape: (n_samples, 384)
        y shape: (n_samples,)
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Cross-validate
        if cross_validate:
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
            self.cv_scores = cv_scores
        
        # Train on full training set
        self.model.fit(X_scaled, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on embeddings."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
```

### EnsembleModel

```python
class EnsembleModel:
    """Combines ML and LLM predictions with calibration."""
    
    def __init__(self, config: EnsembleConfig):
        self.ml_weight = config.ml_weight
        self.llm_weight = config.llm_weight
        self.isotonic_regressors = {}  # Per-trait calibrators
        self.reference_distributions = {}  # Training distribution
        
    def combine(self, 
                ml_scores: Dict[str, float],
                llm_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Weighted combination with calibration.
        
        For each trait:
          1. raw = 0.6 × ml + 0.4 × llm
          2. calibrated = isotonic_reg.transform(raw)
          3. Return calibrated score
        """
        ensemble_scores = {}
        for trait in OCEAN_TRAITS:
            raw = (self.ml_weight * ml_scores[trait] + 
                   self.llm_weight * llm_scores[trait])
            calibrated = self.isotonic_regressors[trait].transform([raw])[0]
            ensemble_scores[trait] = calibrated
        return ensemble_scores
    
    def calculate_percentiles(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Convert scores to percentiles based on training distribution."""
        percentiles = {}
        for trait in OCEAN_TRAITS:
            score = scores[trait]
            dist = self.reference_distributions[trait]
            pct = percentileofscore(dist, score)
            percentiles[trait] = clamp_percentile(pct)  # [1, 99]
        return percentiles
```

### LLMInferenceEngine

```python
class LLMInferenceEngine:
    """Google Gemini API wrapper with rate limiting."""
    
    def __init__(self, config: LLMConfig):
        self.api_key = config.api_key
        self.model = config.model  # "gemini-1.5-flash"
        self.rate_limiter = RateLimiter(15, 60)  # 15 req/min
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential())
    def predict_trait(self, text: str, trait: str) -> LLMPrediction:
        """
        Predict one trait using LLM.
        
        Retry logic: Exponential backoff on failure
        """
        # 1. Rate limit
        self.rate_limiter.wait_if_necessary()
        
        # 2. Build prompt
        prompt = self._build_prompt(text, trait)
        
        # 3. Call API
        response = genai.GenerativeModel(self.model).generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 2048
            }
        )
        
        # 4. Parse response
        json_str = response.text
        parsed = json.loads(json_str)
        
        # 5. Extract results
        return LLMPrediction(
            scores={trait: parsed["score"]},
            evidence={trait: parsed["evidence"]},
            justifications={trait: parsed["justification"]}
        )
    
    def _build_prompt(self, text: str, trait: str) -> str:
        """Build trait-specific prompt."""
        desc = TRAIT_DESCRIPTIONS[trait]
        return f"""
Analyze the following text for {desc['name']}:

Definition: {desc['description']}
High indicators: {', '.join(desc['high_characteristics'])}
Low indicators: {', '.join(desc['low_characteristics'])}

Text: {text}

Respond with ONLY a JSON object (no markdown):
{{
  "score": 0.0 to 1.0,
  "evidence": ["sentence from text supporting score"],
  "justification": "Explanation for score"
}}
"""
```

---

## Configuration Management

### settings.yaml

All configuration centralized:
```yaml
data:
  huggingface_dataset: "Fatima0923/..."
  min_samples: 10000
  train_ratio: 0.7
  
ml_model:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  regressor_type: "ridge"
  cv_folds: 5
  
llm:
  provider: "gemini"
  model: "gemini-1.5-flash"
  requests_per_minute: 15
  
ensemble:
  method: "weighted_average"
  ml_weight: 0.6
  llm_weight: 0.4
  calibration_enabled: true
```

### PipelineConfig (Runtime)

```python
config = PipelineConfig(
    ml_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    ml_regressor_type="ridge",
    llm_model="gemini-1.5-flash",
    llm_api_key=os.getenv("GEMINI_API_KEY"),
    use_mock_llm=False,
    ml_weight=0.6,
    llm_weight=0.4,
    low_threshold=33.0,
    high_threshold=67.0,
    min_text_length=50,
    max_text_length=2048
)

predictor = PersonalityPredictor(config)
```

---

## Error Handling Patterns

### Input Validation

```python
def validate_text_for_prediction(text: str) -> ValidationResult:
    """Comprehensive text validation."""
    
    # Check for None
    if text is None:
        return ValidationResult(
            is_valid=False,
            error_message="Text cannot be empty"
        )
    
    # Strip whitespace
    cleaned = text.strip()
    
    # Check length
    if len(cleaned) < MIN_TEXT_LENGTH:
        return ValidationResult(
            is_valid=False,
            error_message=f"Text too short. Minimum {MIN_TEXT_LENGTH} characters."
        )
    
    if len(cleaned) > MAX_TEXT_LENGTH:
        cleaned = cleaned[:MAX_TEXT_LENGTH]
        # Warn but continue
    
    return ValidationResult(is_valid=True, cleaned_text=cleaned)
```

### API Response Fallback

```python
try:
    prediction = predictor.predict(text)
except ValueError as e:
    # Input validation failed
    return {"error": str(e), "status": "validation_error"}
except TimeoutError:
    # LLM API timeout - use ML only
    prediction_ml = predictor.predict_ml_only(text)
    return {
        "warning": "LLM service unavailable, using ML model only",
        **prediction_ml.to_dict()
    }
except Exception as e:
    # Unexpected error
    logger.error(f"Prediction failed: {e}")
    return {"error": "Internal server error", "status": 500}
```

---

## Performance Optimization Tips

### 1. Batch Inference

```python
# Instead of:
for text in texts:
    pred = predictor.predict(text)  # Slow

# Do:
predictions = predictor.predict_batch(texts)  # Batches embeddings
```

### 2. GPU Acceleration

```python
# Automatically uses CUDA if available
config = PipelineConfig()
predictor = PersonalityPredictor(config)

# Check device used
print(predictor.ml_model.embedder.device)  # cuda / cpu / mps
```

### 3. Caching Embeddings

```python
# For repeated predictions on same text
embedder = predictor.ml_model.embedder
embedding_cache = {}

text_hash = hash(text)
if text_hash not in embedding_cache:
    embedding_cache[text_hash] = embedder.embed_single(text)

embedding = embedding_cache[text_hash]
# Use embedding for all traits
```

### 4. Mock LLM for Testing

```python
# Speeds up testing (no API calls)
predictor = create_predictor(
    use_mock_llm=True  # Deterministic, instant
)
```

---

## Adding New Features

### Example: Adding Percentile Tracking

```python
# In ensemble.py
@dataclass
class PersonalityPrediction:
    scores: Dict[str, float]
    percentiles: Dict[str, float]
    # NEW: Track percentile changes over time
    percentile_history: List[Dict[str, float]] = field(default_factory=list)
    
    def add_to_history(self):
        self.percentile_history.append(self.percentiles.copy())
```

### Example: Adding Custom Trait

```python
# In OCEAN_TRAITS (any module)
OCEAN_TRAITS = ["openness", "conscientiousness", 
                "extraversion", "agreeableness", 
                "neuroticism", "MY_NEW_TRAIT"]

# In TRAIT_DESCRIPTIONS
TRAIT_DESCRIPTIONS["my_new_trait"] = {
    "name": "My New Trait",
    "description": "...",
    "high_characteristics": [...],
    "low_characteristics": [...]
}

# Everything else works automatically!
```

---

## Debugging & Logging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now you'll see:
# 2026-03-27 20:15:42 - src.ml_baseline - INFO - Loading embedding model...
# 2026-03-27 20:15:45 - src.ml_baseline - INFO - Generating embeddings for 10000 texts...
```

### Check Model State

```python
predictor = create_predictor()

# Before training
print(predictor.is_trained)  # False

predictor.train(texts, labels)

# After training
print(predictor.is_trained)  # True
print(predictor.ml_model.is_fitted)  # True
print(predictor.ensemble.ml_weight)  # 0.6

# Check reference distributions
print(predictor.ensemble.reference_distributions["openness"])  # np.ndarray
```

---

## Testing Examples

### Unit Test: TraitRegressor

```python
def test_trait_regressor():
    config = MLConfig()
    regressor = TraitRegressor("openness", config)
    
    # Create dummy data
    X = np.random.randn(100, 384)  # 100 samples, 384D embeddings
    y = np.random.uniform(0, 1, 100)  # 100 personality scores
    
    # Train
    regressor.fit(X, y, cross_validate=True)
    assert regressor.is_fitted
    assert regressor.cv_scores is not None
    
    # Predict
    X_test = np.random.randn(10, 384)
    predictions = regressor.predict(X_test)
    assert predictions.shape == (10,)
    assert np.all((0 <= predictions) & (predictions <= 1))
```

### Integration Test: Full Pipeline

```python
def test_full_pipeline():
    # Create predictor
    predictor = create_predictor(use_mock_llm=True)
    
    # Train
    texts = ["I love new ideas"] * 100
    labels = {trait: np.random.uniform(0, 1, 100) for trait in OCEAN_TRAITS}
    predictor.train(texts, labels)
    
    # Predict
    test_text = "I enjoy exploring new concepts and meeting diverse people."
    prediction = predictor.predict(test_text)
    
    # Validate
    assert all(0 <= prediction.scores[t] <= 1 for t in OCEAN_TRAITS)
    assert all(1 <= prediction.percentiles[t] <= 99 for t in OCEAN_TRAITS)
    assert all(prediction.categories[t] in ["Low", "Medium", "High"] for t in OCEAN_TRAITS)
```

---

## Production Deployment Checklist

- [ ] Set `GEMINI_API_KEY` environment variable
- [ ] Set `use_mock_llm=False` in production
- [ ] Enable CORS for frontend domain
- [ ] Configure rate limiting (15 req/min default)
- [ ] Set up logging and monitoring
- [ ] Load test before production
- [ ] Monitor API latency (target: <5s per request)
- [ ] Monitor error rates
- [ ] Set up alerts for API failures

---

## Useful Commands

```bash
# Training
python train.py --use-llm --learn-weights --run-ablation

# Quick demo
python demo.py

# Inference example
python example_inference.py

# Start backend
python -m uvicorn web_backend.main:app --reload

# Start frontend
cd web_frontend && npm run dev

# Run tests
pytest tests/ -v

# Check code quality
pylint src/

# Profile performance
python -m cProfile -s cumtime example_inference.py
```

---

**Document Version**: 1.0  
**Last Updated**: March 27, 2026  
