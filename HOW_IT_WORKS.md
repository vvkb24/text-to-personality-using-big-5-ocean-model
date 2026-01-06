# How the Personality Detection System Works

## Complete Technical Documentation

**Version:** 1.1.0  
**Last Updated:** January 6, 2026

---

## 📋 Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
3. [ML vs LLM Selection Logic](#3-ml-vs-llm-selection-logic)
4. [ML Baseline Pipeline](#4-ml-baseline-pipeline)
5. [Gemini API Integration](#5-gemini-api-integration)
6. [Ensemble Model](#6-ensemble-model)
7. [Production Safety Layer](#7-production-safety-layer)
8. [Web Application Flow](#8-web-application-flow)
9. [Key Algorithms & Techniques](#9-key-algorithms--techniques)
10. [Mathematics Deep Dive](#10-mathematics-deep-dive)
11. [Mathematical Notation Reference](#11-mathematical-notation-reference)

---

## 1. System Overview

### What Does This System Do?

This system analyzes text (essays, social media posts, emails, etc.) and predicts the author's personality traits based on the **Big Five (OCEAN) model**:

| Trait | What It Measures |
|-------|------------------|
| **O**penness | Creativity, curiosity, openness to new experiences |
| **C**onscientiousness | Organization, dependability, self-discipline |
| **E**xtraversion | Sociability, assertiveness, positive emotions |
| **A**greeableness | Cooperation, trust, empathy |
| **N**euroticism | Emotional instability, anxiety, stress response |

### Output for Each Trait

1. **Score** (0.0 - 1.0): Continuous prediction
2. **Percentile** (1 - 99): Rank compared to reference population
3. **Category**: Low / Medium / High
4. **Evidence**: Specific text excerpts supporting the prediction
5. **Confidence** (0.0 - 1.0): How reliable the prediction is

---

## 2. Architecture Deep Dive

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUT TEXT                          │
│  "I love exploring new ideas and meeting different people..."   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TEXT PREPROCESSING                           │
│  • Whitespace normalization                                      │
│  • Length validation (min 50 characters)                         │
│  • Character count and word count                                │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│     ML BASELINE          │    │     LLM INFERENCE        │
│  (Fast & Consistent)     │    │  (Contextual & Rich)     │
│                          │    │                          │
│  Sentence-BERT → Ridge   │    │  Gemini API → JSON       │
│                          │    │                          │
│  Output: 5 scores        │    │  Output: 5 scores +      │
│                          │    │  evidence + justification│
└──────────────┬───────────┘    └───────────┬──────────────┘
               │                            │
               └──────────────┬─────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ENSEMBLE MODEL                              │
│  • Weighted combination: score = w_ml × ML + w_llm × LLM        │
│  • Learned weights per trait (optimized on training data)       │
│  • Isotonic regression calibration                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PRODUCTION SAFETY LAYER                        │
│  • Percentile clamping [1, 99]                                  │
│  • Confidence estimation                                         │
│  • Response schema validation                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FINAL OUTPUT                              │
│  Scores, Percentiles, Categories, Evidence, Confidence          │
└─────────────────────────────────────────────────────────────────┘
```

### Why Two Models?

| Aspect | ML Baseline | LLM (Gemini) |
|--------|-------------|--------------|
| **Speed** | ~50ms per text | ~2-5s per text |
| **Consistency** | Same input → same output | May vary slightly |
| **Explainability** | Black box embeddings | Extracts evidence quotes |
| **Cost** | Free (local) | API costs |
| **Accuracy** | Good on trained patterns | Better on novel patterns |

**Combining them gives us the best of both worlds.**

---

## 3. ML vs LLM Selection Logic

### The Core Principle: BOTH Are Used Together

The system doesn't choose between ML and LLM—it uses **both simultaneously** in an ensemble. They complement each other:

```
Text Input
    │
    ├──────────────────────────┬──────────────────────────┐
    │                          │                          │
    ▼                          ▼                          │
┌──────────────────┐    ┌──────────────────┐              │
│    ML MODEL      │    │    LLM MODEL     │              │
│    (Local)       │    │    (Gemini)      │              │
│                  │    │                  │              │
│  • Always runs   │    │  • Optional      │              │
│  • ~50ms         │    │  • ~2-5 seconds  │              │
│  • Consistent    │    │  • Evidence      │              │
│  • Free          │    │  • API costs     │              │
└──────────────────┘    └──────────────────┘              │
    │                          │                          │
    ▼                          ▼                          │
 ml_scores                llm_scores                      │
    │                          │                          │
    └────────────┬─────────────┘                          │
                 │                                        │
                 ▼                                        │
    ┌───────────────────────────────┐                     │
    │      ENSEMBLE COMBINE         │                     │
    │                               │                     │
    │  final = 0.6 × ML + 0.4 × LLM │                     │
    │                               │                     │
    └───────────────────────────────┘                     │
                 │                                        │
                 ▼                                        │
         Final Prediction                                 │
```

### Control Parameters

The `predict()` function in `src/pipeline.py` accepts parameters to control model usage:

```python
def predict(
    self,
    text: str,
    include_llm: bool = True,      # ← Controls LLM usage
    include_evidence: bool = True   # ← Controls evidence extraction
) -> PersonalityPrediction:
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `include_llm=True` | ✅ Default | Uses both ML + LLM, combines with ensemble weights |
| `include_llm=False` | | Uses ML only, LLM scores are set to ML scores |
| `include_evidence=True` | ✅ Default | Extracts text evidence from LLM response |
| `include_evidence=False` | | Skips evidence extraction (faster) |

### LLM Availability & Fallback Logic

The system automatically handles missing API keys:

```python
@property
def llm_engine(self) -> Union[LLMInferenceEngine, MockLLMEngine]:
    if self._llm_engine is None:
        if self.config.use_mock_llm:           # ← Config flag
            self._llm_engine = MockLLMEngine()  # Use mock
        else:
            llm_config = LLMConfig(
                api_key=self.config.llm_api_key or os.getenv("GEMINI_API_KEY", "")
            )
            self._llm_engine = create_llm_engine(llm_config)
    return self._llm_engine
```

**Decision Tree:**

```
                    ┌─────────────────────────┐
                    │  use_mock_llm = True?   │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │ YES             │                 │ NO
              ▼                 │                 ▼
    ┌─────────────────┐         │       ┌─────────────────────┐
    │  MockLLMEngine  │         │       │ Has GEMINI_API_KEY? │
    │  (Synthetic)    │         │       └──────────┬──────────┘
    └─────────────────┘         │                  │
                                │    ┌─────────────┼─────────────┐
                                │    │ YES         │             │ NO
                                │    ▼             │             ▼
                                │ ┌──────────────┐ │  ┌─────────────────┐
                                │ │ Real Gemini  │ │  │  MockLLMEngine  │
                                │ │ API Client   │ │  │  (Fallback)     │
                                │ └──────────────┘ │  └─────────────────┘
```

### What Each Engine Does

| Engine | When Used | Behavior |
|--------|-----------|----------|
| **LLMInferenceEngine** | `GEMINI_API_KEY` is set | Calls real Gemini API, extracts evidence |
| **MockLLMEngine** | No API key OR `use_mock_llm=True` | Generates synthetic scores with noise |

### MockLLMEngine Details

When no API key is available, the mock engine simulates LLM behavior:

```python
class MockLLMEngine:
    """Mock LLM for testing without API access."""
    
    def predict(self, text: str) -> LLMPrediction:
        # Generate scores based on text characteristics
        scores = {}
        for trait in OCEAN_TRAITS:
            # Base score with random variation
            base = 0.5 + np.random.randn() * 0.15
            scores[trait] = np.clip(base, 0.1, 0.9)
        
        return LLMPrediction(
            scores=scores,
            evidence={trait: [] for trait in OCEAN_TRAITS},  # No evidence
            justifications={trait: "Mock prediction" for trait in OCEAN_TRAITS}
        )
```

### Training Mode Options

During training, LLM usage can be controlled separately:

```python
def train(
    self,
    train_texts: List[str],
    train_labels: Dict[str, np.ndarray],
    val_texts: List[str] = None,
    val_labels: Dict[str, np.ndarray] = None,
    use_llm_for_training: bool = False    # ← Training-specific flag
):
```

| `use_llm_for_training` | Effect |
|------------------------|--------|
| `False` (default) | ML predictions + noise used as LLM proxy (fast) |
| `True` | Real LLM called for each training sample (slow, accurate) |

**Why default to False?** Calling the LLM API for hundreds of training samples is slow and expensive. Using ML predictions with added noise provides a reasonable proxy for learning ensemble weights.

### Ensemble Weight Summary

| Scenario | ML Weight | LLM Weight | Result |
|----------|-----------|------------|--------|
| Default config | 0.6 | 0.4 | 60% ML, 40% LLM |
| `include_llm=False` | 1.0 | 0.0 | ML only |
| After training | Learned | Learned | Optimized per trait |

### Per-Trait Learned Weights

After training on labeled data, each trait gets its own optimal weights:

| Trait | Typical ML Weight | Typical LLM Weight |
|-------|-------------------|-------------------|
| Openness | 0.95 | 0.05 |
| Conscientiousness | 1.00 | 0.00 |
| Extraversion | 0.95 | 0.05 |
| Agreeableness | 1.00 | 0.00 |
| Neuroticism | 0.90 | 0.10 |

**Note:** On synthetic training data, ML tends to dominate. With diverse real-world data, LLM weights would likely increase.

### Quick Reference

| Question | Answer |
|----------|--------|
| Is ML always used? | ✅ **Yes**, always runs |
| Is LLM always used? | ⚠️ **By default yes**, can be disabled |
| What if no API key? | MockLLMEngine generates synthetic scores |
| Which is faster? | ML (~50ms) vs LLM (~2-5s) |
| Which costs money? | Only LLM (Gemini API) |
| Can I use ML only? | Yes, set `include_llm=False` |

---

## 4. ML Baseline Pipeline

### Step 1: Text Embedding with Sentence-BERT

**What is Sentence-BERT?**

Sentence-BERT (SBERT) is a modification of the BERT transformer that produces semantically meaningful sentence embeddings. Unlike word embeddings, SBERT captures the meaning of entire sentences.

**Model Used:** `sentence-transformers/all-MiniLM-L6-v2`

- **Embedding Dimension:** 384
- **Training:** Trained on 1B+ sentence pairs
- **Speed:** ~14,000 sentences/second on GPU

**How It Works:**

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Convert text to 384-dimensional vector
text = "I love exploring new ideas and meeting different people"
embedding = model.encode(text)  # Shape: (384,)
```

**Why This Model?**

1. **Compact:** 384 dimensions (vs 768 for larger models)
2. **Fast:** 5x faster than BERT-base
3. **Quality:** Top performance on semantic similarity benchmarks
4. **Memory:** ~90MB model size

### Step 2: Ridge Regression (Per-Trait)

For each OCEAN trait, we train a separate Ridge regression model.

**What is Ridge Regression?**

Ridge regression is linear regression with L2 regularization:

$$\hat{y} = \mathbf{w}^T \mathbf{x} + b$$

**Loss Function:**

$$\mathcal{L}(\mathbf{w}) = \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 + \alpha \|\mathbf{w}\|_2^2$$

Where:
- $\alpha = 1.0$ (regularization strength)
- $\|\mathbf{w}\|_2^2$ = sum of squared weights (prevents overfitting)

**Why Ridge Regression?**

1. **Handles high-dimensional data:** 384 features, potentially few samples
2. **Prevents overfitting:** L2 penalty shrinks weights
3. **Fast training:** Closed-form solution
4. **Interpretable:** Linear combination of features

**Training Code Structure:**

```python
from sklearn.linear_model import Ridge

class TraitRegressor:
    def __init__(self, trait_name, alpha=1.0):
        self.trait = trait_name
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
    
    def fit(self, embeddings, labels):
        # Standardize features
        X = self.scaler.fit_transform(embeddings)
        # Fit ridge regression
        self.model.fit(X, labels)
    
    def predict(self, embeddings):
        X = self.scaler.transform(embeddings)
        return self.model.predict(X)
```

### Step 3: Cross-Validation

We use 5-fold cross-validation to estimate model performance:

```
Data: [████████████████████]
      
Fold 1: [TTTT|████████████████]  Train on 80%, Test on 20%
Fold 2: [████|TTTT|████████████]
Fold 3: [████████|TTTT|████████]
Fold 4: [████████████|TTTT|████]
Fold 5: [████████████████|TTTT]

Final Score = Average of 5 fold scores
```

**Metrics Computed:**
- **R² Score:** How much variance is explained (0-1)
- **MAE:** Mean Absolute Error
- **Pearson r:** Correlation with true labels

---

## 5. Gemini API Integration

### Overview

Google's Gemini is a multimodal large language model. We use the `gemini-1.5-flash` model for fast, cost-effective personality inference.

### API Configuration

```python
@dataclass
class LLMConfig:
    api_key: str = ""
    model: str = "gemini-1.5-flash"
    temperature: float = 0.3          # Low for consistency
    max_output_tokens: int = 2048
    max_retries: int = 3
    retry_delay: float = 2.0
    requests_per_minute: int = 15     # Rate limiting
    timeout: int = 60
```

### Prompt Engineering

**System Prompt Structure:**

```
You are an expert psychologist specializing in personality assessment 
using the Big Five (OCEAN) model.

## Big Five Personality Traits:

**Openness to Experience (openness)**
- Description: Reflects imagination, creativity, intellectual curiosity...
- High scorers tend to be: creative, curious, imaginative...
- Low scorers tend to be: conventional, practical, prefers routine...

[... similar for other 4 traits ...]

## Analysis Guidelines:
1. Look for linguistic markers: word choice, sentence structure
2. Consider content themes: topics discussed, interests expressed
3. Analyze writing style: formal vs informal, detailed vs brief
4. Identify behavioral indicators: described actions, preferences
5. Be objective and base assessments on textual evidence only

## Important:
- Scores should be between 0.0 and 1.0
- 0.5 represents average/neutral
- Provide specific text excerpts as evidence
- Be conservative - avoid extreme scores without strong evidence
```

**Analysis Prompt:**

```
Analyze the following text and assess the author's Big Five personality traits.

## Text to Analyze:
"""
[USER'S TEXT HERE]
"""

## Required Output Format:
Respond with ONLY a valid JSON object in this exact format:
{
    "openness": {
        "score": <float 0.0-1.0>,
        "evidence": ["<quote from text>", "<quote from text>"],
        "justification": "<brief explanation>"
    },
    "conscientiousness": { ... },
    "extraversion": { ... },
    "agreeableness": { ... },
    "neuroticism": { ... }
}
```

### Why This Prompt Design?

1. **Role Setting:** "Expert psychologist" primes for professional analysis
2. **Trait Definitions:** Gives model the psychological framework
3. **Guidelines:** Specific instructions prevent hallucination
4. **Structured Output:** JSON format enables reliable parsing
5. **Evidence Requirement:** Forces model to cite specific text

### Rate Limiting & Retry Logic

```python
def _rate_limit(self):
    """Apply rate limiting between requests."""
    elapsed = time.time() - self.last_request_time
    if elapsed < self.min_request_interval:
        sleep_time = self.min_request_interval - elapsed
        time.sleep(sleep_time)
    self.last_request_time = time.time()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_api(self, prompt: str) -> str:
    """Call the Gemini API with retry logic."""
    self._rate_limit()
    response = self.client.generate_content(prompt, generation_config={...})
    return response.text
```

**Retry Strategy:**
- **Exponential backoff:** Wait 2s, 4s, 8s between retries
- **Max 3 attempts:** Fail gracefully after that

### Response Parsing

The LLM returns JSON which we parse and validate:

```python
def _parse_response(self, response_text: str) -> Dict:
    # Try direct JSON parsing first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from response (sometimes LLM adds extra text)
    json_pattern = r'\{[\s\S]*\}'
    matches = re.findall(json_pattern, response_text)
    
    for match in matches:
        try:
            parsed = json.loads(match)
            if all(trait in parsed for trait in OCEAN_TRAITS):
                return parsed
        except json.JSONDecodeError:
            continue
    
    raise ValueError("Could not parse JSON from response")
```

### Mock LLM for Testing

When no API key is available, we use a mock that generates realistic scores:

```python
class MockLLMEngine:
    """Mock LLM for testing without API access."""
    
    def predict(self, text: str) -> LLMPrediction:
        # Generate scores based on text characteristics
        scores = {}
        for trait in OCEAN_TRAITS:
            base = 0.5 + np.random.randn() * 0.15
            scores[trait] = np.clip(base, 0.1, 0.9)
        
        return LLMPrediction(
            scores=scores,
            evidence={trait: [] for trait in OCEAN_TRAITS},
            justifications={trait: "Mock prediction" for trait in OCEAN_TRAITS}
        )
```

---

## 6. Ensemble Model

### Why Ensemble?

| Individual Model | Weakness |
|------------------|----------|
| ML Baseline | May miss nuanced context |
| LLM | Slower, may hallucinate, costs money |

**Ensemble combines strengths:** ML provides consistent baseline, LLM adds contextual understanding.

### Weighted Combination

```python
ensemble_score = w_ml × ml_score + w_llm × llm_score
```

**Default Weights:** ML = 0.6, LLM = 0.4

### Learned Weights (Per-Trait)

Weights are optimized on training data using grid search:

```python
def _learn_weights(self, ml_predictions, llm_predictions, targets):
    for trait in OCEAN_TRAITS:
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
        
        self.weights[trait] = {"ml": best_weight, "llm": 1 - best_weight}
```

**Typical Learned Weights:**

| Trait | ML Weight | LLM Weight |
|-------|-----------|------------|
| Openness | 0.95 | 0.05 |
| Conscientiousness | 1.00 | 0.00 |
| Extraversion | 0.95 | 0.05 |
| Agreeableness | 1.00 | 0.00 |
| Neuroticism | 0.90 | 0.10 |

**Why ML dominates?** On synthetic training data, ML learns the patterns perfectly. With real diverse data, LLM weights would likely increase.

### Isotonic Regression Calibration

**Problem:** Raw scores may not reflect true probabilities.

**Solution:** Isotonic regression maps scores to better-calibrated values.

```
Raw Score:       0.2  0.4  0.5  0.6  0.8
                  │    │    │    │    │
                  ▼    ▼    ▼    ▼    ▼
Calibrated:      0.15 0.35 0.48 0.62 0.85
```

**Isotonic Regression Properties:**
- **Monotonic:** Higher input → higher output (always)
- **Non-parametric:** Learns shape from data
- **Bounded:** Output clipped to [0, 1]

```python
from sklearn.isotonic import IsotonicRegression

calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(raw_predictions, true_labels)
calibrated = calibrator.predict(new_predictions)
```

### Percentile Calculation

Percentiles show where a score ranks compared to the training population:

```python
from scipy.stats import percentileofscore

def get_percentile(score, reference_distribution):
    return percentileofscore(reference_distribution, score, kind='mean')
```

**Example:**
- Training scores for Openness: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
- New score: 0.65
- Percentile: 75th (higher than 75% of training examples)

---

## 7. Production Safety Layer

### Percentile Clamping

**Problem:** 0th or 100th percentile implies certainty we can't guarantee.

**Solution:** Clamp percentiles to [1, 99]:

```python
PERCENTILE_MIN = 1.0
PERCENTILE_MAX = 99.0

def clamp_percentile(percentile: float) -> float:
    return float(np.clip(percentile, PERCENTILE_MIN, PERCENTILE_MAX))
```

### Confidence Estimation

Confidence is computed from two factors:

**1. Text Length Confidence:**
```python
def estimate_text_length_confidence(text_length: int) -> float:
    if text_length < 50:
        return max(0.1, text_length / 50 * 0.5)  # Very low
    if text_length >= 2000:
        return 1.0  # Maximum
    # Logarithmic scaling for middle range
    normalized = (text_length - 50) / (200 - 50)
    return float(np.clip(0.5 + 0.5 * np.tanh(normalized - 0.5), 0.5, 1.0))
```

**2. Prediction Stability Confidence:**
```python
def estimate_prediction_stability_confidence(ml_score, llm_score, ensemble_score):
    # High agreement between ML and LLM = high confidence
    disagreement = abs(ml_score - llm_score)
    stability = np.exp(-disagreement * 3)  # Exponential decay
    
    # Penalty for extreme scores (often less reliable)
    extremity_penalty = 1.0 - 0.1 * (abs(ensemble_score - 0.5) * 2) ** 2
    
    return float(np.clip(stability * extremity_penalty, 0.3, 1.0))
```

**Combined Confidence:**
```python
confidence = 0.3 * text_length_factor + 0.7 * stability_factor
```

### Text Validation

```python
@dataclass
class TextValidationResult:
    is_valid: bool
    cleaned_text: str
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    character_count: int = 0
    word_count: int = 0

def validate_text_for_prediction(text: str, min_length: int = 50) -> TextValidationResult:
    if text is None:
        return TextValidationResult(
            is_valid=False,
            error_message="Text input is required."
        )
    
    cleaned = ' '.join(text.strip().split())
    char_count = len(cleaned)
    
    if char_count < min_length:
        return TextValidationResult(
            is_valid=False,
            error_message=f"Text too short ({char_count} chars). Need {min_length}+."
        )
    
    warning = None
    if char_count < 200:
        warning = "Short text may produce less accurate results."
    
    return TextValidationResult(
        is_valid=True,
        cleaned_text=cleaned,
        warning_message=warning,
        character_count=char_count
    )
```

---

## 8. Web Application Flow

### Backend (FastAPI)

```
┌─────────────────────────────────────────────────────────────┐
│                    STARTUP (Once)                            │
│  1. Load Sentence-BERT model                                │
│  2. Generate synthetic training data                        │
│  3. Train Ridge regression models                           │
│  4. Initialize ensemble with calibration                    │
│  5. Store reference distributions                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              REQUEST: POST /predict                          │
│  Body: { "text": "..." }                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              PROCESSING                                      │
│  1. Pydantic validation (min 50 chars)                      │
│  2. Text preprocessing                                       │
│  3. ML prediction (embeddings → ridge)                      │
│  4. LLM prediction (Gemini API, if available)               │
│  5. Ensemble combination                                     │
│  6. Calibration                                              │
│  7. Percentile calculation                                   │
│  8. Confidence estimation                                    │
│  9. Percentile clamping                                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              RESPONSE                                        │
│  {                                                          │
│    "scores": {...},                                         │
│    "percentiles": {...},                                    │
│    "categories": {...},                                     │
│    "evidence": {...},                                       │
│    "confidences": {...},                                    │
│    "warning": "..." or null                                 │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
```

### Frontend (React)

```
User Types Text
      │
      ▼
TextInput Component
      │
      ├── Character count validation
      ├── Enable/disable submit button
      │
      ▼
Submit → fetch('/predict', {method: 'POST', body: {text}})
      │
      ▼
Loading State (LoadingSpinner)
      │
      ▼
Response Received
      │
      ├── Error? → ErrorMessage component
      │
      └── Success? → ResultsDisplay
                          │
                          ├── RadarChart (scores visualization)
                          ├── TraitCard × 5 (individual traits)
                          │     ├── Score bar
                          │     ├── Percentile
                          │     ├── Category badge
                          │     ├── Confidence indicator
                          │     └── Evidence (expandable)
                          └── Summary Stats
```

---

## 9. Key Algorithms & Techniques

### 1. Sentence Embeddings (SBERT)

**Technique:** Siamese/Triplet networks trained on sentence pairs

**How SBERT Works:**
1. Pass sentence through BERT
2. Apply mean pooling over tokens
3. Result: 384-dimensional dense vector

**Properties:**
- Similar sentences → similar vectors (cosine similarity)
- Captures semantic meaning, not just keywords
- Transfer learning from massive text corpora

### 2. Ridge Regression (L2 Regularization)

**Closed-Form Solution:**
$$\mathbf{w} = (\mathbf{X}^T\mathbf{X} + \alpha\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$

**Benefits:**
- Always has a solution (unlike ordinary least squares)
- Handles multicollinearity
- Shrinks weights smoothly toward zero

### 3. Prompt Engineering

**Techniques Used:**
1. **Role prompting:** "You are an expert psychologist..."
2. **Few-shot examples:** Trait definitions with characteristics
3. **Structured output:** JSON schema specification
4. **Guidelines:** Explicit instructions for edge cases
5. **Evidence extraction:** "Provide specific text excerpts..."

### 4. Ensemble Learning

**Type:** Weighted average ensemble

**Advantages:**
- Reduces variance (averaging smooths predictions)
- Captures different aspects of the problem
- Robust to individual model failures

### 5. Isotonic Regression

**Technique:** Non-parametric monotonic regression

**Algorithm:** Pool Adjacent Violators (PAV)
1. Start with raw scores
2. Find adjacent pairs that violate monotonicity
3. Replace with their weighted average
4. Repeat until monotonic

### 6. Rate Limiting (Token Bucket)

**Implementation:**
```python
min_interval = 60.0 / requests_per_minute  # e.g., 4 seconds

def rate_limit():
    elapsed = time.time() - last_request_time
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    last_request_time = time.time()
```

### 7. Exponential Backoff (Retries)

**Formula:**
$$\text{wait}_n = \min(\text{base} \times 2^n, \text{max\_wait})$$

**Implementation:**
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def call_api():
    ...
```

---

## Summary

This personality detection system combines:

1. **Transformer embeddings** (Sentence-BERT) for rich text representation
2. **Supervised regression** (Ridge) for fast, consistent predictions
3. **Large language model** (Gemini) for contextual analysis and evidence
4. **Ensemble learning** for best-of-both-worlds accuracy
5. **Calibration** (Isotonic regression) for reliable probability estimates
6. **Production hardening** for safety and reliability in production

The result is a robust, explainable personality prediction system suitable for research and production use.

---

## 10. Mathematics Deep Dive

### 10.1 Sentence-BERT Embeddings

#### Mean Pooling

Given a sentence with $n$ tokens, BERT produces hidden states $\mathbf{H} \in \mathbb{R}^{n \times d}$ where $d = 768$ for BERT-base.

**Mean pooling** computes the sentence embedding:

$$\mathbf{e} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{h}_i$$

For `all-MiniLM-L6-v2`, the output is projected to $d = 384$ dimensions.

#### Cosine Similarity

To measure semantic similarity between two sentence embeddings $\mathbf{a}$ and $\mathbf{b}$:

$$\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} = \frac{\sum_{i=1}^{d} a_i b_i}{\sqrt{\sum_{i=1}^{d} a_i^2} \cdot \sqrt{\sum_{i=1}^{d} b_i^2}}$$

**Range:** $[-1, 1]$ where:
- $1$ = identical meaning
- $0$ = orthogonal (unrelated)
- $-1$ = opposite meaning

---

### 10.2 Ridge Regression

#### Problem Setup

Given:
- Training data: $\{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$ where $\mathbf{x}_i \in \mathbb{R}^{384}$ (embedding), $y_i \in [0,1]$ (trait score)
- Goal: Find weights $\mathbf{w} \in \mathbb{R}^{384}$ and bias $b \in \mathbb{R}$

#### Prediction Function

$$\hat{y} = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^{384} w_j x_j + b$$

#### Loss Function (L2 Regularized)

$$\mathcal{L}(\mathbf{w}, b) = \underbrace{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}_{\text{Mean Squared Error}} + \underbrace{\alpha \|\mathbf{w}\|_2^2}_{\text{L2 Penalty}}$$

Expanded:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \left( y_i - \mathbf{w}^T \mathbf{x}_i - b \right)^2 + \alpha \sum_{j=1}^{384} w_j^2$$

Where $\alpha = 1.0$ is the regularization strength.

#### Why L2 Regularization?

Without regularization ($\alpha = 0$), high-dimensional data can lead to:
- **Overfitting:** Model memorizes training data
- **Multicollinearity:** Features are correlated, weights become unstable

L2 penalty **shrinks** weights toward zero:
$$w_j^{\text{ridge}} = \frac{w_j^{\text{OLS}}}{1 + \alpha \lambda_j}$$

where $\lambda_j$ is related to data variance.

#### Closed-Form Solution

In matrix form with design matrix $\mathbf{X} \in \mathbb{R}^{N \times d}$ and targets $\mathbf{y} \in \mathbb{R}^N$:

$$\mathbf{w}^* = (\mathbf{X}^T\mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$

**Key insight:** Adding $\alpha \mathbf{I}$ ensures the matrix is always invertible, even when $N < d$ (more features than samples).

#### Gradient Descent Alternative

For large datasets, we can use gradient descent:

$$\frac{\partial \mathcal{L}}{\partial w_j} = -\frac{2}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i) x_{ij} + 2\alpha w_j$$

Update rule:

$$w_j \leftarrow w_j - \eta \left( -\frac{2}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i) x_{ij} + 2\alpha w_j \right)$$

---

### 10.3 Evaluation Metrics

#### R² Score (Coefficient of Determination)

Measures the proportion of variance explained by the model:

$$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{N}(y_i - \bar{y})^2}$$

Where $\bar{y} = \frac{1}{N}\sum_{i=1}^N y_i$ is the mean.

**Interpretation:**
- $R^2 = 1$: Perfect prediction
- $R^2 = 0$: Model predicts the mean (no better than baseline)
- $R^2 < 0$: Model is worse than predicting the mean

#### Mean Absolute Error (MAE)

Average magnitude of errors:

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

**Advantages:**
- Same units as target variable
- Robust to outliers (compared to MSE)
- Interpretable: "On average, predictions are off by MAE"

#### Pearson Correlation Coefficient

Measures linear correlation between predictions and true values:

$$r = \frac{\sum_{i=1}^{N}(y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum_{i=1}^{N}(y_i - \bar{y})^2} \cdot \sqrt{\sum_{i=1}^{N}(\hat{y}_i - \bar{\hat{y}})^2}}$$

**Range:** $[-1, 1]$
- $r = 1$: Perfect positive correlation
- $r = 0$: No linear correlation
- $r = -1$: Perfect negative correlation

**Relationship to R²:** For simple regression, $R^2 = r^2$

---

### 10.4 Cross-Validation

#### K-Fold Cross-Validation

Dataset $D$ is split into $K$ folds: $D = D_1 \cup D_2 \cup ... \cup D_K$

For each fold $k$:
- **Train** on $D \setminus D_k$ (all data except fold $k$)
- **Test** on $D_k$
- Record score $s_k$

**Final Score:**

$$\bar{s} = \frac{1}{K} \sum_{k=1}^{K} s_k$$

**Standard Error:**

$$SE = \frac{\sigma_s}{\sqrt{K}} = \frac{1}{\sqrt{K}} \sqrt{\frac{1}{K-1} \sum_{k=1}^{K} (s_k - \bar{s})^2}$$

**Why K=5?**
- **Bias-Variance Tradeoff:** Lower K = higher bias, higher K = higher variance
- K=5 or K=10 are empirically good choices
- Computational cost scales with K

---

### 10.5 Ensemble Weighting

#### Weighted Average

For ML prediction $p_{ML}$ and LLM prediction $p_{LLM}$:

$$p_{\text{ensemble}} = w_{ML} \cdot p_{ML} + w_{LLM} \cdot p_{LLM}$$

**Constraint:** $w_{ML} + w_{LLM} = 1$ (weights sum to 1)

#### Grid Search Optimization

Find optimal weights by minimizing MAE on validation data:

$$w^*_{ML} = \arg\min_{w \in [0,1]} \frac{1}{N} \sum_{i=1}^{N} |y_i - (w \cdot p_{ML,i} + (1-w) \cdot p_{LLM,i})|$$

**Algorithm:**
```
best_w = 0.5
best_mae = ∞

for w in {0.00, 0.05, 0.10, ..., 1.00}:
    predictions = w × ML + (1-w) × LLM
    mae = mean(|y - predictions|)
    if mae < best_mae:
        best_mae = mae
        best_w = w

return best_w
```

**Complexity:** $O(N \times G)$ where $G = 21$ grid points

#### Why Per-Trait Weights?

Different traits may have different optimal weights because:
- Some traits are better captured by linguistic patterns (ML)
- Some require contextual understanding (LLM)

---

### 10.6 Isotonic Regression (Calibration)

#### Problem

Raw model outputs may not be well-calibrated probabilities. A score of 0.7 should mean "70% confidence" but often doesn't.

#### Isotonic Constraint

Calibration function $f$ must be **monotonically non-decreasing**:

$$x_1 \leq x_2 \implies f(x_1) \leq f(x_2)$$

#### Pool Adjacent Violators (PAV) Algorithm

Given sorted predictions $\hat{y}_1 \leq \hat{y}_2 \leq ... \leq \hat{y}_N$ with true labels $y_1, y_2, ..., y_N$:

**Step 1:** Initialize $f(\hat{y}_i) = y_i$

**Step 2:** Find adjacent violators (where $f(\hat{y}_i) > f(\hat{y}_{i+1})$)

**Step 3:** Pool them by taking weighted average:
$$f(\hat{y}_i) = f(\hat{y}_{i+1}) = \frac{w_i \cdot y_i + w_{i+1} \cdot y_{i+1}}{w_i + w_{i+1}}$$

**Step 4:** Repeat until no violators remain

**Example:**
```
Raw:    [0.2, 0.4, 0.6, 0.8]
Labels: [0.3, 0.2, 0.7, 0.9]  ← Violation: 0.3 > 0.2

After PAV:
Pool positions 1,2: (0.3 + 0.2) / 2 = 0.25

Result: [0.25, 0.25, 0.7, 0.9]  ← Now monotonic!
```

#### Prediction for New Points

Use **linear interpolation** between calibration points:

$$f(x) = f(x_i) + \frac{f(x_{i+1}) - f(x_i)}{x_{i+1} - x_i} (x - x_i)$$

where $x_i \leq x \leq x_{i+1}$

---

### 10.7 Percentile Calculation

#### Definition

The percentile rank of score $s$ in distribution $D = \{d_1, d_2, ..., d_N\}$:

$$\text{percentile}(s, D) = \frac{|\{d_i : d_i < s\}| + 0.5 \times |\{d_i : d_i = s\}|}{N} \times 100$$

This is the "mean" method, handling ties by giving them the average rank.

#### Example

Distribution: $D = [0.3, 0.4, 0.5, 0.6, 0.7]$
Score: $s = 0.55$

- Values less than 0.55: {0.3, 0.4, 0.5} → 3 values
- Values equal to 0.55: {} → 0 values

$$\text{percentile} = \frac{3 + 0.5 \times 0}{5} \times 100 = 60\%$$

#### Percentile Clamping

To avoid certainty claims:

$$\text{percentile}_{\text{safe}} = \text{clip}(\text{percentile}, 1, 99)$$

$$= \begin{cases} 
1 & \text{if percentile} < 1 \\
99 & \text{if percentile} > 99 \\
\text{percentile} & \text{otherwise}
\end{cases}$$

---

### 10.8 Confidence Estimation

#### Text Length Factor

Short texts provide less signal. Using logistic-like scaling:

$$c_{\text{length}}(n) = \begin{cases}
\max(0.1, \frac{n}{50} \times 0.5) & \text{if } n < 50 \\
0.5 + 0.5 \times \tanh\left(\frac{n-50}{200-50} - 0.5\right) & \text{if } 50 \leq n < 2000 \\
1.0 & \text{if } n \geq 2000
\end{cases}$$

Where $n$ = character count.

**Hyperbolic Tangent:**
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

Range: $(-1, 1)$, smooth S-curve centered at 0.

#### Prediction Stability Factor

Agreement between ML and LLM models:

$$\text{disagreement} = |p_{ML} - p_{LLM}|$$

$$c_{\text{stability}} = e^{-3 \times \text{disagreement}}$$

**Properties:**
- Perfect agreement (disagreement = 0): $c = e^0 = 1.0$
- Large disagreement (disagreement = 0.5): $c = e^{-1.5} \approx 0.22$

#### Extremity Penalty

Predictions near 0 or 1 are often less reliable:

$$c_{\text{extremity}} = 1.0 - 0.1 \times (2 \times |p_{\text{ensemble}} - 0.5|)^2$$

**Example:**
- $p = 0.5$: penalty = $1.0 - 0.1 \times 0^2 = 1.0$ (no penalty)
- $p = 0.9$: penalty = $1.0 - 0.1 \times 0.8^2 = 0.936$

#### Combined Confidence

$$c_{\text{final}} = \text{clip}\left(0.3 \times c_{\text{length}} + 0.7 \times (c_{\text{stability}} \times c_{\text{extremity}}), 0.3, 1.0\right)$$

**Weights:** Stability (70%) is weighted more than length (30%) because model agreement is a stronger signal.

---

### 10.9 Rate Limiting Mathematics

#### Token Bucket Algorithm (Simplified)

**Parameters:**
- $R$ = requests per minute (e.g., 15)
- $T_{\min} = \frac{60}{R}$ = minimum interval between requests

**Algorithm:**
```
t_last = 0  # timestamp of last request

def rate_limit():
    t_now = current_time()
    Δt = t_now - t_last
    
    if Δt < T_min:
        sleep(T_min - Δt)
    
    t_last = current_time()
```

For $R = 15$ req/min: $T_{\min} = \frac{60}{15} = 4$ seconds

#### Exponential Backoff

On retry attempt $n$ (starting from 0):

$$T_{\text{wait}}(n) = \min(\text{base} \times 2^n, T_{\max})$$

With base = 2s and $T_{\max}$ = 10s:

| Attempt | Wait Time |
|---------|-----------|
| 1 | $\min(2 \times 2^0, 10) = 2$s |
| 2 | $\min(2 \times 2^1, 10) = 4$s |
| 3 | $\min(2 \times 2^2, 10) = 8$s |
| 4 | $\min(2 \times 2^3, 10) = 10$s |

**Total maximum wait:** $2 + 4 + 8 = 14$s for 3 retries.

---

### 10.10 Feature Standardization

Before Ridge regression, features are standardized:

$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

Where:
- $\mu_j = \frac{1}{N}\sum_{i=1}^N x_{ij}$ (feature mean)
- $\sigma_j = \sqrt{\frac{1}{N-1}\sum_{i=1}^N (x_{ij} - \mu_j)^2}$ (feature std)

**Why Standardize?**
1. **Fair regularization:** Without standardization, features with larger scales would be penalized less
2. **Numerical stability:** Gradient descent converges faster
3. **Interpretability:** Weights reflect relative importance

---

### 10.11 Softmax for Category Assignment

Categories (Low/Medium/High) are assigned based on percentile:

$$\text{category} = \begin{cases}
\text{Low} & \text{if percentile} < 33 \\
\text{Medium} & \text{if } 33 \leq \text{percentile} < 67 \\
\text{High} & \text{if percentile} \geq 67
\end{cases}$$

More sophisticated systems might use softmax for probabilistic categories:

$$P(\text{category}_k | s) = \frac{e^{a_k(s)}}{\sum_{j=1}^{3} e^{a_j(s)}}$$

Where $a_k(s)$ is an activation function for each category.

---

## 11. Mathematical Notation Reference

| Symbol | Meaning |
|--------|---------|
| $\mathbf{x}$ | Feature vector (embedding) |
| $y$ | True label (trait score) |
| $\hat{y}$ | Predicted value |
| $\mathbf{w}$ | Weight vector |
| $b$ | Bias term |
| $\alpha$ | Regularization strength |
| $N$ | Number of samples |
| $d$ | Feature dimension (384) |
| $\|\cdot\|_2$ | L2 norm (Euclidean) |
| $\mathbf{I}$ | Identity matrix |
| $\mathbf{X}^T$ | Matrix transpose |
| $\eta$ | Learning rate |
| $\tanh$ | Hyperbolic tangent |
| $\text{clip}(x, a, b)$ | Clamp $x$ to $[a, b]$ |

---

*Documentation generated: January 6, 2026*

