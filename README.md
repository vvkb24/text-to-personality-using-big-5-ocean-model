# Personality Detection System using Big Five (OCEAN) Model

A research-grade, end-to-end personality trait detection system that analyzes English text to predict Big Five personality traits with continuous scores, percentiles, categorical labels, and evidence-based explanations.

## 🎯 Overview

This system implements a state-of-the-art approach to personality detection combining:
- **Transformer-based ML Baseline**: Sentence embeddings + supervised regression
- **LLM-based Inference**: Google Gemini for contextual analysis and evidence extraction
- **Ensemble Learning**: Weighted combination with learned weights and calibration

### Key Features

- ✅ **Continuous OCEAN Scores** (0-1 scale)
- ✅ **Percentile Rankings** (relative to training distribution)
- ✅ **Category Labels** (Low / Medium / High)
- ✅ **Evidence Sentences** for each trait prediction
- ✅ **Confidence Scores** based on model agreement
- ✅ **Comprehensive Evaluation** with ablation studies

## 📊 The Big Five (OCEAN) Model

| Trait | Description | High Scorers | Low Scorers |
|-------|-------------|--------------|-------------|
| **O**penness | Creativity, curiosity, intellectual interests | Creative, curious, imaginative | Conventional, practical |
| **C**onscientiousness | Organization, dependability, self-discipline | Organized, reliable, goal-oriented | Spontaneous, flexible |
| **E**xtraversion | Sociability, assertiveness, positive emotions | Outgoing, energetic, talkative | Reserved, quiet, solitary |
| **A**greeableness | Cooperation, trust, empathy | Helpful, trusting, empathetic | Competitive, skeptical |
| **N**euroticism | Emotional instability, anxiety | Anxious, moody, stressed | Calm, stable, resilient |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Text                                │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
┌─────────────────────┐       ┌─────────────────────┐
│   ML Baseline       │       │   LLM Inference     │
│   - SBERT Embeddings│       │   - Gemini API      │
│   - Ridge Regression│       │   - Evidence Extract│
│   - Per-trait Models│       │   - Structured JSON │
└─────────┬───────────┘       └─────────┬───────────┘
          │                             │
          │     ┌─────────────────┐     │
          └────►│   Ensemble      │◄────┘
                │   - Learned     │
                │     Weights     │
                │   - Calibration │
                │   - Percentiles │
                └────────┬────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Output: Scores, Percentiles, Categories, Evidence          │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
personality-detection/
├── config/
│   └── settings.yaml        # Configuration file
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── ml_baseline.py       # ML baseline model
│   ├── llm_inference.py     # LLM inference engine
│   ├── ensemble.py          # Ensemble and calibration
│   ├── evaluation.py        # Evaluation framework
│   ├── ablation.py          # Ablation studies
│   └── pipeline.py          # Unified inference pipeline
├── train.py                 # Main training script
├── example_inference.py     # Inference examples
├── requirements.txt         # Dependencies
├── .env                     # Environment variables
└── README.md               # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to the project
cd "personality detection vscode"

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Set up your Gemini API key in `.env`:
```
GEMINI_API_KEY=your_api_key_here
```

2. Or use the mock LLM for testing without an API key.

### Training

```bash
# Basic training with synthetic data
python train.py --data-source synthetic --min-samples 1000

# Training with HuggingFace dataset
python train.py --data-source huggingface --dataset-name "Fatima0923/Automated-Personality-Prediction"

# Training with LLM integration
python train.py --use-llm --llm-samples 100

# Full training with ablation studies
python train.py --use-llm --run-ablation --learn-weights
```

### Inference

```bash
# Run example inference
python example_inference.py
```

### Programmatic Usage

```python
from src.pipeline import create_predictor

# Create predictor
predictor = create_predictor(
    api_key="your_gemini_api_key",
    use_mock_llm=False,  # Set True for testing without API
    ml_weight=0.6,
    llm_weight=0.4
)

# Train (or load pre-trained model)
predictor.train(train_texts, train_labels)

# Single prediction
prediction = predictor.predict("Your text here...")
print(prediction.summary())

# Detailed analysis
analysis = predictor.analyze("Your text here...")
print(analysis)

# Batch prediction
predictions = predictor.predict_batch(list_of_texts)
```

## 📈 Methodology

### 1. Data Pipeline

- **Source**: HuggingFace `Fatima0923/Automated-Personality-Prediction` dataset
- **Preprocessing**: URL removal, whitespace normalization, length filtering
- **Split**: 70% train / 15% validation / 15% test
- **Normalization**: Trait scores normalized to [0, 1] range

### 2. ML Baseline

- **Embeddings**: Sentence-BERT (all-MiniLM-L6-v2)
- **Model**: Ridge regression (per-trait)
- **Cross-validation**: 5-fold CV during training

### 3. LLM Inference

- **Model**: Google Gemini 1.5 Flash
- **Prompt Engineering**: Psychologically-grounded prompts with trait definitions
- **Output**: Structured JSON with scores, evidence, and justifications
- **Rate Limiting**: Built-in request throttling

### 4. Ensemble & Calibration

- **Combination**: Weighted average with learnable weights
- **Calibration**: Isotonic regression for score calibration
- **Percentiles**: Based on training distribution
- **Categories**: Threshold-based (33rd/67th percentiles)

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Pearson r | Correlation between predictions and ground truth |
| Spearman ρ | Rank correlation (robust to outliers) |
| MAE | Mean Absolute Error |
| R² | Coefficient of determination |
| RMSE | Root Mean Square Error |

## 🔬 Ablation Studies

The system includes comprehensive ablation studies:

1. **Model Comparison**: ML-only vs LLM-only vs Ensemble
2. **Text Length Effect**: Performance across different text lengths
3. **Calibration Impact**: With vs without score calibration
4. **Weight Sensitivity**: Effect of ensemble weight ratios

## 📋 Example Output

```
==================================================
PERSONALITY ANALYSIS RESULTS
==================================================

OPENNESS
  Score: 0.723 | Percentile: 78.5 | Category: High
  Evidence:
    - "exploring new ideas and meeting different people..."
    - "researching ancient philosophy..."
  Justification: Text shows curiosity and intellectual interests

CONSCIENTIOUSNESS
  Score: 0.681 | Percentile: 65.2 | Category: Medium
  Evidence:
    - "quite organized with my work..."
    - "always make detailed plans..."
  Justification: Demonstrates planning and organization

EXTRAVERSION
  Score: 0.612 | Percentile: 55.8 | Category: Medium
  Evidence:
    - "enjoy social gatherings..."
  Justification: Moderate social engagement mentioned

AGREEABLENESS
  Score: 0.745 | Percentile: 82.1 | Category: High
  Evidence:
    - "always ready to help when needed..."
    - "easy to talk to..."
  Justification: Shows helpfulness and social consideration

NEUROTICISM
  Score: 0.421 | Percentile: 38.4 | Category: Medium
  Evidence:
    - "Sometimes I feel anxious about upcoming deadlines..."
  Justification: Occasional anxiety but generally stable
```

## ⚠️ Limitations

1. **Text Dependency**: Predictions are only as good as the text input
2. **Cultural Bias**: Model trained primarily on English, Western data
3. **Context Missing**: Cannot capture situational personality variations
4. **Self-Report Bias**: Training data may reflect self-presentation
5. **Short Text**: Performance degrades with very short inputs (<50 words)

## 🔒 Ethical Considerations

### Responsible Use

- **Consent**: Only analyze text with proper consent
- **Privacy**: Do not store or share analyzed texts
- **Decisions**: Never use for high-stakes decisions (hiring, etc.)
- **Transparency**: Disclose when personality analysis is used

### Bias Awareness

- Training data may not represent all populations
- Results should be interpreted with cultural context
- System should not reinforce stereotypes

### Recommended Applications

✅ Self-reflection and personal development
✅ Educational and research purposes
✅ Content personalization (with consent)
✅ Team dynamics understanding

### Not Recommended For

❌ Employment screening
❌ Clinical diagnosis
❌ Legal proceedings
❌ Surveillance

## 📚 References

1. Goldberg, L. R. (1990). An alternative "description of personality": The Big-Five factor structure. *Journal of Personality and Social Psychology*, 59(6), 1216.

2. Pennebaker, J. W., & King, L. A. (1999). Linguistic styles: Language use as an individual difference. *Journal of Personality and Social Psychology*, 77(6), 1296.

3. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP*.

4. Majumder, N., et al. (2017). Deep learning-based document modeling for personality detection from text. *IEEE Intelligent Systems*, 32(2), 74-79.

## 📄 License

This project is for educational and research purposes. See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## 📧 Contact

For questions or collaboration, please open an issue on the repository.

---

**Note**: This is a research implementation. Performance will vary based on the quality and quantity of training data, as well as the specific domain of texts being analyzed.
