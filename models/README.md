# Models Directory

This directory contains trained models saved during training.

## Contents

After training, you'll find:

- `ml_model.pkl` - Trained ML baseline model (embeddings + regressors)
- `ensemble.pkl` - Trained ensemble model (weights + calibrators)
- `config.json` - Pipeline configuration

## Loading Models

```python
from src.pipeline import PersonalityPredictor

predictor = PersonalityPredictor()
predictor.load("models")

# Now ready for inference
prediction = predictor.predict("Your text here")
```

## Model Size

Typical model sizes:
- ML model: ~50-100 MB (depends on embedding model)
- Ensemble: ~1 MB

Note: The sentence transformer model will be downloaded on first use (~90 MB for MiniLM-L6-v2).
