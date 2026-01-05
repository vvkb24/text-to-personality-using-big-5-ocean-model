# Results Directory

This directory contains evaluation results, plots, and reports generated during training and evaluation.

## Contents

After running the training script, you'll find:

- `data_analysis.json` - Dataset statistics and distribution analysis
- `training_report.txt` - Comprehensive training report
- `*_predictions.png` - Scatter plots of predicted vs actual values
- `*_errors.png` - Error distribution histograms
- `model_comparison.png` - Comparison across models

## Ablation Studies

If ablation studies are enabled (`--run-ablation`), you'll also find:

- `ablation/model_comparison.png` - ML vs LLM vs Ensemble
- `ablation/text_length_effect.png` - Performance by text length
- `ablation/calibration_effect.png` - Calibration impact
- `ablation/weight_sensitivity.png` - Ensemble weight analysis
- `ablation/ablation_report.txt` - Complete ablation report
