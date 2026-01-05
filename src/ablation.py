"""
Ablation Studies Module
=======================

This module implements ablation experiments to understand model behavior:
1. ML-only vs LLM-only vs Ensemble comparison
2. Effect of text length on prediction quality
3. Effect of removing calibration
4. Per-trait analysis
5. Cross-validation studies
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from .evaluation import PersonalityEvaluator, EvaluationResults, evaluate_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OCEAN traits
OCEAN_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


@dataclass
class AblationConfig:
    """Configuration for ablation studies."""
    text_length_bins: List[int] = field(default_factory=lambda: [100, 250, 500, 1000, 2000])
    output_dir: str = "results/ablation"
    random_seed: int = 42


@dataclass
class AblationResult:
    """Container for ablation study results."""
    study_name: str
    conditions: List[str]
    results: Dict[str, Dict]  # condition -> metrics
    summary: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "study_name": self.study_name,
            "conditions": self.conditions,
            "results": self.results,
            "summary": self.summary
        }


class AblationStudies:
    """
    Conducts ablation experiments for personality prediction.
    
    Studies:
    1. Model comparison (ML vs LLM vs Ensemble)
    2. Text length analysis
    3. Calibration impact
    4. Component weight sensitivity
    """
    
    def __init__(self, config: AblationConfig = None):
        """
        Initialize ablation studies.
        
        Args:
            config: Ablation configuration
        """
        self.config = config or AblationConfig()
        self.evaluator = PersonalityEvaluator(self.config.output_dir)
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        self.results: List[AblationResult] = []
    
    def study_model_comparison(
        self,
        ml_predictions: Dict[str, np.ndarray],
        llm_predictions: Dict[str, np.ndarray],
        ensemble_predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray]
    ) -> AblationResult:
        """
        Compare ML-only, LLM-only, and Ensemble performance.
        
        Args:
            ml_predictions: ML model predictions
            llm_predictions: LLM predictions
            ensemble_predictions: Ensemble predictions
            targets: Ground truth values
            
        Returns:
            AblationResult with comparison metrics
        """
        logger.info("Running model comparison ablation study...")
        
        # Evaluate each model
        ml_results = self.evaluator.evaluate(ml_predictions, targets, "ML Only")
        llm_results = self.evaluator.evaluate(llm_predictions, targets, "LLM Only")
        ensemble_results = self.evaluator.evaluate(ensemble_predictions, targets, "Ensemble")
        
        # Compile results
        results = {
            "ml_only": {
                "avg_pearson": ml_results.average_metrics.pearson_correlation,
                "avg_mae": ml_results.average_metrics.mae,
                "avg_r2": ml_results.average_metrics.r2,
                "trait_metrics": {t: m.to_dict() for t, m in ml_results.trait_metrics.items()}
            },
            "llm_only": {
                "avg_pearson": llm_results.average_metrics.pearson_correlation,
                "avg_mae": llm_results.average_metrics.mae,
                "avg_r2": llm_results.average_metrics.r2,
                "trait_metrics": {t: m.to_dict() for t, m in llm_results.trait_metrics.items()}
            },
            "ensemble": {
                "avg_pearson": ensemble_results.average_metrics.pearson_correlation,
                "avg_mae": ensemble_results.average_metrics.mae,
                "avg_r2": ensemble_results.average_metrics.r2,
                "trait_metrics": {t: m.to_dict() for t, m in ensemble_results.trait_metrics.items()}
            }
        }
        
        # Generate summary
        summary_lines = [
            "MODEL COMPARISON ABLATION STUDY",
            "=" * 50,
            "",
            "Average Metrics:",
            f"  ML Only:   Pearson r = {results['ml_only']['avg_pearson']:.4f}, MAE = {results['ml_only']['avg_mae']:.4f}",
            f"  LLM Only:  Pearson r = {results['llm_only']['avg_pearson']:.4f}, MAE = {results['llm_only']['avg_mae']:.4f}",
            f"  Ensemble:  Pearson r = {results['ensemble']['avg_pearson']:.4f}, MAE = {results['ensemble']['avg_mae']:.4f}",
            "",
            "Improvement over best single model:",
        ]
        
        best_single = max(results['ml_only']['avg_pearson'], results['llm_only']['avg_pearson'])
        improvement = results['ensemble']['avg_pearson'] - best_single
        summary_lines.append(f"  Ensemble improvement: {improvement:+.4f} ({improvement/best_single*100:+.2f}%)")
        
        summary = "\n".join(summary_lines)
        
        # Create visualization
        self._plot_model_comparison(results)
        
        ablation_result = AblationResult(
            study_name="Model Comparison",
            conditions=["ml_only", "llm_only", "ensemble"],
            results=results,
            summary=summary
        )
        
        self.results.append(ablation_result)
        return ablation_result
    
    def study_text_length_effect(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        text_lengths: np.ndarray
    ) -> AblationResult:
        """
        Analyze effect of text length on prediction quality.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            text_lengths: Length of each text
            
        Returns:
            AblationResult with length-based analysis
        """
        logger.info("Running text length effect ablation study...")
        
        bins = self.config.text_length_bins + [float('inf')]
        bin_labels = [f"<{bins[0]}"]
        for i in range(len(bins) - 2):
            bin_labels.append(f"{bins[i]}-{bins[i+1]}")
        bin_labels.append(f">{bins[-2]}")
        
        results = {}
        
        for i, (lower, upper) in enumerate(zip([0] + bins[:-1], bins)):
            mask = (text_lengths >= lower) & (text_lengths < upper)
            n_samples = np.sum(mask)
            
            if n_samples < 10:
                continue
            
            bin_preds = {t: p[mask] for t, p in predictions.items()}
            bin_targets = {t: t_arr[mask] for t, t_arr in targets.items()}
            
            eval_results = self.evaluator.evaluate(bin_preds, bin_targets, f"Length {bin_labels[i]}")
            
            results[bin_labels[i]] = {
                "n_samples": int(n_samples),
                "avg_pearson": eval_results.average_metrics.pearson_correlation,
                "avg_mae": eval_results.average_metrics.mae,
                "avg_r2": eval_results.average_metrics.r2
            }
        
        # Generate summary
        summary_lines = [
            "TEXT LENGTH EFFECT ABLATION STUDY",
            "=" * 50,
            "",
            "Performance by text length:"
        ]
        
        for bin_label, metrics in results.items():
            summary_lines.append(
                f"  {bin_label}: n={metrics['n_samples']}, r={metrics['avg_pearson']:.4f}, MAE={metrics['avg_mae']:.4f}"
            )
        
        summary = "\n".join(summary_lines)
        
        # Create visualization
        self._plot_text_length_effect(results)
        
        ablation_result = AblationResult(
            study_name="Text Length Effect",
            conditions=list(results.keys()),
            results=results,
            summary=summary
        )
        
        self.results.append(ablation_result)
        return ablation_result
    
    def study_calibration_effect(
        self,
        uncalibrated_predictions: Dict[str, np.ndarray],
        calibrated_predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray]
    ) -> AblationResult:
        """
        Analyze effect of score calibration.
        
        Args:
            uncalibrated_predictions: Raw predictions
            calibrated_predictions: Calibrated predictions
            targets: Ground truth values
            
        Returns:
            AblationResult with calibration analysis
        """
        logger.info("Running calibration effect ablation study...")
        
        uncal_results = self.evaluator.evaluate(uncalibrated_predictions, targets, "Uncalibrated")
        cal_results = self.evaluator.evaluate(calibrated_predictions, targets, "Calibrated")
        
        results = {
            "uncalibrated": {
                "avg_pearson": uncal_results.average_metrics.pearson_correlation,
                "avg_mae": uncal_results.average_metrics.mae,
                "avg_r2": uncal_results.average_metrics.r2,
                "trait_metrics": {t: m.to_dict() for t, m in uncal_results.trait_metrics.items()}
            },
            "calibrated": {
                "avg_pearson": cal_results.average_metrics.pearson_correlation,
                "avg_mae": cal_results.average_metrics.mae,
                "avg_r2": cal_results.average_metrics.r2,
                "trait_metrics": {t: m.to_dict() for t, m in cal_results.trait_metrics.items()}
            }
        }
        
        # Generate summary
        mae_improvement = results['uncalibrated']['avg_mae'] - results['calibrated']['avg_mae']
        r_improvement = results['calibrated']['avg_pearson'] - results['uncalibrated']['avg_pearson']
        
        summary_lines = [
            "CALIBRATION EFFECT ABLATION STUDY",
            "=" * 50,
            "",
            "Performance comparison:",
            f"  Uncalibrated: Pearson r = {results['uncalibrated']['avg_pearson']:.4f}, MAE = {results['uncalibrated']['avg_mae']:.4f}",
            f"  Calibrated:   Pearson r = {results['calibrated']['avg_pearson']:.4f}, MAE = {results['calibrated']['avg_mae']:.4f}",
            "",
            f"Calibration improvement:",
            f"  MAE reduction: {mae_improvement:.4f}",
            f"  Pearson improvement: {r_improvement:+.4f}"
        ]
        
        summary = "\n".join(summary_lines)
        
        # Create visualization
        self._plot_calibration_effect(results)
        
        ablation_result = AblationResult(
            study_name="Calibration Effect",
            conditions=["uncalibrated", "calibrated"],
            results=results,
            summary=summary
        )
        
        self.results.append(ablation_result)
        return ablation_result
    
    def study_weight_sensitivity(
        self,
        ml_predictions: Dict[str, np.ndarray],
        llm_predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        weight_range: np.ndarray = None
    ) -> AblationResult:
        """
        Analyze sensitivity to ensemble weights.
        
        Args:
            ml_predictions: ML model predictions
            llm_predictions: LLM predictions
            targets: Ground truth values
            weight_range: Range of ML weights to test
            
        Returns:
            AblationResult with weight sensitivity analysis
        """
        logger.info("Running weight sensitivity ablation study...")
        
        if weight_range is None:
            weight_range = np.arange(0, 1.05, 0.1)
        
        results = {}
        
        for ml_weight in weight_range:
            llm_weight = 1.0 - ml_weight
            
            # Combine predictions
            combined = {}
            for trait in OCEAN_TRAITS:
                if trait in ml_predictions and trait in llm_predictions:
                    combined[trait] = ml_weight * ml_predictions[trait] + llm_weight * llm_predictions[trait]
            
            eval_results = self.evaluator.evaluate(combined, targets, f"w={ml_weight:.1f}")
            
            results[f"ml_{ml_weight:.1f}"] = {
                "ml_weight": float(ml_weight),
                "llm_weight": float(llm_weight),
                "avg_pearson": eval_results.average_metrics.pearson_correlation,
                "avg_mae": eval_results.average_metrics.mae,
                "avg_r2": eval_results.average_metrics.r2
            }
        
        # Find optimal weight
        optimal = max(results.items(), key=lambda x: x[1]['avg_pearson'])
        
        summary_lines = [
            "WEIGHT SENSITIVITY ABLATION STUDY",
            "=" * 50,
            "",
            "Performance by ML weight (LLM weight = 1 - ML weight):",
        ]
        
        for key, metrics in sorted(results.items(), key=lambda x: x[1]['ml_weight']):
            summary_lines.append(
                f"  ML={metrics['ml_weight']:.1f}: r={metrics['avg_pearson']:.4f}, MAE={metrics['avg_mae']:.4f}"
            )
        
        summary_lines.extend([
            "",
            f"Optimal ML weight: {optimal[1]['ml_weight']:.1f}",
            f"  Pearson r: {optimal[1]['avg_pearson']:.4f}",
            f"  MAE: {optimal[1]['avg_mae']:.4f}"
        ])
        
        summary = "\n".join(summary_lines)
        
        # Create visualization
        self._plot_weight_sensitivity(results)
        
        ablation_result = AblationResult(
            study_name="Weight Sensitivity",
            conditions=list(results.keys()),
            results=results,
            summary=summary
        )
        
        self.results.append(ablation_result)
        return ablation_result
    
    def _plot_model_comparison(self, results: Dict):
        """Create model comparison visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pearson correlation comparison
        models = ['ML Only', 'LLM Only', 'Ensemble']
        pearson_values = [
            results['ml_only']['avg_pearson'],
            results['llm_only']['avg_pearson'],
            results['ensemble']['avg_pearson']
        ]
        
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        bars = axes[0].bar(models, pearson_values, color=colors, edgecolor='black')
        axes[0].set_ylabel('Pearson Correlation')
        axes[0].set_title('Model Comparison - Pearson r')
        axes[0].set_ylim(0, max(pearson_values) * 1.2)
        
        # Add value labels
        for bar, val in zip(bars, pearson_values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom')
        
        # MAE comparison
        mae_values = [
            results['ml_only']['avg_mae'],
            results['llm_only']['avg_mae'],
            results['ensemble']['avg_mae']
        ]
        
        bars = axes[1].bar(models, mae_values, color=colors, edgecolor='black')
        axes[1].set_ylabel('Mean Absolute Error')
        axes[1].set_title('Model Comparison - MAE (lower is better)')
        
        for bar, val in zip(bars, mae_values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'model_comparison.png'), dpi=300)
        plt.close()
    
    def _plot_text_length_effect(self, results: Dict):
        """Create text length effect visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        bins = list(results.keys())
        pearson_values = [results[b]['avg_pearson'] for b in bins]
        mae_values = [results[b]['avg_mae'] for b in bins]
        n_samples = [results[b]['n_samples'] for b in bins]
        
        # Pearson by length
        ax1 = axes[0]
        ax1.bar(bins, pearson_values, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Text Length Range')
        ax1.set_ylabel('Pearson Correlation')
        ax1.set_title('Prediction Quality vs Text Length')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add sample counts
        ax1_twin = ax1.twinx()
        ax1_twin.plot(bins, n_samples, 'ro-', label='Sample Count')
        ax1_twin.set_ylabel('Number of Samples', color='red')
        
        # MAE by length
        axes[1].bar(bins, mae_values, color='coral', edgecolor='black')
        axes[1].set_xlabel('Text Length Range')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Prediction Error vs Text Length')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'text_length_effect.png'), dpi=300)
        plt.close()
    
    def _plot_calibration_effect(self, results: Dict):
        """Create calibration effect visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        conditions = ['Uncalibrated', 'Calibrated']
        
        # Pearson comparison
        pearson_values = [results['uncalibrated']['avg_pearson'], results['calibrated']['avg_pearson']]
        axes[0].bar(conditions, pearson_values, color=['#e74c3c', '#2ecc71'], edgecolor='black')
        axes[0].set_ylabel('Pearson Correlation')
        axes[0].set_title('Calibration Effect - Pearson r')
        
        # MAE comparison
        mae_values = [results['uncalibrated']['avg_mae'], results['calibrated']['avg_mae']]
        axes[1].bar(conditions, mae_values, color=['#e74c3c', '#2ecc71'], edgecolor='black')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Calibration Effect - MAE')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'calibration_effect.png'), dpi=300)
        plt.close()
    
    def _plot_weight_sensitivity(self, results: Dict):
        """Create weight sensitivity visualization."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ml_weights = [results[k]['ml_weight'] for k in sorted(results.keys(), key=lambda x: results[x]['ml_weight'])]
        pearson_values = [results[k]['avg_pearson'] for k in sorted(results.keys(), key=lambda x: results[x]['ml_weight'])]
        mae_values = [results[k]['avg_mae'] for k in sorted(results.keys(), key=lambda x: results[x]['ml_weight'])]
        
        ax.plot(ml_weights, pearson_values, 'b-o', label='Pearson r', linewidth=2)
        ax.set_xlabel('ML Weight')
        ax.set_ylabel('Pearson Correlation', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        ax_twin = ax.twinx()
        ax_twin.plot(ml_weights, mae_values, 'r-s', label='MAE', linewidth=2)
        ax_twin.set_ylabel('MAE', color='red')
        ax_twin.tick_params(axis='y', labelcolor='red')
        
        ax.axhline(y=max(pearson_values), color='blue', linestyle='--', alpha=0.5)
        
        ax.set_title('Ensemble Weight Sensitivity')
        ax.legend(loc='upper left')
        ax_twin.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'weight_sensitivity.png'), dpi=300)
        plt.close()
    
    def generate_report(self, save_path: str = None) -> str:
        """
        Generate comprehensive ablation study report.
        
        Args:
            save_path: Path to save report
            
        Returns:
            Report string
        """
        lines = [
            "=" * 70,
            "ABLATION STUDIES REPORT",
            "=" * 70,
            ""
        ]
        
        for result in self.results:
            lines.append(result.summary)
            lines.append("")
            lines.append("-" * 70)
            lines.append("")
        
        report = "\n".join(lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Saved ablation report to {save_path}")
        
        return report
    
    def save_results(self, filename: str = "ablation_results.json"):
        """Save all ablation results to JSON."""
        filepath = os.path.join(self.config.output_dir, filename)
        
        all_results = {
            result.study_name: result.to_dict() for result in self.results
        }
        
        with open(filepath, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Saved ablation results to {filepath}")


def run_ablation_studies(
    ml_predictions: Dict[str, np.ndarray],
    llm_predictions: Dict[str, np.ndarray],
    ensemble_predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    text_lengths: np.ndarray = None,
    calibrated_predictions: Dict[str, np.ndarray] = None,
    output_dir: str = "results/ablation"
) -> AblationStudies:
    """
    Run all ablation studies.
    
    Args:
        ml_predictions: ML model predictions
        llm_predictions: LLM predictions
        ensemble_predictions: Ensemble predictions
        targets: Ground truth values
        text_lengths: Text lengths for length analysis
        calibrated_predictions: Calibrated predictions for calibration analysis
        output_dir: Output directory
        
    Returns:
        AblationStudies object with all results
    """
    config = AblationConfig(output_dir=output_dir)
    ablation = AblationStudies(config)
    
    # Model comparison
    ablation.study_model_comparison(ml_predictions, llm_predictions, ensemble_predictions, targets)
    
    # Text length effect
    if text_lengths is not None:
        ablation.study_text_length_effect(ensemble_predictions, targets, text_lengths)
    
    # Calibration effect
    if calibrated_predictions is not None:
        ablation.study_calibration_effect(ensemble_predictions, calibrated_predictions, targets)
    
    # Weight sensitivity
    ablation.study_weight_sensitivity(ml_predictions, llm_predictions, targets)
    
    # Generate report
    ablation.generate_report(os.path.join(output_dir, "ablation_report.txt"))
    ablation.save_results()
    
    return ablation


if __name__ == "__main__":
    # Test ablation studies
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 500
    targets = {trait: np.random.rand(n_samples) for trait in OCEAN_TRAITS}
    
    # Simulate different models
    ml_predictions = {trait: targets[trait] + np.random.randn(n_samples) * 0.12 
                     for trait in OCEAN_TRAITS}
    ml_predictions = {trait: np.clip(p, 0, 1) for trait, p in ml_predictions.items()}
    
    llm_predictions = {trait: targets[trait] + np.random.randn(n_samples) * 0.15 
                      for trait in OCEAN_TRAITS}
    llm_predictions = {trait: np.clip(p, 0, 1) for trait, p in llm_predictions.items()}
    
    ensemble_predictions = {trait: 0.6 * ml_predictions[trait] + 0.4 * llm_predictions[trait]
                           for trait in OCEAN_TRAITS}
    
    text_lengths = np.random.exponential(300, n_samples).astype(int)
    
    # Run ablation studies
    ablation = run_ablation_studies(
        ml_predictions, llm_predictions, ensemble_predictions,
        targets, text_lengths
    )
    
    print(ablation.generate_report())
