"""
Evaluation Framework Module
===========================

This module implements comprehensive evaluation for personality prediction:
1. Regression metrics (Pearson correlation, MAE, R², RMSE)
2. Cross-validation
3. Statistical significance testing
4. Visualization
5. Per-trait and aggregate analysis
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OCEAN traits
OCEAN_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    pearson_correlation: float
    pearson_p_value: float
    spearman_correlation: float
    spearman_p_value: float
    mae: float
    r2: float
    rmse: float
    mse: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "pearson_correlation": self.pearson_correlation,
            "pearson_p_value": self.pearson_p_value,
            "spearman_correlation": self.spearman_correlation,
            "spearman_p_value": self.spearman_p_value,
            "mae": self.mae,
            "r2": self.r2,
            "rmse": self.rmse,
            "mse": self.mse
        }


@dataclass
class EvaluationResults:
    """Complete evaluation results."""
    trait_metrics: Dict[str, EvaluationMetrics]
    average_metrics: EvaluationMetrics
    n_samples: int
    model_name: str = ""
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "n_samples": self.n_samples,
            "trait_metrics": {
                trait: metrics.to_dict() 
                for trait, metrics in self.trait_metrics.items()
            },
            "average_metrics": self.average_metrics.to_dict(),
            "metadata": self.metadata
        }
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"\n{'='*60}",
            f"EVALUATION RESULTS: {self.model_name}",
            f"{'='*60}",
            f"Number of samples: {self.n_samples}",
            "",
            "Per-Trait Metrics:",
            "-" * 50
        ]
        
        for trait in OCEAN_TRAITS:
            if trait in self.trait_metrics:
                m = self.trait_metrics[trait]
                lines.append(f"\n{trait.upper()}")
                lines.append(f"  Pearson r: {m.pearson_correlation:.4f} (p={m.pearson_p_value:.4f})")
                lines.append(f"  Spearman ρ: {m.spearman_correlation:.4f}")
                lines.append(f"  MAE: {m.mae:.4f}")
                lines.append(f"  R²: {m.r2:.4f}")
                lines.append(f"  RMSE: {m.rmse:.4f}")
        
        lines.extend([
            "",
            "-" * 50,
            "AVERAGE METRICS:",
            f"  Pearson r: {self.average_metrics.pearson_correlation:.4f}",
            f"  MAE: {self.average_metrics.mae:.4f}",
            f"  R²: {self.average_metrics.r2:.4f}",
            f"  RMSE: {self.average_metrics.rmse:.4f}",
            "=" * 60
        ])
        
        return "\n".join(lines)


class PersonalityEvaluator:
    """
    Comprehensive evaluator for personality prediction models.
    
    Features:
    - Multiple regression metrics
    - Statistical significance testing
    - Cross-validation
    - Visualization
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize the evaluator.
        
        Args:
            output_dir: Directory for saving results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def compute_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> EvaluationMetrics:
        """
        Compute all evaluation metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            EvaluationMetrics object
        """
        # Handle edge cases
        if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
            return EvaluationMetrics(
                pearson_correlation=0.0,
                pearson_p_value=1.0,
                spearman_correlation=0.0,
                spearman_p_value=1.0,
                mae=float(np.mean(np.abs(y_true - y_pred))),
                r2=0.0,
                rmse=float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                mse=float(np.mean((y_true - y_pred) ** 2))
            )
        
        # Correlation metrics
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        spearman_r, spearman_p = spearmanr(y_true, y_pred)
        
        # Regression metrics
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        return EvaluationMetrics(
            pearson_correlation=float(pearson_r),
            pearson_p_value=float(pearson_p),
            spearman_correlation=float(spearman_r),
            spearman_p_value=float(spearman_p),
            mae=float(mae),
            r2=float(r2),
            rmse=float(rmse),
            mse=float(mse)
        )
    
    def evaluate(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        model_name: str = "Model"
    ) -> EvaluationResults:
        """
        Evaluate predictions against targets.
        
        Args:
            predictions: Predicted values per trait
            targets: Ground truth values per trait
            model_name: Name of the model
            
        Returns:
            EvaluationResults object
        """
        logger.info(f"Evaluating {model_name}...")
        
        trait_metrics = {}
        n_samples = 0
        
        for trait in OCEAN_TRAITS:
            if trait in predictions and trait in targets:
                y_pred = predictions[trait]
                y_true = targets[trait]
                n_samples = len(y_true)
                
                metrics = self.compute_metrics(y_true, y_pred)
                trait_metrics[trait] = metrics
        
        # Calculate average metrics
        avg_pearson = np.mean([m.pearson_correlation for m in trait_metrics.values()])
        avg_spearman = np.mean([m.spearman_correlation for m in trait_metrics.values()])
        avg_mae = np.mean([m.mae for m in trait_metrics.values()])
        avg_r2 = np.mean([m.r2 for m in trait_metrics.values()])
        avg_rmse = np.mean([m.rmse for m in trait_metrics.values()])
        avg_mse = np.mean([m.mse for m in trait_metrics.values()])
        
        average_metrics = EvaluationMetrics(
            pearson_correlation=float(avg_pearson),
            pearson_p_value=0.0,  # Not applicable for average
            spearman_correlation=float(avg_spearman),
            spearman_p_value=0.0,
            mae=float(avg_mae),
            r2=float(avg_r2),
            rmse=float(avg_rmse),
            mse=float(avg_mse)
        )
        
        return EvaluationResults(
            trait_metrics=trait_metrics,
            average_metrics=average_metrics,
            n_samples=n_samples,
            model_name=model_name
        )
    
    def compare_models(
        self,
        results_list: List[EvaluationResults]
    ) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            results_list: List of EvaluationResults
            
        Returns:
            Comparison DataFrame
        """
        data = []
        
        for results in results_list:
            row = {"Model": results.model_name}
            
            # Add per-trait metrics
            for trait in OCEAN_TRAITS:
                if trait in results.trait_metrics:
                    m = results.trait_metrics[trait]
                    row[f"{trait}_pearson"] = m.pearson_correlation
                    row[f"{trait}_mae"] = m.mae
                    row[f"{trait}_r2"] = m.r2
            
            # Add average metrics
            row["avg_pearson"] = results.average_metrics.pearson_correlation
            row["avg_mae"] = results.average_metrics.mae
            row["avg_r2"] = results.average_metrics.r2
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def statistical_significance_test(
        self,
        predictions1: Dict[str, np.ndarray],
        predictions2: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray]
    ) -> Dict[str, Dict]:
        """
        Test statistical significance between two models.
        
        Uses paired t-test and Wilcoxon signed-rank test.
        
        Args:
            predictions1: First model predictions
            predictions2: Second model predictions
            targets: Ground truth values
            
        Returns:
            Dictionary of test results per trait
        """
        results = {}
        
        for trait in OCEAN_TRAITS:
            if trait in predictions1 and trait in predictions2 and trait in targets:
                errors1 = np.abs(predictions1[trait] - targets[trait])
                errors2 = np.abs(predictions2[trait] - targets[trait])
                
                # Paired t-test
                t_stat, t_pvalue = ttest_rel(errors1, errors2)
                
                # Wilcoxon test (non-parametric)
                try:
                    w_stat, w_pvalue = wilcoxon(errors1, errors2)
                except ValueError:
                    w_stat, w_pvalue = 0.0, 1.0
                
                results[trait] = {
                    "t_statistic": float(t_stat),
                    "t_pvalue": float(t_pvalue),
                    "wilcoxon_statistic": float(w_stat),
                    "wilcoxon_pvalue": float(w_pvalue),
                    "mean_error1": float(np.mean(errors1)),
                    "mean_error2": float(np.mean(errors2)),
                    "significant_t": t_pvalue < 0.05,
                    "significant_wilcoxon": w_pvalue < 0.05
                }
        
        return results
    
    def plot_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        model_name: str = "Model",
        save_path: str = None
    ):
        """
        Plot predicted vs actual values.
        
        Args:
            predictions: Predicted values per trait
            targets: Ground truth values per trait
            model_name: Name of the model
            save_path: Path to save the figure
        """
        n_traits = len([t for t in OCEAN_TRAITS if t in predictions])
        fig, axes = plt.subplots(1, n_traits, figsize=(4 * n_traits, 4))
        
        if n_traits == 1:
            axes = [axes]
        
        for idx, trait in enumerate(OCEAN_TRAITS):
            if trait not in predictions:
                continue
            
            ax = axes[idx]
            y_true = targets[trait]
            y_pred = predictions[trait]
            
            # Scatter plot
            ax.scatter(y_true, y_pred, alpha=0.5, s=10)
            
            # Perfect prediction line
            ax.plot([0, 1], [0, 1], 'r--', label='Perfect')
            
            # Regression line
            z = np.polyfit(y_true, y_pred, 1)
            p = np.poly1d(z)
            ax.plot([0, 1], [p(0), p(1)], 'b-', alpha=0.7, label='Regression')
            
            # Metrics
            metrics = self.compute_metrics(y_true, y_pred)
            ax.text(0.05, 0.95, f'r={metrics.pearson_correlation:.3f}\nMAE={metrics.mae:.3f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(trait.capitalize())
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(loc='lower right')
        
        plt.suptitle(f'{model_name} - Predicted vs Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved prediction plot to {save_path}")
        
        plt.close()
    
    def plot_error_distribution(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        model_name: str = "Model",
        save_path: str = None
    ):
        """
        Plot error distribution per trait.
        
        Args:
            predictions: Predicted values per trait
            targets: Ground truth values per trait
            model_name: Name of the model
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        
        for idx, trait in enumerate(OCEAN_TRAITS):
            if trait not in predictions:
                continue
            
            ax = axes[idx]
            errors = predictions[trait] - targets[trait]
            
            ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--')
            ax.axvline(x=np.mean(errors), color='g', linestyle='-', label=f'Mean={np.mean(errors):.3f}')
            
            ax.set_xlabel('Prediction Error')
            ax.set_ylabel('Frequency')
            ax.set_title(trait.capitalize())
            ax.legend()
        
        # Remove extra subplot
        axes[-1].axis('off')
        
        plt.suptitle(f'{model_name} - Error Distribution')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved error distribution plot to {save_path}")
        
        plt.close()
    
    def plot_comparison(
        self,
        comparison_df: pd.DataFrame,
        metric: str = "avg_pearson",
        save_path: str = None
    ):
        """
        Plot model comparison.
        
        Args:
            comparison_df: DataFrame from compare_models
            metric: Metric to compare
            save_path: Path to save the figure
        """
        # Per-trait comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Trait-level metrics
        trait_cols = [f"{t}_pearson" for t in OCEAN_TRAITS if f"{t}_pearson" in comparison_df.columns]
        if trait_cols:
            plot_data = comparison_df[["Model"] + trait_cols].melt(
                id_vars="Model", var_name="Trait", value_name="Pearson r"
            )
            plot_data["Trait"] = plot_data["Trait"].str.replace("_pearson", "").str.capitalize()
            
            sns.barplot(data=plot_data, x="Trait", y="Pearson r", hue="Model", ax=axes[0])
            axes[0].set_title("Pearson Correlation by Trait")
            axes[0].set_ylim(0, 1)
            axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # Average metrics comparison
        avg_metrics = ["avg_pearson", "avg_mae", "avg_r2"]
        available_metrics = [m for m in avg_metrics if m in comparison_df.columns]
        if available_metrics:
            plot_data = comparison_df[["Model"] + available_metrics].melt(
                id_vars="Model", var_name="Metric", value_name="Value"
            )
            plot_data["Metric"] = plot_data["Metric"].str.replace("avg_", "").str.upper()
            
            sns.barplot(data=plot_data, x="Metric", y="Value", hue="Model", ax=axes[1])
            axes[1].set_title("Average Metrics Comparison")
            axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {save_path}")
        
        plt.close()
    
    def generate_report(
        self,
        results_list: List[EvaluationResults],
        save_path: str = None
    ) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            results_list: List of evaluation results
            save_path: Path to save the report
            
        Returns:
            Report string
        """
        lines = [
            "=" * 70,
            "PERSONALITY DETECTION SYSTEM - EVALUATION REPORT",
            "=" * 70,
            ""
        ]
        
        # Model summaries
        for results in results_list:
            lines.append(results.summary())
            lines.append("")
        
        # Comparison table
        if len(results_list) > 1:
            comparison_df = self.compare_models(results_list)
            lines.extend([
                "",
                "MODEL COMPARISON",
                "-" * 50,
                comparison_df.to_string(index=False),
                ""
            ])
        
        report = "\n".join(lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Saved evaluation report to {save_path}")
        
        return report
    
    def save_results(
        self,
        results: EvaluationResults,
        filename: str = "evaluation_results.json"
    ):
        """
        Save evaluation results to JSON.
        
        Args:
            results: EvaluationResults object
            filename: Output filename
        """
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        logger.info(f"Saved results to {filepath}")


def evaluate_model(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    model_name: str = "Model",
    output_dir: str = "results",
    generate_plots: bool = True
) -> EvaluationResults:
    """
    Convenience function to evaluate a model.
    
    Args:
        predictions: Predicted values per trait
        targets: Ground truth values per trait
        model_name: Name of the model
        output_dir: Output directory
        generate_plots: Whether to generate plots
        
    Returns:
        EvaluationResults object
    """
    evaluator = PersonalityEvaluator(output_dir)
    results = evaluator.evaluate(predictions, targets, model_name)
    
    if generate_plots:
        evaluator.plot_predictions(
            predictions, targets, model_name,
            os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_predictions.png")
        )
        evaluator.plot_error_distribution(
            predictions, targets, model_name,
            os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_errors.png")
        )
    
    evaluator.save_results(results, f"{model_name.lower().replace(' ', '_')}_results.json")
    
    return results


if __name__ == "__main__":
    # Test evaluation
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 500
    targets = {trait: np.random.rand(n_samples) for trait in OCEAN_TRAITS}
    
    # Model 1: Good predictions
    predictions1 = {trait: targets[trait] + np.random.randn(n_samples) * 0.1 
                   for trait in OCEAN_TRAITS}
    predictions1 = {trait: np.clip(p, 0, 1) for trait, p in predictions1.items()}
    
    # Model 2: Worse predictions
    predictions2 = {trait: targets[trait] + np.random.randn(n_samples) * 0.2 
                   for trait in OCEAN_TRAITS}
    predictions2 = {trait: np.clip(p, 0, 1) for trait, p in predictions2.items()}
    
    # Evaluate
    evaluator = PersonalityEvaluator("results")
    
    results1 = evaluator.evaluate(predictions1, targets, "ML Baseline")
    results2 = evaluator.evaluate(predictions2, targets, "LLM Only")
    
    print(results1.summary())
    print(results2.summary())
    
    # Compare
    comparison = evaluator.compare_models([results1, results2])
    print("\nComparison:")
    print(comparison.to_string())
    
    # Statistical significance
    sig_tests = evaluator.statistical_significance_test(predictions1, predictions2, targets)
    print("\nStatistical Significance (Model 1 vs Model 2):")
    for trait, test in sig_tests.items():
        print(f"  {trait}: t-test p={test['t_pvalue']:.4f}, significant={test['significant_t']}")
