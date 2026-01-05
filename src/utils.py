"""
Utility Functions
=================

Common utility functions used across the personality detection system.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

import numpy as np
import yaml


def load_config(config_path: str = "config/settings.yaml") -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_json(data: Any, filepath: str, indent: int = 2):
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(filepath: str) -> Any:
    """
    Load data from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_str: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
):
    """
    Set up logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        format_str: Log format string
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=handlers
    )


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def create_timestamp() -> str:
    """Create timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                           np.int16, np.int32, np.int64, np.uint8,
                           np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def validate_ocean_scores(scores: Dict[str, float]) -> bool:
    """
    Validate OCEAN scores are in valid range.
    
    Args:
        scores: Dictionary of trait scores
        
    Returns:
        True if valid, False otherwise
    """
    OCEAN_TRAITS = ["openness", "conscientiousness", "extraversion", 
                    "agreeableness", "neuroticism"]
    
    for trait in OCEAN_TRAITS:
        if trait in scores:
            score = scores[trait]
            if not isinstance(score, (int, float)):
                return False
            if score < 0 or score > 1:
                return False
    
    return True


def scores_to_categories(
    scores: Dict[str, float],
    low_threshold: float = 0.33,
    high_threshold: float = 0.67
) -> Dict[str, str]:
    """
    Convert continuous scores to category labels.
    
    Args:
        scores: Dictionary of trait scores
        low_threshold: Threshold for "Low" category
        high_threshold: Threshold for "High" category
        
    Returns:
        Dictionary of category labels
    """
    categories = {}
    for trait, score in scores.items():
        if score < low_threshold:
            categories[trait] = "Low"
        elif score > high_threshold:
            categories[trait] = "High"
        else:
            categories[trait] = "Medium"
    return categories


def print_prediction_summary(prediction: Dict, include_evidence: bool = True):
    """
    Print a formatted prediction summary.
    
    Args:
        prediction: Prediction dictionary
        include_evidence: Whether to include evidence
    """
    OCEAN_TRAITS = ["openness", "conscientiousness", "extraversion", 
                    "agreeableness", "neuroticism"]
    
    print("\n" + "=" * 60)
    print("PERSONALITY ANALYSIS RESULTS")
    print("=" * 60)
    
    traits = prediction.get("traits", prediction.get("scores", {}))
    
    for trait in OCEAN_TRAITS:
        if trait in traits:
            data = traits[trait]
            
            if isinstance(data, dict):
                score = data.get("score", 0)
                percentile = data.get("percentile", 50)
                category = data.get("category", "Medium")
                evidence = data.get("evidence", [])
                justification = data.get("justification", "")
            else:
                score = data
                percentile = 50
                category = "Medium"
                evidence = []
                justification = ""
            
            print(f"\n{trait.upper()}")
            print(f"  Score: {score:.3f} | Percentile: {percentile:.1f} | Category: {category}")
            
            if include_evidence and evidence:
                print("  Evidence:")
                for ev in evidence[:2]:
                    ev_short = ev[:80] + "..." if len(ev) > 80 else ev
                    print(f"    - \"{ev_short}\"")
            
            if include_evidence and justification:
                print(f"  Justification: {justification}")
    
    print("\n" + "=" * 60)
