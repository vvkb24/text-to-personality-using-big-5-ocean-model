"""
Data Loading and Preprocessing Module
=====================================

This module handles:
1. Loading personality datasets from HuggingFace and other sources
2. Data cleaning and normalization
3. Text preprocessing
4. Label distribution analysis
5. Train/validation/test splitting

Datasets Used:
- HuggingFace: Fatima0923/Automated-Personality-Prediction
- Essays Dataset (Big Five, Pennebaker et al.)
"""

import os
import re
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, DatasetDict
import nltk
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OCEAN trait names
OCEAN_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

# Alternative column name mappings for different datasets
COLUMN_MAPPINGS = {
    # Standard mappings
    "text": ["text", "TEXT", "essay", "Essay", "content", "Content", "post", "status"],
    "openness": ["openness", "Openness", "OPN", "opn", "O", "cOPN", "open"],
    "conscientiousness": ["conscientiousness", "Conscientiousness", "CON", "con", "C", "cCON", "consc"],
    "extraversion": ["extraversion", "Extraversion", "EXT", "ext", "E", "cEXT", "extra"],
    "agreeableness": ["agreeableness", "Agreeableness", "AGR", "agr", "A", "cAGR", "agree"],
    "neuroticism": ["neuroticism", "Neuroticism", "NEU", "neu", "N", "cNEU", "neuro"],
}


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    min_text_length: int = 50
    max_text_length: int = 2048
    remove_urls: bool = True
    remove_special_chars: bool = False
    lowercase: bool = False
    remove_extra_whitespace: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    min_samples: int = 10000


@dataclass
class PersonalityDataset:
    """Container for personality dataset with OCEAN labels."""
    texts: List[str]
    labels: Dict[str, np.ndarray]  # trait_name -> scores
    metadata: Dict = field(default_factory=dict)
    
    def __len__(self):
        return len(self.texts)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        data = {"text": self.texts}
        for trait in OCEAN_TRAITS:
            if trait in self.labels:
                data[trait] = self.labels[trait]
        return pd.DataFrame(data)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, text_col: str = "text") -> "PersonalityDataset":
        """Create from pandas DataFrame."""
        texts = df[text_col].tolist()
        labels = {}
        for trait in OCEAN_TRAITS:
            if trait in df.columns:
                labels[trait] = df[trait].values
        return cls(texts=texts, labels=labels)


class TextPreprocessor:
    """Text preprocessing for personality detection."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self._ensure_nltk_data()
    
    def _ensure_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.download('punkt_tab', quiet=True)
            except:
                pass
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        if self.config.remove_urls:
            text = re.sub(r'http[s]?://\S+', '', text)
            text = re.sub(r'www\.\S+', '', text)
        
        # Remove special characters (optional)
        if self.config.remove_special_chars:
            text = re.sub(r'[^\w\s\.\,\!\?\'\"]', '', text)
        
        # Remove extra whitespace
        if self.config.remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        # Lowercase (optional)
        if self.config.lowercase:
            text = text.lower()
        
        return text
    
    def is_valid_text(self, text: str) -> bool:
        """Check if text meets minimum requirements."""
        if not isinstance(text, str):
            return False
        text = text.strip()
        return len(text) >= self.config.min_text_length
    
    def truncate_text(self, text: str) -> str:
        """Truncate text to maximum length."""
        if len(text) > self.config.max_text_length:
            # Try to truncate at sentence boundary
            truncated = text[:self.config.max_text_length]
            last_period = truncated.rfind('.')
            if last_period > self.config.max_text_length * 0.7:
                return truncated[:last_period + 1]
            return truncated
        return text
    
    def preprocess(self, text: str) -> Optional[str]:
        """Full preprocessing pipeline."""
        text = self.clean_text(text)
        if not self.is_valid_text(text):
            return None
        return self.truncate_text(text)


class PersonalityDataLoader:
    """
    Data loader for personality datasets.
    
    Handles loading from multiple sources and standardizing format.
    """
    
    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self.preprocessor = TextPreprocessor(self.config)
    
    def _find_column(self, df: pd.DataFrame, target: str) -> Optional[str]:
        """Find column name from possible mappings."""
        possible_names = COLUMN_MAPPINGS.get(target, [target])
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to expected format."""
        column_map = {}
        
        # Find text column
        text_col = self._find_column(df, "text")
        if text_col and text_col != "text":
            column_map[text_col] = "text"
        
        # Find trait columns
        for trait in OCEAN_TRAITS:
            trait_col = self._find_column(df, trait)
            if trait_col and trait_col != trait:
                column_map[trait_col] = trait
        
        if column_map:
            df = df.rename(columns=column_map)
        
        return df
    
    def _normalize_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize trait labels to 0-1 range."""
        for trait in OCEAN_TRAITS:
            if trait in df.columns:
                values = df[trait].values
                min_val, max_val = values.min(), values.max()
                
                # Check if already normalized
                if min_val >= 0 and max_val <= 1:
                    continue
                
                # Check for binary labels (0/1 classification)
                unique_vals = np.unique(values[~np.isnan(values)])
                if len(unique_vals) == 2 and set(unique_vals) == {0, 1}:
                    # Binary labels - keep as is but note in logs
                    logger.info(f"Trait {trait} has binary labels")
                    continue
                
                # Normalize to 0-1
                if max_val > min_val:
                    df[trait] = (values - min_val) / (max_val - min_val)
                    logger.info(f"Normalized {trait} from [{min_val}, {max_val}] to [0, 1]")
        
        return df
    
    def load_huggingface_dataset(
        self, 
        dataset_name: str = "Fatima0923/Automated-Personality-Prediction"
    ) -> pd.DataFrame:
        """Load dataset from HuggingFace Hub."""
        logger.info(f"Loading dataset from HuggingFace: {dataset_name}")
        
        try:
            dataset = load_dataset(dataset_name)
            
            # Convert to DataFrame
            if isinstance(dataset, DatasetDict):
                # Combine all splits
                dfs = []
                for split_name, split_data in dataset.items():
                    df = split_data.to_pandas()
                    df['_split'] = split_name
                    dfs.append(df)
                df = pd.concat(dfs, ignore_index=True)
            else:
                df = dataset.to_pandas()
            
            logger.info(f"Loaded {len(df)} samples from HuggingFace")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace dataset: {e}")
            raise
    
    def load_csv_dataset(self, filepath: str) -> pd.DataFrame:
        """Load dataset from CSV file."""
        logger.info(f"Loading dataset from CSV: {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} samples from CSV")
        return df
    
    def create_synthetic_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Create a synthetic dataset for testing/demonstration.
        
        This generates realistic-looking personality data with text samples
        that correlate with trait scores.
        """
        logger.warning("Creating synthetic dataset for demonstration purposes")
        
        np.random.seed(self.config.random_seed)
        
        # Trait-related vocabulary
        trait_vocab = {
            "openness": {
                "high": ["I love exploring new ideas", "Art and creativity inspire me", 
                        "I'm always curious about different perspectives", "Abstract concepts fascinate me",
                        "I enjoy philosophical discussions", "New experiences excite me"],
                "low": ["I prefer practical solutions", "Traditional approaches work best",
                       "I stick to what I know", "Routine gives me comfort"]
            },
            "conscientiousness": {
                "high": ["I always plan ahead", "Organization is key to success",
                        "I complete tasks on time", "Details matter to me",
                        "I follow through on commitments", "I set clear goals"],
                "low": ["I go with the flow", "Deadlines are flexible",
                       "I prefer spontaneity", "Too much planning is restrictive"]
            },
            "extraversion": {
                "high": ["I love meeting new people", "Parties energize me",
                        "I enjoy being the center of attention", "Social activities are fun",
                        "I speak my mind freely", "Group activities are great"],
                "low": ["I prefer quiet evenings", "Small gatherings are better",
                       "I need alone time to recharge", "I listen more than I talk"]
            },
            "agreeableness": {
                "high": ["I care about others' feelings", "Helping people brings me joy",
                        "I believe in cooperation", "Empathy is important to me",
                        "I trust others easily", "Harmony matters in relationships"],
                "low": ["I speak my mind regardless", "Competition drives excellence",
                       "I question others' motives", "I prioritize my own needs"]
            },
            "neuroticism": {
                "high": ["I often feel anxious", "Stress affects me deeply",
                        "I worry about many things", "My mood changes frequently",
                        "I'm sensitive to criticism", "Uncertainty makes me nervous"],
                "low": ["I stay calm under pressure", "I rarely feel anxious",
                       "I handle stress well", "My emotions are stable"]
            }
        }
        
        data = []
        for _ in range(n_samples):
            # Generate trait scores with some correlation structure
            base = np.random.randn(5) * 0.2 + 0.5
            scores = np.clip(base + np.random.randn(5) * 0.15, 0.1, 0.9)
            
            # Generate text based on trait scores
            text_parts = []
            for i, trait in enumerate(OCEAN_TRAITS):
                score = scores[i]
                vocab = trait_vocab[trait]
                
                # Higher score = more high-trait sentences
                if score > 0.5:
                    n_high = int((score - 0.3) * 5)
                    n_low = int((0.7 - score) * 3)
                else:
                    n_high = int(score * 3)
                    n_low = int((0.7 - score) * 4)
                
                n_high = max(0, min(n_high, len(vocab["high"])))
                n_low = max(0, min(n_low, len(vocab["low"])))
                
                if n_high > 0:
                    text_parts.extend(np.random.choice(vocab["high"], n_high, replace=True))
                if n_low > 0:
                    text_parts.extend(np.random.choice(vocab["low"], n_low, replace=True))
            
            np.random.shuffle(text_parts)
            text = ". ".join(text_parts) + "."
            
            data.append({
                "text": text,
                "openness": scores[0],
                "conscientiousness": scores[1],
                "extraversion": scores[2],
                "agreeableness": scores[3],
                "neuroticism": scores[4]
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Created synthetic dataset with {len(df)} samples")
        return df
    
    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to dataset."""
        logger.info("Preprocessing dataset...")
        
        # Standardize columns
        df = self._standardize_columns(df)
        
        # Check required columns
        if "text" not in df.columns:
            raise ValueError("Dataset must have a 'text' column")
        
        available_traits = [t for t in OCEAN_TRAITS if t in df.columns]
        if not available_traits:
            raise ValueError("Dataset must have at least one OCEAN trait column")
        
        logger.info(f"Available traits: {available_traits}")
        
        # Preprocess text
        original_len = len(df)
        df["text"] = df["text"].apply(self.preprocessor.preprocess)
        df = df.dropna(subset=["text"])
        
        # Drop rows with missing trait values
        df = df.dropna(subset=available_traits)
        
        logger.info(f"Kept {len(df)}/{original_len} samples after preprocessing")
        
        # Normalize labels
        df = self._normalize_labels(df)
        
        return df
    
    def analyze_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze label distribution in dataset."""
        analysis = {
            "n_samples": len(df),
            "text_length_stats": {
                "mean": df["text"].str.len().mean(),
                "std": df["text"].str.len().std(),
                "min": df["text"].str.len().min(),
                "max": df["text"].str.len().max(),
                "median": df["text"].str.len().median()
            },
            "trait_stats": {}
        }
        
        for trait in OCEAN_TRAITS:
            if trait in df.columns:
                values = df[trait].values
                analysis["trait_stats"][trait] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                    "25th_percentile": float(np.percentile(values, 25)),
                    "75th_percentile": float(np.percentile(values, 75))
                }
        
        return analysis
    
    def split_dataset(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train/validation/test sets."""
        logger.info("Splitting dataset...")
        
        # First split: train + val vs test
        train_val_ratio = self.config.train_ratio + self.config.val_ratio
        train_val, test = train_test_split(
            df, 
            test_size=self.config.test_ratio,
            random_state=self.config.random_seed
        )
        
        # Second split: train vs val
        val_ratio_adjusted = self.config.val_ratio / train_val_ratio
        train, val = train_test_split(
            train_val,
            test_size=val_ratio_adjusted,
            random_state=self.config.random_seed
        )
        
        logger.info(f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        return train, val, test
    
    def load_and_prepare(
        self, 
        source: str = "huggingface",
        dataset_name: str = "Fatima0923/Automated-Personality-Prediction",
        filepath: str = None,
        use_synthetic_if_needed: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Complete data loading and preparation pipeline.
        
        Returns:
            train_df, val_df, test_df, analysis_dict
        """
        # Load data
        try:
            if source == "huggingface":
                df = self.load_huggingface_dataset(dataset_name)
            elif source == "csv" and filepath:
                df = self.load_csv_dataset(filepath)
            else:
                raise ValueError(f"Unknown source: {source}")
        except Exception as e:
            if use_synthetic_if_needed:
                logger.warning(f"Failed to load dataset ({e}), using synthetic data")
                df = self.create_synthetic_dataset(self.config.min_samples)
            else:
                raise
        
        # Preprocess
        df = self.preprocess_dataset(df)
        
        # Check minimum samples
        if len(df) < self.config.min_samples and use_synthetic_if_needed:
            logger.warning(f"Dataset has {len(df)} samples, below minimum {self.config.min_samples}")
            logger.warning("Augmenting with synthetic data...")
            synthetic = self.create_synthetic_dataset(self.config.min_samples - len(df))
            df = pd.concat([df, synthetic], ignore_index=True)
        
        # Analyze distribution
        analysis = self.analyze_distribution(df)
        
        # Split
        train, val, test = self.split_dataset(df)
        
        return train, val, test, analysis


def load_data(
    config: DataConfig = None,
    source: str = "huggingface",
    dataset_name: str = "Fatima0923/Automated-Personality-Prediction"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Convenience function to load and prepare personality data.
    
    Args:
        config: Data configuration
        source: Data source ("huggingface", "csv")
        dataset_name: Dataset name for HuggingFace
        
    Returns:
        train_df, val_df, test_df, analysis_dict
    """
    loader = PersonalityDataLoader(config)
    return loader.load_and_prepare(source=source, dataset_name=dataset_name)


if __name__ == "__main__":
    # Test data loading
    config = DataConfig(min_samples=1000)
    train, val, test, analysis = load_data(config)
    
    print("\n=== Dataset Analysis ===")
    print(f"Total samples: {analysis['n_samples']}")
    print(f"\nText length statistics:")
    for k, v in analysis['text_length_stats'].items():
        print(f"  {k}: {v:.2f}")
    print(f"\nTrait statistics:")
    for trait, stats in analysis['trait_stats'].items():
        print(f"  {trait}:")
        for k, v in stats.items():
            print(f"    {k}: {v:.4f}")
