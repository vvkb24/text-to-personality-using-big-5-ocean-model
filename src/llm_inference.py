"""
LLM-Based Inference Module
==========================

This module implements personality trait inference using Large Language Models.
Uses Google Gemini API for:
1. OCEAN score inference
2. Evidence sentence extraction
3. Structured JSON output

Features:
- Carefully crafted prompts based on psychological theory
- Rate limiting and retry logic
- Structured output parsing
- Evidence extraction for explainability
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import re

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OCEAN traits
OCEAN_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

# Trait descriptions for prompting
TRAIT_DESCRIPTIONS = {
    "openness": {
        "name": "Openness to Experience",
        "description": "Reflects imagination, creativity, intellectual curiosity, and preference for novelty and variety.",
        "high_characteristics": ["creative", "curious", "imaginative", "open to new experiences", "appreciates art and beauty", "intellectually curious"],
        "low_characteristics": ["conventional", "practical", "prefers routine", "traditional", "down-to-earth"]
    },
    "conscientiousness": {
        "name": "Conscientiousness",
        "description": "Reflects organization, dependability, self-discipline, and preference for planned rather than spontaneous behavior.",
        "high_characteristics": ["organized", "reliable", "disciplined", "goal-oriented", "careful", "thorough", "hardworking"],
        "low_characteristics": ["spontaneous", "flexible", "careless", "impulsive", "disorganized"]
    },
    "extraversion": {
        "name": "Extraversion",
        "description": "Reflects sociability, assertiveness, positive emotions, and tendency to seek stimulation in the company of others.",
        "high_characteristics": ["outgoing", "energetic", "talkative", "assertive", "socially confident", "enthusiastic"],
        "low_characteristics": ["reserved", "quiet", "solitary", "introspective", "prefers small groups"]
    },
    "agreeableness": {
        "name": "Agreeableness",
        "description": "Reflects cooperation, trust, empathy, and concern for social harmony.",
        "high_characteristics": ["cooperative", "trusting", "helpful", "empathetic", "considerate", "kind", "altruistic"],
        "low_characteristics": ["competitive", "skeptical", "challenging", "detached", "analytical about others' motives"]
    },
    "neuroticism": {
        "name": "Neuroticism",
        "description": "Reflects emotional instability, tendency to experience negative emotions like anxiety, anger, or depression.",
        "high_characteristics": ["anxious", "moody", "emotionally reactive", "worried", "vulnerable to stress", "self-conscious"],
        "low_characteristics": ["emotionally stable", "calm", "resilient", "secure", "handles stress well"]
    }
}


@dataclass
class LLMConfig:
    """Configuration for LLM inference."""
    api_key: str = ""
    model: str = "gemini-1.5-flash"
    temperature: float = 0.3
    max_output_tokens: int = 2048
    max_retries: int = 3
    retry_delay: float = 2.0
    requests_per_minute: int = 15
    timeout: int = 60


@dataclass
class LLMPrediction:
    """Container for LLM prediction results."""
    scores: Dict[str, float]
    evidence: Dict[str, List[str]]
    justifications: Dict[str, str]
    raw_response: str = ""
    success: bool = True
    error_message: str = ""


class LLMInferenceEngine:
    """
    LLM-based personality inference engine.
    
    Uses Google Gemini to analyze text and infer OCEAN personality traits
    with evidence extraction.
    """
    
    def __init__(self, config: LLMConfig = None):
        """
        Initialize the LLM inference engine.
        
        Args:
            config: LLM configuration
        """
        self.config = config or LLMConfig()
        
        # Get API key from config or environment
        self.api_key = self.config.api_key or os.getenv("GEMINI_API_KEY", "")
        
        if not self.api_key:
            logger.warning("No Gemini API key provided. LLM inference will be unavailable.")
            self.client = None
        else:
            self._initialize_client()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 60.0 / self.config.requests_per_minute
    
    def _initialize_client(self):
        """Initialize the Gemini client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.config.model)
            logger.info(f"Initialized Gemini client with model: {self.config.model}")
        except ImportError:
            logger.error("google-generativeai package not installed")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.client = None
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for personality analysis."""
        trait_info = []
        for trait, info in TRAIT_DESCRIPTIONS.items():
            trait_info.append(f"""
**{info['name']} ({trait.capitalize()})**
- Description: {info['description']}
- High scorers tend to be: {', '.join(info['high_characteristics'])}
- Low scorers tend to be: {', '.join(info['low_characteristics'])}
""")
        
        return f"""You are an expert psychologist specializing in personality assessment using the Big Five (OCEAN) model.

Your task is to analyze the given text and infer the author's personality traits based on linguistic cues, content, and writing style.

## Big Five Personality Traits:
{''.join(trait_info)}

## Analysis Guidelines:
1. Look for linguistic markers: word choice, sentence structure, emotional expressions
2. Consider content themes: topics discussed, interests expressed, values mentioned
3. Analyze writing style: formal vs informal, detailed vs brief, emotional vs neutral
4. Identify behavioral indicators: described actions, preferences, social patterns
5. Be objective and base assessments on textual evidence only

## Important:
- Scores should be between 0.0 and 1.0
- 0.5 represents average/neutral
- Provide specific text excerpts as evidence
- Be conservative - avoid extreme scores without strong evidence
- Consider that people may not express all traits in a single text"""
    
    def _build_analysis_prompt(self, text: str) -> str:
        """Build the analysis prompt for a specific text."""
        return f"""Analyze the following text and assess the author's Big Five personality traits.

## Text to Analyze:
\"\"\"
{text}
\"\"\"

## Required Output Format:
Respond with ONLY a valid JSON object in this exact format:
{{
    "openness": {{
        "score": <float 0.0-1.0>,
        "evidence": ["<quote from text>", "<quote from text>"],
        "justification": "<brief explanation>"
    }},
    "conscientiousness": {{
        "score": <float 0.0-1.0>,
        "evidence": ["<quote from text>", "<quote from text>"],
        "justification": "<brief explanation>"
    }},
    "extraversion": {{
        "score": <float 0.0-1.0>,
        "evidence": ["<quote from text>", "<quote from text>"],
        "justification": "<brief explanation>"
    }},
    "agreeableness": {{
        "score": <float 0.0-1.0>,
        "evidence": ["<quote from text>", "<quote from text>"],
        "justification": "<brief explanation>"
    }},
    "neuroticism": {{
        "score": <float 0.0-1.0>,
        "evidence": ["<quote from text>", "<quote from text>"],
        "justification": "<brief explanation>"
    }}
}}

Remember:
- Extract actual quotes from the text as evidence
- If no clear evidence exists for a trait, use score 0.5 and note "No clear indicators"
- Be precise with scores (e.g., 0.65, not just 0.5 or 1.0)
- Output ONLY the JSON, no additional text"""
    
    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _parse_response(self, response_text: str) -> Dict:
        """Parse the LLM response into structured data."""
        # Try to extract JSON from response
        try:
            # First, try direct JSON parsing
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in the response
        json_pattern = r'\{[\s\S]*\}'
        matches = re.findall(json_pattern, response_text)
        
        for match in matches:
            try:
                parsed = json.loads(match)
                # Validate structure
                if all(trait in parsed for trait in OCEAN_TRAITS):
                    return parsed
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON found, raise error
        raise ValueError(f"Could not parse JSON from response: {response_text[:200]}...")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_api(self, prompt: str) -> str:
        """Call the Gemini API with retry logic."""
        if not self.client:
            raise RuntimeError("Gemini client not initialized")
        
        self._rate_limit()
        
        response = self.client.generate_content(
            prompt,
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_output_tokens,
            }
        )
        
        return response.text
    
    def predict(self, text: str) -> LLMPrediction:
        """
        Predict personality traits for a single text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            LLMPrediction containing scores, evidence, and justifications
        """
        if not self.client:
            return LLMPrediction(
                scores={trait: 0.5 for trait in OCEAN_TRAITS},
                evidence={trait: [] for trait in OCEAN_TRAITS},
                justifications={trait: "LLM unavailable" for trait in OCEAN_TRAITS},
                success=False,
                error_message="Gemini client not initialized"
            )
        
        try:
            # Build prompts
            system_prompt = self._build_system_prompt()
            analysis_prompt = self._build_analysis_prompt(text)
            full_prompt = f"{system_prompt}\n\n{analysis_prompt}"
            
            # Call API
            response_text = self._call_api(full_prompt)
            
            # Parse response
            parsed = self._parse_response(response_text)
            
            # Extract data
            scores = {}
            evidence = {}
            justifications = {}
            
            for trait in OCEAN_TRAITS:
                trait_data = parsed.get(trait, {})
                scores[trait] = float(trait_data.get("score", 0.5))
                scores[trait] = np.clip(scores[trait], 0.0, 1.0)
                evidence[trait] = trait_data.get("evidence", [])
                justifications[trait] = trait_data.get("justification", "")
            
            return LLMPrediction(
                scores=scores,
                evidence=evidence,
                justifications=justifications,
                raw_response=response_text,
                success=True
            )
            
        except Exception as e:
            logger.error(f"LLM prediction failed: {e}")
            return LLMPrediction(
                scores={trait: 0.5 for trait in OCEAN_TRAITS},
                evidence={trait: [] for trait in OCEAN_TRAITS},
                justifications={trait: str(e) for trait in OCEAN_TRAITS},
                success=False,
                error_message=str(e)
            )
    
    def predict_batch(
        self, 
        texts: List[str], 
        show_progress: bool = True
    ) -> List[LLMPrediction]:
        """
        Predict personality traits for multiple texts.
        
        Args:
            texts: List of input texts
            show_progress: Whether to show progress bar
            
        Returns:
            List of LLMPrediction objects
        """
        from tqdm import tqdm
        
        predictions = []
        iterator = tqdm(texts, desc="LLM Inference") if show_progress else texts
        
        for text in iterator:
            prediction = self.predict(text)
            predictions.append(prediction)
        
        return predictions
    
    def get_scores_array(self, predictions: List[LLMPrediction]) -> Dict[str, np.ndarray]:
        """
        Convert predictions to numpy arrays.
        
        Args:
            predictions: List of LLMPrediction objects
            
        Returns:
            Dictionary mapping trait names to score arrays
        """
        scores = {trait: [] for trait in OCEAN_TRAITS}
        
        for pred in predictions:
            for trait in OCEAN_TRAITS:
                scores[trait].append(pred.scores.get(trait, 0.5))
        
        return {trait: np.array(values) for trait, values in scores.items()}


class MockLLMEngine:
    """
    Mock LLM engine for testing without API calls.
    
    Uses simple heuristics to generate plausible predictions.
    """
    
    def __init__(self):
        """Initialize the mock LLM engine."""
        logger.info("Using Mock LLM Engine (no API calls)")
        
        # Simple keyword dictionaries for trait detection
        self.trait_keywords = {
            "openness": {
                "high": ["creative", "curious", "imagine", "art", "philosophy", "explore", "novel", "abstract", "idea"],
                "low": ["practical", "conventional", "traditional", "routine", "concrete", "realistic"]
            },
            "conscientiousness": {
                "high": ["plan", "organize", "careful", "detail", "goal", "discipline", "work hard", "responsible", "thorough"],
                "low": ["spontaneous", "flexible", "relax", "easy-going", "casual"]
            },
            "extraversion": {
                "high": ["social", "party", "talk", "friend", "energy", "excited", "outgoing", "fun", "people"],
                "low": ["quiet", "alone", "solitude", "reserved", "private", "introvert"]
            },
            "agreeableness": {
                "high": ["help", "kind", "trust", "cooperate", "empathy", "care", "support", "generous", "compassion"],
                "low": ["compete", "challenge", "skeptic", "critic", "independent", "assertive"]
            },
            "neuroticism": {
                "high": ["worry", "anxious", "stress", "nervous", "fear", "sad", "upset", "emotional", "tense"],
                "low": ["calm", "relax", "stable", "confident", "secure", "peaceful"]
            }
        }
    
    def predict(self, text: str) -> LLMPrediction:
        """Generate mock predictions based on keyword matching."""
        text_lower = text.lower()
        scores = {}
        evidence = {}
        justifications = {}
        
        for trait, keywords in self.trait_keywords.items():
            high_count = sum(1 for kw in keywords["high"] if kw in text_lower)
            low_count = sum(1 for kw in keywords["low"] if kw in text_lower)
            
            # Calculate score based on keyword balance
            total = high_count + low_count
            if total > 0:
                score = 0.5 + (high_count - low_count) / (total * 4)
            else:
                score = 0.5
            
            # Add some noise
            score = np.clip(score + np.random.normal(0, 0.05), 0.1, 0.9)
            scores[trait] = float(score)
            
            # Extract evidence (find sentences with keywords)
            sentences = re.split(r'[.!?]', text)
            trait_evidence = []
            for sent in sentences:
                sent = sent.strip()
                if any(kw in sent.lower() for kw in keywords["high"] + keywords["low"]):
                    if len(sent) > 10:
                        trait_evidence.append(sent[:100])
            evidence[trait] = trait_evidence[:2]  # Max 2 evidence pieces
            
            # Generate justification
            if high_count > low_count:
                justifications[trait] = f"Text shows {high_count} high-{trait} indicators"
            elif low_count > high_count:
                justifications[trait] = f"Text shows {low_count} low-{trait} indicators"
            else:
                justifications[trait] = "No clear indicators found"
        
        return LLMPrediction(
            scores=scores,
            evidence=evidence,
            justifications=justifications,
            success=True
        )
    
    def predict_batch(
        self, 
        texts: List[str], 
        show_progress: bool = True
    ) -> List[LLMPrediction]:
        """Generate mock predictions for multiple texts."""
        from tqdm import tqdm
        
        predictions = []
        iterator = tqdm(texts, desc="Mock LLM Inference") if show_progress else texts
        
        for text in iterator:
            predictions.append(self.predict(text))
        
        return predictions
    
    def get_scores_array(self, predictions: List[LLMPrediction]) -> Dict[str, np.ndarray]:
        """Convert predictions to numpy arrays."""
        scores = {trait: [] for trait in OCEAN_TRAITS}
        
        for pred in predictions:
            for trait in OCEAN_TRAITS:
                scores[trait].append(pred.scores.get(trait, 0.5))
        
        return {trait: np.array(values) for trait, values in scores.items()}


def create_llm_engine(config: LLMConfig = None, use_mock: bool = False) -> LLMInferenceEngine:
    """
    Factory function to create LLM inference engine.
    
    Args:
        config: LLM configuration
        use_mock: Whether to use mock engine
        
    Returns:
        LLM inference engine
    """
    if use_mock:
        return MockLLMEngine()
    
    config = config or LLMConfig()
    engine = LLMInferenceEngine(config)
    
    if engine.client is None:
        logger.warning("Falling back to mock LLM engine")
        return MockLLMEngine()
    
    return engine


if __name__ == "__main__":
    # Test LLM inference
    config = LLMConfig(api_key=os.getenv("GEMINI_API_KEY", ""))
    engine = create_llm_engine(config)
    
    # Test text
    test_text = """
    I absolutely love exploring new places and meeting different people from various cultures.
    Yesterday, I spent hours researching ancient philosophy and came across some fascinating ideas.
    I'm quite organized with my work and always make detailed plans before starting any project.
    Sometimes I feel anxious about upcoming deadlines, but I try to stay positive.
    My friends often say I'm easy to talk to and always ready to help when needed.
    """
    
    print("Analyzing text...")
    prediction = engine.predict(test_text)
    
    print("\n=== LLM Prediction Results ===")
    print(f"Success: {prediction.success}")
    
    for trait in OCEAN_TRAITS:
        print(f"\n{trait.capitalize()}:")
        print(f"  Score: {prediction.scores[trait]:.3f}")
        print(f"  Evidence: {prediction.evidence[trait]}")
        print(f"  Justification: {prediction.justifications[trait]}")
