"""
Example Inference Script
========================

This script demonstrates how to use the trained personality prediction system
for inference on new texts.
"""

import os
import json

from dotenv import load_dotenv
from src.pipeline import PersonalityPredictor, PipelineConfig, create_predictor

# Load environment variables
load_dotenv()


def example_single_prediction():
    """Demonstrate single text prediction."""
    print("=" * 70)
    print("EXAMPLE 1: Single Text Prediction")
    print("=" * 70)
    
    # Sample text to analyze
    sample_text = """
    I absolutely love exploring new places and meeting different people from various cultures.
    Yesterday, I spent hours researching ancient philosophy and came across some fascinating ideas.
    I'm quite organized with my work and always make detailed plans before starting any project.
    Sometimes I feel anxious about upcoming deadlines, but I try to stay positive.
    My friends often say I'm easy to talk to and always ready to help when needed.
    I enjoy social gatherings and feel energized after spending time with friends.
    """
    
    # Create predictor (using mock LLM for demo)
    predictor = create_predictor(use_mock_llm=True)
    
    # Quick train on synthetic data for demo
    print("\nTraining on synthetic data for demonstration...")
    from src.data_loader import PersonalityDataLoader, DataConfig
    
    loader = PersonalityDataLoader(DataConfig(min_samples=500))
    df = loader.create_synthetic_dataset(500)
    
    OCEAN_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    train_texts = df["text"].tolist()
    train_labels = {trait: df[trait].values for trait in OCEAN_TRAITS}
    
    predictor.train(train_texts, train_labels)
    
    # Make prediction
    print("\nAnalyzing text...")
    prediction = predictor.predict(sample_text)
    
    # Print results
    print("\n" + prediction.summary())


def example_batch_prediction():
    """Demonstrate batch prediction."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Batch Prediction")
    print("=" * 70)
    
    # Multiple texts to analyze
    texts = [
        "I love art and creativity. Abstract concepts fascinate me and I'm always curious about new ideas.",
        "I'm very organized and always plan ahead. Meeting deadlines is important to me.",
        "I enjoy parties and social events. Meeting new people energizes me.",
        "I always try to help others and believe in cooperation over competition.",
        "I often feel worried and anxious about things. Stress affects me deeply."
    ]
    
    # Create predictor
    predictor = create_predictor(use_mock_llm=True)
    
    # Train
    from src.data_loader import PersonalityDataLoader, DataConfig
    loader = PersonalityDataLoader(DataConfig(min_samples=500))
    df = loader.create_synthetic_dataset(500)
    
    OCEAN_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    train_texts = df["text"].tolist()
    train_labels = {trait: df[trait].values for trait in OCEAN_TRAITS}
    
    predictor.train(train_texts, train_labels)
    
    # Batch prediction
    predictions = predictor.predict_batch(texts, include_llm=True)
    
    # Print results
    print("\nBatch Results:")
    print("-" * 50)
    
    for i, (text, pred) in enumerate(zip(texts, predictions)):
        print(f"\nText {i+1}: \"{text[:60]}...\"")
        print("Scores:")
        for trait in OCEAN_TRAITS:
            if trait in pred.scores:
                score = pred.scores[trait]
                category = pred.categories[trait]
                print(f"  {trait.capitalize()}: {score:.3f} ({category})")


def example_detailed_analysis():
    """Demonstrate detailed analysis with evidence."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Detailed Analysis with Evidence")
    print("=" * 70)
    
    text = """
    As a scientist, I spend my days exploring the unknown and questioning established theories.
    I find great joy in discovering new patterns and making unexpected connections.
    My laboratory is meticulously organized - every sample labeled, every result documented.
    I collaborate closely with colleagues around the world, sharing findings openly.
    While research can be stressful, especially during grant deadlines, I remain focused on the bigger picture.
    The pursuit of knowledge gives my life meaning and purpose.
    """
    
    # Create predictor
    predictor = create_predictor(
        api_key=os.getenv("GEMINI_API_KEY"),  # Use real API if available
        use_mock_llm=not os.getenv("GEMINI_API_KEY")  # Fall back to mock
    )
    
    # Train
    from src.data_loader import PersonalityDataLoader, DataConfig
    loader = PersonalityDataLoader(DataConfig(min_samples=500))
    df = loader.create_synthetic_dataset(500)
    
    OCEAN_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    train_texts = df["text"].tolist()
    train_labels = {trait: df[trait].values for trait in OCEAN_TRAITS}
    
    predictor.train(train_texts, train_labels)
    
    # Full analysis
    analysis = predictor.analyze(text)
    
    # Print detailed results
    print("\nInput Text:")
    print(f"  {analysis['input_text']}")
    print(f"\nText Length: {analysis['text_length']} characters")
    
    print("\nDetailed Trait Analysis:")
    print("-" * 50)
    
    for trait, data in analysis['traits'].items():
        print(f"\n{trait.upper()}")
        print(f"  Score: {data['score']:.3f}")
        print(f"  Percentile: {data['percentile']:.1f}")
        print(f"  Category: {data['category']}")
        print(f"  Confidence: {data['confidence']:.3f}")
        print(f"  ML Score: {data['ml_score']:.3f}")
        print(f"  LLM Score: {data['llm_score']:.3f}")
        
        if data['evidence']:
            print("  Evidence:")
            for ev in data['evidence'][:2]:
                print(f"    - \"{ev[:80]}{'...' if len(ev) > 80 else ''}\"")
        
        if data['justification']:
            print(f"  Justification: {data['justification']}")


def example_json_output():
    """Demonstrate JSON output format."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: JSON Output Format")
    print("=" * 70)
    
    text = "I enjoy spending time alone reading and thinking about life."
    
    # Create and train predictor
    predictor = create_predictor(use_mock_llm=True)
    
    from src.data_loader import PersonalityDataLoader, DataConfig
    loader = PersonalityDataLoader(DataConfig(min_samples=500))
    df = loader.create_synthetic_dataset(500)
    
    OCEAN_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    train_texts = df["text"].tolist()
    train_labels = {trait: df[trait].values for trait in OCEAN_TRAITS}
    
    predictor.train(train_texts, train_labels)
    
    # Get analysis as JSON
    analysis = predictor.analyze(text)
    
    print("\nJSON Output:")
    print(json.dumps(analysis, indent=2))


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("PERSONALITY DETECTION SYSTEM - INFERENCE EXAMPLES")
    print("=" * 70)
    
    example_single_prediction()
    example_batch_prediction()
    example_detailed_analysis()
    example_json_output()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
