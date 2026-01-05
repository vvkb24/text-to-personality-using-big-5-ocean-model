"""
Quick Demo Script
=================

A simple script to quickly test the personality detection system.
Run this to see the system in action without full training.
"""

import os
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()


def quick_demo():
    """Run a quick demonstration of the personality detection system."""
    print("=" * 70)
    print("PERSONALITY DETECTION SYSTEM - QUICK DEMO")
    print("=" * 70)
    
    # Import modules
    print("\n[1/4] Loading modules...")
    from src.data_loader import PersonalityDataLoader, DataConfig
    from src.pipeline import create_predictor
    
    # Generate synthetic training data
    print("[2/4] Creating training data...")
    loader = PersonalityDataLoader(DataConfig(min_samples=500))
    df = loader.create_synthetic_dataset(500)
    
    OCEAN_TRAITS = ["openness", "conscientiousness", "extraversion", 
                    "agreeableness", "neuroticism"]
    train_texts = df["text"].tolist()
    train_labels = {trait: df[trait].values for trait in OCEAN_TRAITS}
    
    # Create and train predictor
    print("[3/4] Training model...")
    
    # Check if we have a Gemini API key
    api_key = os.getenv("GEMINI_API_KEY", "")
    use_mock = not api_key or api_key == ""
    
    if use_mock:
        print("      (Using mock LLM - set GEMINI_API_KEY for real LLM)")
    else:
        print("      (Using Gemini API)")
    
    predictor = create_predictor(
        api_key=api_key,
        use_mock_llm=use_mock
    )
    
    predictor.train(train_texts, train_labels)
    
    # Test predictions
    print("[4/4] Running predictions...")
    
    test_texts = [
        """I'm an artist who loves exploring new techniques and styles. 
        Creativity drives everything I do, and I'm constantly curious about 
        different cultures and philosophies. I find abstract ideas fascinating.""",
        
        """I run a tight schedule and never miss a deadline. My workspace is 
        always organized, and I believe in doing things right the first time.
        Planning ahead gives me peace of mind.""",
        
        """Parties and social events are my favorite! I love meeting new people 
        and can talk for hours. Being around others energizes me, and I'm usually 
        the one organizing group activities.""",
        
        """I believe everyone deserves kindness and understanding. I always try 
        to see things from others' perspectives and offer help whenever I can. 
        Cooperation over competition is my motto.""",
        
        """I tend to worry a lot about things, even small stuff. Deadlines make 
        me anxious, and I often feel stressed. My mood can change quickly 
        depending on what's happening around me."""
    ]
    
    trait_names = ["Openness", "Conscientiousness", "Extraversion", 
                   "Agreeableness", "Neuroticism"]
    
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    
    for i, (text, expected_high) in enumerate(zip(test_texts, trait_names)):
        print(f"\n--- Test {i+1}: Expected high {expected_high} ---")
        print(f"Text: \"{text[:100]}...\"")
        
        prediction = predictor.predict(text, include_llm=True)
        
        print("\nPredictions:")
        for trait in OCEAN_TRAITS:
            score = prediction.scores[trait]
            category = prediction.categories[trait]
            marker = " ◄" if trait == expected_high.lower() else ""
            print(f"  {trait.capitalize():20s}: {score:.3f} ({category}){marker}")
    
    # Interactive mode
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("Enter your own text to analyze (or 'quit' to exit):")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if len(user_input) < 20:
                print("Please enter a longer text (at least 20 characters).")
                continue
            
            prediction = predictor.predict(user_input)
            print("\nResults:")
            for trait in OCEAN_TRAITS:
                score = prediction.scores[trait]
                percentile = prediction.percentiles[trait]
                category = prediction.categories[trait]
                print(f"  {trait.capitalize():20s}: {score:.3f} | {percentile:.0f}th percentile | {category}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    quick_demo()
