"""Analyze a specific essay for personality traits."""
import warnings
warnings.filterwarnings('ignore')

from src.pipeline import create_predictor
from src.data_loader import PersonalityDataLoader, DataConfig

# Create predictor
predictor = create_predictor(use_mock_llm=True)

# Train on synthetic data
loader = PersonalityDataLoader(DataConfig(min_samples=500))
df = loader.create_synthetic_dataset(500)

OCEAN_TRAITS = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
train_texts = df['text'].tolist()
train_labels = {trait: df[trait].values for trait in OCEAN_TRAITS}

print('Training model...')
predictor.train(train_texts, train_labels)

# The essay to analyze
essay = """I have always been curious about how things work, especially when it comes to complex systems and ideas. From a young age, I enjoyed reading books that challenged my way of thinking and introduced perspectives different from my own. I often find myself experimenting with new approaches, whether in my studies or in personal projects, simply to see what happens and what I can learn from the outcome."""

print()
print('=' * 70)
print('PERSONALITY ANALYSIS RESULT')
print('=' * 70)
print()
print('INPUT TEXT:')
print('-' * 70)
print(essay)
print('-' * 70)

# Get prediction
result = predictor.predict(essay)

print()
print('OCEAN PERSONALITY SCORES:')
print('=' * 70)
for trait in OCEAN_TRAITS:
    score = result.scores[trait]
    category = result.categories[trait]
    bar = '█' * int(score * 20) + '░' * (20 - int(score * 20))
    print(f'{trait.capitalize():18} [{bar}] {score:.3f} ({category})')

print('=' * 70)
print()
print('INTERPRETATION:')
print('-' * 70)

# Interpret results
interpretations = {
    'openness': {
        'High': 'Highly creative, curious, and open to new experiences. Enjoys intellectual pursuits and abstract thinking.',
        'Medium': 'Moderately open to new experiences. Balances curiosity with practicality.',
        'Low': 'Prefers routine and familiarity. More practical and conventional in thinking.'
    },
    'conscientiousness': {
        'High': 'Very organized, disciplined, and goal-oriented. Strong sense of duty and reliability.',
        'Medium': 'Reasonably organized and dependable. Balances structure with flexibility.',
        'Low': 'More spontaneous and flexible. May prefer adaptability over rigid planning.'
    },
    'extraversion': {
        'High': 'Highly sociable, energetic, and outgoing. Draws energy from social interactions.',
        'Medium': 'Comfortable in social situations but also values alone time.',
        'Low': 'More introverted and reserved. Prefers smaller groups or solitary activities.'
    },
    'agreeableness': {
        'High': 'Very cooperative, trusting, and empathetic. Prioritizes harmony and helping others.',
        'Medium': 'Generally cooperative but can be assertive when needed.',
        'Low': 'More competitive and skeptical. Values independence over conformity.'
    },
    'neuroticism': {
        'High': 'More prone to stress, anxiety, and emotional fluctuations.',
        'Medium': 'Average emotional stability. Handles stress reasonably well.',
        'Low': 'Emotionally stable and resilient. Calm under pressure.'
    }
}

for trait in OCEAN_TRAITS:
    category = result.categories[trait]
    print(f'\n{trait.upper()}:')
    print(f'  {interpretations[trait][category]}')

print()
print('=' * 70)
