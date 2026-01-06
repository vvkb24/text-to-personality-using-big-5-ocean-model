import RadarChart from './RadarChart'
import TraitCard from './TraitCard'

const TRAIT_INFO = {
  openness: {
    name: 'Openness',
    description: 'Creativity, curiosity, and openness to new experiences',
    icon: '🎨',
    color: '#8B5CF6'
  },
  conscientiousness: {
    name: 'Conscientiousness',
    description: 'Organization, dependability, and self-discipline',
    icon: '📋',
    color: '#3B82F6'
  },
  extraversion: {
    name: 'Extraversion',
    description: 'Sociability, assertiveness, and positive emotions',
    icon: '🎉',
    color: '#F59E0B'
  },
  agreeableness: {
    name: 'Agreeableness',
    description: 'Cooperation, trust, and empathy',
    icon: '🤝',
    color: '#10B981'
  },
  neuroticism: {
    name: 'Neuroticism',
    description: 'Emotional sensitivity and stress response',
    icon: '💭',
    color: '#EF4444'
  }
}

const TRAITS_ORDER = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

export default function ResultsDisplay({ results }) {
  if (!results) return null

  const { scores, percentiles, categories, evidence, traits } = results

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Results Header */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-slate-800 mb-2">Analysis Results</h2>
        <p className="text-slate-600">Big Five (OCEAN) personality trait scores</p>
      </div>

      {/* Radar Chart */}
      <div className="bg-white rounded-xl shadow-lg border border-slate-200 p-6">
        <h3 className="text-lg font-semibold text-slate-800 mb-4 text-center">
          Personality Profile Overview
        </h3>
        <RadarChart scores={scores} />
      </div>

      {/* Trait Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {TRAITS_ORDER.map((trait) => (
          <TraitCard
            key={trait}
            trait={trait}
            info={TRAIT_INFO[trait]}
            score={scores[trait]}
            percentile={percentiles[trait]}
            category={categories[trait]}
            evidence={evidence?.[trait] || []}
          />
        ))}
      </div>

      {/* Summary Stats */}
      <div className="bg-gradient-to-r from-violet-50 to-purple-50 rounded-xl border border-violet-200 p-6">
        <h3 className="text-lg font-semibold text-slate-800 mb-4">Analysis Summary</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <p className="text-sm text-slate-600">Dominant Trait</p>
            <p className="text-lg font-semibold text-violet-700">
              {TRAIT_INFO[getDominantTrait(scores)]?.name || 'N/A'}
            </p>
          </div>
          <div className="text-center">
            <p className="text-sm text-slate-600">High Traits</p>
            <p className="text-lg font-semibold text-green-600">
              {Object.values(categories).filter(c => c === 'High').length}
            </p>
          </div>
          <div className="text-center">
            <p className="text-sm text-slate-600">Medium Traits</p>
            <p className="text-lg font-semibold text-amber-600">
              {Object.values(categories).filter(c => c === 'Medium').length}
            </p>
          </div>
          <div className="text-center">
            <p className="text-sm text-slate-600">Low Traits</p>
            <p className="text-lg font-semibold text-blue-600">
              {Object.values(categories).filter(c => c === 'Low').length}
            </p>
          </div>
        </div>
      </div>

      {/* Model Info */}
      <div className="text-center text-sm text-slate-500">
        <p>Analysis performed using Sentence-BERT embeddings with Ridge regression ensemble</p>
        <p className="mt-1">Text length: {results.text_length} characters</p>
      </div>
    </div>
  )
}

function getDominantTrait(scores) {
  return Object.entries(scores).reduce((a, b) => 
    (scores[a] || 0) > (b[1] || 0) ? a : b[0]
  , 'openness')
}
