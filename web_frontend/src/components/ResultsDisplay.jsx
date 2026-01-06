import RadarChart from './RadarChart'
import TraitCard from './TraitCard'

/**
 * WHY: Centralized trait metadata for consistent display across components.
 * This prevents duplication and ensures UI consistency.
 */
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

/**
 * WHY: Percentile bounds must match backend PERCENTILE_MIN/MAX (1.0-99.0).
 * This prevents displaying extreme percentiles that imply false certainty.
 */
const PERCENTILE_MIN = 1.0
const PERCENTILE_MAX = 99.0

/**
 * WHY: Clamp percentile to safe display bounds.
 * Even if backend sends 0 or 100 (it shouldn't), frontend guards against it.
 */
function clampPercentile(percentile) {
  if (percentile === null || percentile === undefined || isNaN(percentile)) {
    return 50.0 // WHY: Neutral fallback for missing data
  }
  return Math.max(PERCENTILE_MIN, Math.min(PERCENTILE_MAX, percentile))
}

/**
 * WHY: Safely get a value from an object with a default fallback.
 * Prevents crashes when backend response has unexpected structure.
 */
function safeGet(obj, key, defaultValue) {
  if (!obj || typeof obj !== 'object') return defaultValue
  const value = obj[key]
  return value !== undefined && value !== null ? value : defaultValue
}

/**
 * WHY: Safely get evidence array, ensuring it's always an array.
 * Backend may return null, undefined, or non-array values.
 */
function safeGetEvidence(evidence, trait) {
  if (!evidence || typeof evidence !== 'object') return []
  const traitEvidence = evidence[trait]
  if (!traitEvidence) return []
  if (!Array.isArray(traitEvidence)) return []
  // WHY: Filter out any non-string items for safety
  return traitEvidence.filter(item => typeof item === 'string')
}

export default function ResultsDisplay({ results }) {
  // WHY: Guard against null/undefined results
  if (!results) return null

  // WHY: Destructure with defaults to handle missing fields gracefully
  const { 
    scores = {}, 
    percentiles = {}, 
    categories = {}, 
    evidence = {},
    confidences = {},
    warning = null,
    traits = {} 
  } = results

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* WHY: Display warning message if backend indicates suboptimal input */}
      {warning && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-amber-800 text-sm">
          <span className="font-medium">⚠️ Note:</span> {warning}
        </div>
      )}

      {/* Results Header */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-slate-800 mb-2">Analysis Results</h2>
        <p className="text-slate-600">Big Five (OCEAN) personality trait scores</p>
      </div>

      {/* Radar Chart - WHY: Pass safe scores object */}
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
            // WHY: Use safeGet for all values to handle missing fields
            score={safeGet(scores, trait, 0.5)}
            // WHY: Clamp percentile to safe display bounds
            percentile={clampPercentile(safeGet(percentiles, trait, 50))}
            category={safeGet(categories, trait, 'Medium')}
            // WHY: Use safeGetEvidence to ensure array type
            evidence={safeGetEvidence(evidence, trait)}
            // WHY: Pass confidence score (new field from backend)
            confidence={safeGet(confidences, trait, 0.5)}
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
              {/* WHY: Safe access with fallback to prevent crash */}
              {TRAIT_INFO[getDominantTrait(scores)]?.name || 'N/A'}
            </p>
          </div>
          <div className="text-center">
            <p className="text-sm text-slate-600">High Traits</p>
            <p className="text-lg font-semibold text-green-600">
              {/* WHY: Guard against missing categories object */}
              {categories ? Object.values(categories).filter(c => c === 'High').length : 0}
            </p>
          </div>
          <div className="text-center">
            <p className="text-sm text-slate-600">Medium Traits</p>
            <p className="text-lg font-semibold text-amber-600">
              {categories ? Object.values(categories).filter(c => c === 'Medium').length : 0}
            </p>
          </div>
          <div className="text-center">
            <p className="text-sm text-slate-600">Low Traits</p>
            <p className="text-lg font-semibold text-blue-600">
              {categories ? Object.values(categories).filter(c => c === 'Low').length : 0}
            </p>
          </div>
        </div>
      </div>

      {/* Model Info */}
      <div className="text-center text-sm text-slate-500">
        <p>Analysis performed using Sentence-BERT embeddings with Ridge regression ensemble</p>
        {/* WHY: Safe access to text_length with fallback */}
        <p className="mt-1">Text length: {results.text_length || 'N/A'} characters</p>
      </div>
    </div>
  )
}

/**
 * WHY: Safely determine dominant trait from scores object.
 * Handles missing scores, non-numeric values, and empty objects.
 */
function getDominantTrait(scores) {
  // WHY: Guard against null/undefined/empty scores
  if (!scores || typeof scores !== 'object' || Object.keys(scores).length === 0) {
    return 'openness' // Fallback to first trait
  }
  
  try {
    return Object.entries(scores).reduce((dominant, [trait, score]) => {
      // WHY: Validate score is a valid number
      const currentScore = typeof score === 'number' && !isNaN(score) ? score : 0
      const dominantScore = typeof scores[dominant] === 'number' && !isNaN(scores[dominant]) 
        ? scores[dominant] 
        : 0
      return currentScore > dominantScore ? trait : dominant
    }, 'openness')
  } catch (e) {
    // WHY: Catch any unexpected errors and return safe fallback
    console.error('Error computing dominant trait:', e)
    return 'openness'
  }
}
