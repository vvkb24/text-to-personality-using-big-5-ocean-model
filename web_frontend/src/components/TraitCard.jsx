import { useState } from 'react'

/**
 * WHY: Category styling map for consistent visual feedback.
 */
const CATEGORY_STYLES = {
  High: 'bg-green-100 text-green-800 border-green-200',
  Medium: 'bg-amber-100 text-amber-800 border-amber-200',
  Low: 'bg-blue-100 text-blue-800 border-blue-200'
}

/**
 * WHY: Percentile bounds match backend PERCENTILE_MIN/MAX.
 * Frontend clamping as additional safety layer.
 */
const PERCENTILE_MIN = 1.0
const PERCENTILE_MAX = 99.0

/**
 * WHY: Clamp percentile to safe display bounds.
 */
function clampPercentile(percentile) {
  if (percentile === null || percentile === undefined || isNaN(percentile)) {
    return 50.0
  }
  return Math.max(PERCENTILE_MIN, Math.min(PERCENTILE_MAX, percentile))
}

/**
 * WHY: Safely format score for display, handling edge cases.
 */
function formatScore(score) {
  if (score === null || score === undefined || isNaN(score)) {
    return '0.500' // Neutral fallback
  }
  return score.toFixed(3)
}

/**
 * WHY: Get confidence level label for UI display.
 */
function getConfidenceLabel(confidence) {
  if (confidence === null || confidence === undefined || isNaN(confidence)) {
    return 'Unknown'
  }
  if (confidence >= 0.8) return 'High'
  if (confidence >= 0.5) return 'Medium'
  return 'Low'
}

/**
 * WHY: Get confidence badge styling based on level.
 */
function getConfidenceStyle(confidence) {
  const level = getConfidenceLabel(confidence)
  switch (level) {
    case 'High': return 'text-green-600'
    case 'Medium': return 'text-amber-600'
    case 'Low': return 'text-red-500'
    default: return 'text-slate-400'
  }
}

export default function TraitCard({ trait, info, score, percentile, category, evidence, confidence }) {
  const [showEvidence, setShowEvidence] = useState(false)
  
  // WHY: Safely compute score percentage, clamping to valid range
  const safeScore = (score !== null && score !== undefined && !isNaN(score)) ? score : 0.5
  const scorePercent = Math.round(Math.max(0, Math.min(1, safeScore)) * 100)
  
  // WHY: Clamp percentile for display
  const safePercentile = clampPercentile(percentile)
  
  // WHY: Ensure category has valid value
  const safeCategory = CATEGORY_STYLES[category] ? category : 'Medium'
  
  // WHY: Ensure evidence is always an array for safe iteration
  const safeEvidence = Array.isArray(evidence) ? evidence : []
  
  return (
    <div className="bg-white rounded-xl shadow-md border border-slate-200 overflow-hidden hover:shadow-lg transition-shadow">
      {/* Header with color bar */}
      <div 
        className="h-2" 
        style={{ backgroundColor: info?.color || '#6B7280' }}
      />
      
      <div className="p-5">
        {/* Trait name and icon */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <span className="text-2xl">{info?.icon || '📊'}</span>
            <h4 className="font-semibold text-slate-800">{info?.name || trait}</h4>
          </div>
          <span className={`text-xs px-2 py-1 rounded-full border ${CATEGORY_STYLES[safeCategory]}`}>
            {safeCategory}
          </span>
        </div>
        
        {/* Description */}
        <p className="text-sm text-slate-500 mb-4">{info?.description || ''}</p>
        
        {/* Score bar */}
        <div className="mb-4">
          <div className="flex justify-between text-sm mb-1">
            <span className="text-slate-600">Score</span>
            <span className="font-semibold" style={{ color: info?.color || '#6B7280' }}>
              {formatScore(safeScore)}
            </span>
          </div>
          <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
            <div 
              className="h-full rounded-full transition-all duration-500 ease-out"
              style={{ 
                width: `${scorePercent}%`,
                backgroundColor: info?.color || '#6B7280'
              }}
            />
          </div>
        </div>
        
        {/* Percentile - WHY: Display clamped percentile with bounds indicator */}
        <div className="flex items-center justify-between text-sm mb-2">
          <span className="text-slate-600">Percentile</span>
          <span className="font-medium text-slate-800">
            {safePercentile.toFixed(1)}th
          </span>
        </div>

        {/* WHY: Display confidence score if available */}
        {confidence !== undefined && (
          <div className="flex items-center justify-between text-sm mb-4">
            <span className="text-slate-600">Confidence</span>
            <span className={`font-medium ${getConfidenceStyle(confidence)}`}>
              {getConfidenceLabel(confidence)} ({((confidence || 0) * 100).toFixed(0)}%)
            </span>
          </div>
        )}
        
        {/* Evidence section - WHY: Only show if evidence array has items */}
        {safeEvidence.length > 0 && (
          <div className="border-t border-slate-100 pt-3 mt-3">
            <button
              onClick={() => setShowEvidence(!showEvidence)}
              className="flex items-center justify-between w-full text-sm text-slate-600 hover:text-violet-600"
            >
              <span>Evidence ({safeEvidence.length})</span>
              <svg 
                className={`w-4 h-4 transition-transform ${showEvidence ? 'rotate-180' : ''}`}
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            
            {showEvidence && (
              <div className="mt-2 space-y-2">
                {/* WHY: Limit to 3 items and validate each is a string */}
                {safeEvidence.slice(0, 3).map((sentence, idx) => (
                  <p key={idx} className="text-xs text-slate-500 italic bg-slate-50 p-2 rounded">
                    "{typeof sentence === 'string' ? sentence : String(sentence)}"
                  </p>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
