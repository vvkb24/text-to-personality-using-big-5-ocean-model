import { useState } from 'react'

const CATEGORY_STYLES = {
  High: 'bg-green-100 text-green-800 border-green-200',
  Medium: 'bg-amber-100 text-amber-800 border-amber-200',
  Low: 'bg-blue-100 text-blue-800 border-blue-200'
}

export default function TraitCard({ trait, info, score, percentile, category, evidence }) {
  const [showEvidence, setShowEvidence] = useState(false)
  
  const scorePercent = Math.round(score * 100)
  
  return (
    <div className="bg-white rounded-xl shadow-md border border-slate-200 overflow-hidden hover:shadow-lg transition-shadow">
      {/* Header with color bar */}
      <div 
        className="h-2" 
        style={{ backgroundColor: info.color }}
      />
      
      <div className="p-5">
        {/* Trait name and icon */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <span className="text-2xl">{info.icon}</span>
            <h4 className="font-semibold text-slate-800">{info.name}</h4>
          </div>
          <span className={`text-xs px-2 py-1 rounded-full border ${CATEGORY_STYLES[category]}`}>
            {category}
          </span>
        </div>
        
        {/* Description */}
        <p className="text-sm text-slate-500 mb-4">{info.description}</p>
        
        {/* Score bar */}
        <div className="mb-4">
          <div className="flex justify-between text-sm mb-1">
            <span className="text-slate-600">Score</span>
            <span className="font-semibold" style={{ color: info.color }}>
              {score.toFixed(3)}
            </span>
          </div>
          <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
            <div 
              className="h-full rounded-full transition-all duration-500 ease-out"
              style={{ 
                width: `${scorePercent}%`,
                backgroundColor: info.color
              }}
            />
          </div>
        </div>
        
        {/* Percentile */}
        <div className="flex items-center justify-between text-sm mb-4">
          <span className="text-slate-600">Percentile</span>
          <span className="font-medium text-slate-800">
            {percentile?.toFixed(1) || 50}th
          </span>
        </div>
        
        {/* Evidence section (if available) */}
        {evidence && evidence.length > 0 && (
          <div className="border-t border-slate-100 pt-3 mt-3">
            <button
              onClick={() => setShowEvidence(!showEvidence)}
              className="flex items-center justify-between w-full text-sm text-slate-600 hover:text-violet-600"
            >
              <span>Evidence ({evidence.length})</span>
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
                {evidence.slice(0, 3).map((sentence, idx) => (
                  <p key={idx} className="text-xs text-slate-500 italic bg-slate-50 p-2 rounded">
                    "{sentence}"
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
