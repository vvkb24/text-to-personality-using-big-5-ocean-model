import { 
  Radar, 
  RadarChart as RechartsRadarChart, 
  PolarGrid, 
  PolarAngleAxis, 
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip
} from 'recharts'

const TRAIT_LABELS = {
  openness: 'Openness',
  conscientiousness: 'Conscientiousness',
  extraversion: 'Extraversion',
  agreeableness: 'Agreeableness',
  neuroticism: 'Neuroticism'
}

const TRAITS_ORDER = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

export default function RadarChart({ scores }) {
  // Transform scores for Recharts
  const data = TRAITS_ORDER.map(trait => ({
    trait: TRAIT_LABELS[trait],
    score: Math.round((scores[trait] || 0.5) * 100),
    fullMark: 100
  }))

  return (
    <div className="w-full h-80">
      <ResponsiveContainer width="100%" height="100%">
        <RechartsRadarChart cx="50%" cy="50%" outerRadius="70%" data={data}>
          <PolarGrid 
            stroke="#e2e8f0" 
            strokeDasharray="3 3"
          />
          <PolarAngleAxis 
            dataKey="trait" 
            tick={{ fill: '#64748b', fontSize: 12 }}
            tickLine={false}
          />
          <PolarRadiusAxis 
            angle={90} 
            domain={[0, 100]} 
            tick={{ fill: '#94a3b8', fontSize: 10 }}
            tickCount={6}
            axisLine={false}
          />
          <Radar
            name="Score"
            dataKey="score"
            stroke="#8B5CF6"
            fill="#8B5CF6"
            fillOpacity={0.3}
            strokeWidth={2}
            dot={{ fill: '#8B5CF6', strokeWidth: 0, r: 4 }}
            activeDot={{ fill: '#7C3AED', strokeWidth: 0, r: 6 }}
          />
          <Tooltip 
            content={<CustomTooltip />}
          />
        </RechartsRadarChart>
      </ResponsiveContainer>
    </div>
  )
}

function CustomTooltip({ active, payload }) {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    return (
      <div className="bg-white border border-slate-200 rounded-lg shadow-lg p-3">
        <p className="font-semibold text-slate-800">{data.trait}</p>
        <p className="text-sm text-violet-600">
          Score: <span className="font-medium">{data.score}%</span>
        </p>
      </div>
    )
  }
  return null
}
