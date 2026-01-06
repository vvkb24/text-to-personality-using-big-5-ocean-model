import { useState } from 'react'

const EXAMPLE_TEXTS = [
  {
    label: "Creative Explorer",
    text: `I have always been curious about how things work, especially when it comes to complex systems and ideas. From a young age, I enjoyed reading books that challenged my way of thinking and introduced perspectives different from my own. I often find myself experimenting with new approaches, whether in my studies or in personal projects, simply to see what happens and what I can learn from the outcome.`
  },
  {
    label: "Organized Planner",
    text: `I run a tight schedule and never miss a deadline. My workspace is always organized, and I believe in doing things right the first time. Planning ahead gives me peace of mind, and I take pride in my attention to detail. I keep detailed to-do lists and calendars, reviewing them daily to ensure nothing falls through the cracks.`
  },
  {
    label: "Social Enthusiast",
    text: `Parties and social events are my favorite! I love meeting new people and can talk for hours. Being around others energizes me, and I'm usually the one organizing group activities. I thrive in team environments and feel most alive when I'm surrounded by friends and interesting conversations.`
  }
]

export default function TextInput({ onAnalyze, loading }) {
  const [text, setText] = useState('')
  const [charCount, setCharCount] = useState(0)
  const MIN_CHARS = 50

  const handleTextChange = (e) => {
    const newText = e.target.value
    setText(newText)
    setCharCount(newText.length)
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (text.trim().length >= MIN_CHARS) {
      onAnalyze(text.trim())
    }
  }

  const handleExampleClick = (exampleText) => {
    setText(exampleText)
    setCharCount(exampleText.length)
  }

  const isValid = charCount >= MIN_CHARS

  return (
    <div className="bg-white rounded-xl shadow-lg border border-slate-200 p-6 mb-8">
      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <label htmlFor="text-input" className="block text-sm font-medium text-slate-700 mb-2">
            Enter text to analyze
          </label>
          <textarea
            id="text-input"
            value={text}
            onChange={handleTextChange}
            placeholder="Write or paste text here to analyze personality traits. For best results, provide at least 100 characters of natural writing such as a personal essay, journal entry, or self-description..."
            className="w-full h-48 px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-violet-500 focus:border-violet-500 resize-none text-slate-700 placeholder-slate-400"
            disabled={loading}
          />
        </div>

        {/* Character count */}
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center space-x-2">
            <span className={`text-sm ${isValid ? 'text-green-600' : 'text-slate-500'}`}>
              {charCount} / {MIN_CHARS} min characters
            </span>
            {isValid && (
              <svg className="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
            )}
          </div>
          <button
            type="button"
            onClick={() => { setText(''); setCharCount(0); }}
            className="text-sm text-slate-500 hover:text-slate-700"
          >
            Clear
          </button>
        </div>

        {/* Example texts */}
        <div className="mb-6">
          <p className="text-sm text-slate-600 mb-2">Try an example:</p>
          <div className="flex flex-wrap gap-2">
            {EXAMPLE_TEXTS.map((example, index) => (
              <button
                key={index}
                type="button"
                onClick={() => handleExampleClick(example.text)}
                className="text-xs bg-slate-100 text-slate-600 px-3 py-1.5 rounded-full hover:bg-violet-100 hover:text-violet-700 transition"
                disabled={loading}
              >
                {example.label}
              </button>
            ))}
          </div>
        </div>

        {/* Submit button */}
        <button
          type="submit"
          disabled={!isValid || loading}
          className={`w-full py-3 px-6 rounded-lg font-medium transition flex items-center justify-center space-x-2
            ${isValid && !loading
              ? 'bg-gradient-to-r from-violet-600 to-purple-600 text-white hover:from-violet-700 hover:to-purple-700 shadow-md hover:shadow-lg'
              : 'bg-slate-200 text-slate-400 cursor-not-allowed'
            }`}
        >
          {loading ? (
            <>
              <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span>Analyzing...</span>
            </>
          ) : (
            <>
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
              <span>Analyze Personality</span>
            </>
          )}
        </button>
      </form>
    </div>
  )
}
