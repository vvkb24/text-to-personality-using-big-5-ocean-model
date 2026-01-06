import { useState } from 'react'
import TextInput from './components/TextInput'
import ResultsDisplay from './components/ResultsDisplay'
import Header from './components/Header'
import Footer from './components/Footer'
import LoadingSpinner from './components/LoadingSpinner'
import ErrorMessage from './components/ErrorMessage'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080'

function App() {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleAnalyze = async (text) => {
    setLoading(true)
    setError(null)
    setResults(null)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setResults(data)
    } catch (err) {
      console.error('Prediction error:', err)
      setError(err.message || 'Failed to analyze text. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-slate-50 to-slate-100">
      <Header />
      
      <main className="flex-grow container mx-auto px-4 py-8 max-w-6xl">
        {/* Hero Section */}
        <div className="text-center mb-10">
          <h1 className="text-4xl md:text-5xl font-bold text-slate-800 mb-4">
            Personality Detection
          </h1>
          <p className="text-lg text-slate-600 max-w-2xl mx-auto">
            Analyze text to discover Big Five (OCEAN) personality traits using 
            machine learning and natural language processing.
          </p>
        </div>

        {/* Input Section */}
        <TextInput onAnalyze={handleAnalyze} loading={loading} />

        {/* Loading State */}
        {loading && <LoadingSpinner />}

        {/* Error State */}
        {error && <ErrorMessage message={error} onDismiss={() => setError(null)} />}

        {/* Results */}
        {results && !loading && <ResultsDisplay results={results} />}
      </main>

      <Footer />
    </div>
  )
}

export default App
