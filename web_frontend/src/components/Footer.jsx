export default function Footer() {
  return (
    <footer className="bg-white border-t border-slate-200 mt-auto">
      <div className="container mx-auto px-4 py-6 max-w-6xl">
        {/* Disclaimer */}
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 mb-6">
          <div className="flex items-start space-x-3">
            <svg className="w-5 h-5 text-amber-500 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            <div>
              <h4 className="text-sm font-medium text-amber-800">Research Disclaimer</h4>
              <p className="text-sm text-amber-700 mt-1">
                Personality predictions are probabilistic estimates based on text analysis. 
                They are <strong>not clinical diagnoses</strong> and should not be used for 
                medical, employment, or legal decisions. Results are for educational and 
                research purposes only.
              </p>
            </div>
          </div>
        </div>

        {/* Footer Info */}
        <div className="flex flex-col md:flex-row justify-between items-center text-sm text-slate-500">
          <div className="mb-4 md:mb-0">
            <p>© 2026 Personality Detection System. Research Project.</p>
          </div>
          <div className="flex items-center space-x-4">
            <span className="flex items-center">
              <span className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></span>
              ML Model Active
            </span>
            <span>|</span>
            <span>Big Five (OCEAN) Model</span>
          </div>
        </div>
      </div>
    </footer>
  )
}
