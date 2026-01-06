export default function Header() {
  return (
    <header className="bg-white shadow-sm border-b border-slate-200">
      <div className="container mx-auto px-4 py-4 max-w-6xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-violet-500 to-purple-600 rounded-lg flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-semibold text-slate-800">OCEAN Analyzer</h1>
              <p className="text-xs text-slate-500">Big Five Personality Model</p>
            </div>
          </div>
          
          <nav className="hidden md:flex items-center space-x-6">
            <a href="#about" className="text-sm text-slate-600 hover:text-violet-600 transition">About</a>
            <a href="#methodology" className="text-sm text-slate-600 hover:text-violet-600 transition">Methodology</a>
            <a 
              href="http://localhost:8000/docs" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-sm bg-violet-100 text-violet-700 px-3 py-1.5 rounded-md hover:bg-violet-200 transition"
            >
              API Docs
            </a>
          </nav>
        </div>
      </div>
    </header>
  )
}
