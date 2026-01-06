export default function LoadingSpinner() {
  return (
    <div className="flex flex-col items-center justify-center py-12">
      <div className="relative">
        {/* Outer ring */}
        <div className="w-16 h-16 border-4 border-violet-200 rounded-full"></div>
        {/* Spinning ring */}
        <div className="absolute top-0 left-0 w-16 h-16 border-4 border-violet-600 rounded-full border-t-transparent animate-spin"></div>
      </div>
      <p className="mt-4 text-slate-600 font-medium">Analyzing personality traits...</p>
      <p className="text-sm text-slate-500 mt-1">This may take a few seconds</p>
    </div>
  )
}
