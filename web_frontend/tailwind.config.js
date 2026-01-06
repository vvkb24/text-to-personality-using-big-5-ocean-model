/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ocean: {
          openness: '#8B5CF6',
          conscientiousness: '#3B82F6',
          extraversion: '#F59E0B',
          agreeableness: '#10B981',
          neuroticism: '#EF4444',
        }
      },
      fontFamily: {
        'research': ['Inter', 'system-ui', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
