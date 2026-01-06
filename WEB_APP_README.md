# Personality Detection Web Application

A production-quality research demo for Big Five (OCEAN) personality trait detection from text.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      FRONTEND (React)                            │
│                    http://localhost:3000                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  • Text Input Component                                  │    │
│  │  • Radar Chart (Recharts)                               │    │
│  │  • Trait Cards with Scores                              │    │
│  │  • Evidence Display                                      │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ REST API
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND (FastAPI)                           │
│                    http://localhost:8000                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  POST /predict  - Analyze text for personality traits    │    │
│  │  GET  /health   - Health check endpoint                  │    │
│  │  GET  /docs     - Swagger API documentation              │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Direct Import
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML PIPELINE (src/)                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  • Sentence-BERT Embeddings                              │    │
│  │  • Ridge Regression Models                               │    │
│  │  • Ensemble Prediction                                   │    │
│  │  • Score Calibration                                     │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm or yarn

### 1. Start the Backend

```bash
# Navigate to project root
cd "personality detection vscode"

# Install Python dependencies (if not already installed)
pip install -r requirements.txt
pip install -r web_backend/requirements.txt

# Start the FastAPI server
cd web_backend
python main.py
```

The backend will be available at:
- **API**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 2. Start the Frontend

Open a new terminal:

```bash
# Navigate to frontend directory
cd "personality detection vscode/web_frontend"

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at:
- **Web App**: http://localhost:3000

## API Endpoints

### POST /predict

Analyze text for personality traits.

**Request:**
```json
{
  "text": "I have always been curious about how things work..."
}
```

**Response:**
```json
{
  "scores": {
    "openness": 0.897,
    "conscientiousness": 0.754,
    "extraversion": 0.658,
    "agreeableness": 0.900,
    "neuroticism": 0.450
  },
  "percentiles": {
    "openness": 90.0,
    "conscientiousness": 75.0,
    "extraversion": 66.0,
    "agreeableness": 90.0,
    "neuroticism": 45.0
  },
  "categories": {
    "openness": "High",
    "conscientiousness": "High",
    "extraversion": "Medium",
    "agreeableness": "High",
    "neuroticism": "Medium"
  },
  "evidence": {},
  "traits": {...},
  "text_length": 391,
  "model_info": {
    "model_type": "ensemble",
    "ml_model": "sentence-transformers/all-MiniLM-L6-v2",
    "calibrated": true
  }
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

## Configuration

### Backend Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here  # Optional: for LLM-enhanced predictions
```

### Frontend Environment Variables

Create a `.env` file in `web_frontend/`:

```env
VITE_API_URL=http://localhost:8000
```

## Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation

### Frontend
- **React 18** - UI library
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Recharts** - Charts and visualizations

### ML Pipeline (unchanged)
- **Sentence-BERT** - Text embeddings
- **Scikit-learn** - Ridge regression
- **PyTorch** - Deep learning backend

## Features

### UI Features
- ✅ Text input with character count validation
- ✅ Example text templates
- ✅ Radar chart for personality overview
- ✅ Individual trait cards with score bars
- ✅ Category badges (Low/Medium/High)
- ✅ Percentile rankings
- ✅ Evidence sentences (when available)
- ✅ Loading and error states
- ✅ Research disclaimer

### API Features
- ✅ Swagger/OpenAPI documentation
- ✅ CORS enabled
- ✅ Input validation
- ✅ Meaningful error responses
- ✅ Health check endpoint
- ✅ Model auto-loading at startup

## Development

### Backend Development

```bash
cd web_backend
python main.py  # Runs with auto-reload
```

### Frontend Development

```bash
cd web_frontend
npm run dev     # Runs with hot module replacement
npm run build   # Production build
npm run preview # Preview production build
```

## Disclaimer

⚠️ **Research Disclaimer**: Personality predictions are probabilistic estimates based on text analysis. They are **not clinical diagnoses** and should not be used for medical, employment, or legal decisions. Results are for educational and research purposes only.

## License

Research Project © 2026
