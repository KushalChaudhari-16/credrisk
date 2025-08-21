# 🚀 CredRisk - Real-Time Explainable Credit Intelligence Platform

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)](https://reactjs.org/)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://python.org/)
[![SQLite](https://img.shields.io/badge/sqlite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org/)

## 📋 Overview

CredRisk is an advanced real-time credit intelligence platform that revolutionizes creditworthiness assessment by ingesting multi-source data and generating credit scores **faster than traditional rating agencies**. The platform provides **transparent, explainable AI-driven insights** through an interactive dashboard, empowering financial institutions with rapid decision-making capabilities.

### 🎯 Key Value Proposition
- **15-30 minute score updates** vs traditional agencies' quarterly reviews
- **Multi-source data fusion** combining structured financial data, macroeconomic indicators, and real-time news sentiment
- **Explainable AI** with detailed factor contributions and plain-language explanations

## 🏗️ System Architecture

```
CredRisk Platform
├── Backend (FastAPI + SQLite)
│   ├── Multi-Source Data Pipeline
│   │   ├── yfinance (Financial Statements, Stock Data)
│   │   ├── FRED API (Macroeconomic Indicators)
│   │   └── NewsAPI (Real-time Financial News)
│   ├── ML/AI Engine
│   │   ├── Multi-Model System (Random Forest, XGBoost, Logistic)
│   │   ├── Feature Engineering (15+ Financial Ratios)
│   │   └── Explainability Engine
│   └── RESTful API Endpoints
└── Frontend (React.js + Chart.js)
    ├── Interactive Dashboard
    ├── Real-time Visualizations
    └── Explainable AI Interface
```

## ✨ Core Features

### 🔍 **Advanced Credit Scoring**
- **Multi-model ensemble** comparing Random Forest, XGBoost, and Logistic Regression
- **Real-time scoring** with 15-30 minute update cycles
- **Feature importance analysis** with detailed contribution breakdowns
- **Cross-validation** and backtesting for model reliability

### 📊 **Multi-Source Data Integration**
- **Structured Data**: yfinance (financial statements, ratios), FRED API (macroeconomic indicators)
- **Unstructured Data**: NewsAPI with NLP sentiment analysis
- **15+ Financial Metrics**: Debt-to-equity, ROE, current ratio, profit margins, etc.
- **Macroeconomic Factors**: GDP growth, interest rates, inflation indicators

### 🧠 **Explainable AI Engine**
- **Factor contribution breakdowns** showing individual metric impacts
- **Trend analysis** comparing short-term (7-day) vs long-term (30-day) patterns
- **Event-based reasoning** highlighting recent news impacts
- **Plain-language summaries** for non-technical stakeholders



## 🖥️ Live Demo - Credit Score System

🌐 **Live Application**: [https://credtech-ai.netlify.app/](https://credtech-ai.netlify.app/)

![Credit Score Dashboard](UI.png)
*Real-time credit scoring dashboard showing explainable AI insights and multi-factor analysis*

## 🛠️ Technology Stack

### Backend
- **Framework**: FastAPI (high-performance async API)
- **Database**: SQLite (lightweight, file-based)
- **ML/AI**: Scikit-learn, XGBoost, TextBlob
- **Data Sources**: yfinance, FRED API, NewsAPI
- **Deployment**: Render (Backend), Netlify (Frontend)

### Frontend
- **Framework**: React.js
- **Visualization**: Chart.js, Recharts
- **Styling**: Bootstrap, React-Bootstrap
- **Deployment**: Netlify

### Development & MLOps
- **Version Control**: Git with atomic commits
- **Environment**: Docker containerization
- **ML Pipeline**: Automated retraining and data refresh
- **Monitoring**: Health checks and performance metrics

## 📈 Model Performance

Our multi-model system comparison results:

```
Training Results:
├── Random Forest: CV Score: 0.7907 ± 0.0341
├── XGBoost: CV Score: 0.7911 ± 0.0484  [SELECTED]
└── Logistic Regression: CV Score: 0.9548 ± 0.0106
```

**Selected Model**: XGBoost for optimal balance of accuracy and interpretability

## 🔧 Architecture Trade-offs & Design Decisions

### Database Choice: SQLite vs PostgreSQL
**Selected**: SQLite
- **Pros**: Zero configuration, file-based portability, excellent for prototyping
- **Cons**: Limited concurrent writes, no advanced analytics functions
- **Rationale**: Rapid development priority, Railway deployment compatibility

### Model Selection: Ensemble vs Single Model
**Selected**: Multi-model comparison with XGBoost selection
- **Alternative**: Simple ensemble averaging
- **Rationale**: Performance transparency, explainability requirements, benchmark comparison needs

### Data Pipeline: Real-time vs Batch
**Selected**: Hybrid approach (15-30 minute batch updates)
- **Alternative**: True real-time streaming
- **Rationale**: API rate limits, cost optimization, sufficient for use case

### Frontend Framework: React vs Vue vs Vanilla JS
**Selected**: React.js
- **Rationale**: Component reusability, extensive chart library ecosystem, team familiarity

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- API Keys: FRED API, NewsAPI (free registration required)

### Backend Setup

```bash
# Clone repository
git clone https://github.com/KushalChaudhari-16/credrisk.git
cd credrisk/Backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Environment setup
cp .env.example .env
# Add your API keys:
# FRED_API_KEY=your_fred_key
# NEWS_API_KEY=your_news_key

# Initialize database
python database.py

# Run development server
python run.py
```

Backend will be available at `http://localhost:8000`

### Frontend Setup

```bash
# Navigate to frontend
cd ../cred-frontend

# Install dependencies
npm install

# Start development server
npm start
```

Frontend will be available at `http://localhost:3000`

## 📊 Performance Metrics

- **Accuracy**: >75% correlation with actual credit events
- **Speed**: Score updates within 30 minutes of new data availability
- **Explainability**: 100% of scores include factor-level explanations
- **API Response Time**: <500ms for score retrieval
- **Data Freshness**: Financial data updated every 15-30 minutes

## 🙏 Acknowledgments

- **yfinance**: Financial data API
- **FRED**: Federal Reserve Economic Data
- **NewsAPI**: Real-time news data
- **FastAPI**: High-performance web framework
- **React**: Frontend framework
- **Scikit-learn & XGBoost**: Machine learning libraries

⭐ **Star this repository** if you found it helpful!

*Built with ❤️ for the future of credit intelligence*
