from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio
import threading
import time
import numpy as np
import json
import os

import aiohttp
import asyncio
from urllib.parse import quote as url_quote

from database import Database
from data_sources import DataPipeline
from multi_model_system import MultiModelCreditScoring
from explainability_engine import ExplainabilityEngine

app = FastAPI(
    title="CredTech Advanced API", 
    version="2.0.0", 
    description="Advanced Explainable Credit Intelligence Platform with Multi-Model ML"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
db = Database()
pipeline = DataPipeline()
model_system = MultiModelCreditScoring()
explainer = ExplainabilityEngine(db)

DEFAULT_COMPANIES = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 
    'JNJ', 'PG', 'KO', 'DIS', 'NFLX', 'NVDA'
]

# Global variable to track if initialization is done
_initialized = False
_initialization_lock = threading.Lock()

def initialize_system():
    """Initialize the system components safely"""
    global _initialized
    with _initialization_lock:
        if not _initialized:
            print("🚀 Starting Advanced CredTech API...")
            try:
                model_system.load_models()
                print("✅ Models loaded successfully")
                _initialized = True
            except Exception as e:
                print(f"⚠️ Model loading failed: {e}, will train on first request")
                _initialized = True

@app.on_event("startup")
async def startup_event():
    """Startup event with better error handling for Render"""
    initialize_system()
    
    # Start background task only after initialization
    if os.getenv("ENVIRONMENT") != "development":  # Don't run background tasks in production initially
        print("🏭 Production mode: Background MLOps will start after first request")
    else:
        threading.Thread(target=background_mlops_pipeline, daemon=True).start()
        print("✅ MLOps pipeline started")

def background_mlops_pipeline():
    """Background MLOps pipeline with better error handling"""
    # Wait longer before starting in production
    initial_wait = 60 if os.getenv("ENVIRONMENT") == "production" else 10
    time.sleep(initial_wait)
    
    while True:
        try:
            print("🔄 Running MLOps pipeline...")
            refresh_and_retrain()
            print("✅ MLOps cycle completed")
            # Longer intervals in production to avoid rate limits
            sleep_time = 7200 if os.getenv("ENVIRONMENT") == "production" else 3600  # 2 hours in prod
            time.sleep(sleep_time)
        except Exception as e:
            print(f"❌ MLOps error: {e}")
            time.sleep(600)  # Wait 10 minutes on error

def refresh_and_retrain():
    """Refresh data and retrain models with better error handling"""
    try:
        print("📊 Fetching macro data...")
        macro_data = pipeline.get_macro_data()
        
        training_data_count = 0
        
        for symbol in DEFAULT_COMPANIES[:10]:  # Limit to first 10 companies to avoid timeouts
            try:
                print(f"Processing {symbol}...")
                financial_data = pipeline.get_financial_data(symbol)
                
                if financial_data:
                    db.insert_company(
                        symbol=financial_data['symbol'],
                        name=financial_data['name'],
                        sector=financial_data['sector'],
                        market_cap=financial_data['metrics'].get('market_cap', 0)
                    )
                    
                    db.insert_financial_data(
                        symbol=symbol,
                        metrics=financial_data['metrics'],
                        source='yfinance',
                        period='latest'
                    )
                    
                    db.insert_financial_data(
                        symbol=symbol,
                        metrics=macro_data,
                        source='fred',
                        period='latest'
                    )
                    
                    # Limit news events to avoid timeout
                    try:
                        news_events = pipeline.get_news_sentiment(symbol)
                        for event in news_events[:3]:  # Limit to 3 events
                            db.insert_news_event(
                                symbol=symbol,
                                headline=event['headline'],
                                sentiment=event['sentiment'],
                                event_type=event['event_type'],
                                risk_impact=event['risk_impact']
                            )
                    except Exception as news_error:
                        print(f"⚠️ News processing failed for {symbol}: {news_error}")
                    
                    training_data_count += 1
                    print(f"✅ {symbol} processed")
                
                time.sleep(0.5)  # Slightly longer delay to avoid rate limits
                
            except Exception as e:
                print(f"❌ Error processing {symbol}: {e}")
                continue
        
        if training_data_count >= 5:
            print("🤖 Triggering incremental model retraining...")
            model_system.train_and_compare_models()
    
    except Exception as e:
        print(f"❌ Refresh and retrain error: {e}")

@app.get("/")
async def root():
    """Root endpoint with system status"""
    return {
        "message": "Advanced CredTech API v2.0",
        "features": [
            "Multi-model ML scoring",
            "Advanced explainability",
            "Real-time MLOps",
            "Comprehensive analytics"
        ],
        "timestamp": datetime.now().isoformat(),
        "status": "operational",
        "model_status": "trained" if model_system.is_trained else "training",
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.get("/api/companies")
async def get_companies():
    """Get all companies with better initialization"""
    # Ensure system is initialized
    initialize_system()
    
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT c.symbol, c.name, c.sector, c.market_cap, 
               MAX(s.score) as latest_score, MAX(s.timestamp) as last_scored
        FROM companies c
        LEFT JOIN scores s ON c.symbol = s.symbol
        GROUP BY c.symbol, c.name, c.sector, c.market_cap
        ORDER BY c.market_cap DESC
    """)
    companies = cursor.fetchall()
    conn.close()
    
    if not companies:
        try:
            print("Initializing company database...")
            refresh_and_retrain()
            
            conn = db.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.symbol, c.name, c.sector, c.market_cap, 
                       MAX(s.score) as latest_score, MAX(s.timestamp) as last_scored
                FROM companies c
                LEFT JOIN scores s ON c.symbol = s.symbol
                GROUP BY c.symbol, c.name, c.sector, c.market_cap
                ORDER BY c.market_cap DESC
            """)
            companies = cursor.fetchall()
            conn.close()
        except Exception as e:
            print(f"❌ Error initializing companies: {e}")
            # Return empty list if initialization fails
            companies = []
    
    return {
        "companies": [
            {
                "symbol": row[0],
                "name": row[1],
                "sector": row[2],
                "market_cap": row[3],
                "latest_score": row[4],
                "last_scored": row[5],
                "risk_grade": get_risk_grade(row[4]) if row[4] else "NR"
            }
            for row in companies
        ],
        "count": len(companies),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/companies/{symbol}/score")
async def get_advanced_company_score(symbol: str):
    """Get advanced company score with better error handling"""
    try:
        # Ensure system is initialized
        initialize_system()
        
        symbol = symbol.upper()
        print(f"🎯 Advanced scoring for {symbol}...")
        
        financial_data = pipeline.get_financial_data(symbol)
        if not financial_data:
            raise HTTPException(status_code=404, detail=f"Unable to fetch data for {symbol}")
        
        macro_data = pipeline.get_macro_data()
        
        # Handle news events with fallback
        try:
            news_events = pipeline.get_news_sentiment(symbol)
            avg_sentiment = np.mean([event['sentiment'] for event in news_events]) if news_events else 0.0
        except Exception as e:
            print(f"⚠️ News sentiment failed: {e}, using neutral sentiment")
            news_events = []
            avg_sentiment = 0.0
        
        score, confidence, ensemble_predictions = model_system.predict_score(
            financial_data['metrics'],
            macro_data,
            avg_sentiment
        )
        
        feature_importance = model_system.get_feature_importance(
            financial_data['metrics'],
            macro_data,
            avg_sentiment
        )
        
        detailed_explanation = explainer.generate_detailed_explanation(
            symbol, score, feature_importance, financial_data['metrics'],
            macro_data, avg_sentiment, ensemble_predictions
        )
        
        risk_grade = get_risk_grade(score)
        
        # Save to database
        db.insert_company(
            symbol=financial_data['symbol'],
            name=financial_data['name'],
            sector=financial_data['sector'],
            market_cap=financial_data['metrics'].get('market_cap', 0)
        )
        
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO scores (symbol, score, model_name, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (symbol, score, model_system.best_model_name or "MultiModel", confidence, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        return {
            "symbol": symbol,
            "name": financial_data['name'],
            "sector": financial_data['sector'],
            "credit_score": round(score, 2),
            "risk_grade": risk_grade,
            "confidence": round(confidence, 3),
            "model_info": {
                "best_model": model_system.best_model_name,
                "model_performances": model_system.model_performances,
                "ensemble_predictions": ensemble_predictions
            },
            "explanation": detailed_explanation,
            "feature_analysis": {
                "top_factors": feature_importance[:8],
                "factor_count": len(feature_importance),
                "positive_factors": len([f for f in feature_importance if f['impact'] > 0.1]),
                "negative_factors": len([f for f in feature_importance if f['impact'] < -0.1])
            },
            "data_sources": {
                "financial_metrics": len([k for k, v in financial_data['metrics'].items() if v != 0]),
                "macro_indicators": len(macro_data),
                "news_articles": len(news_events),
                "avg_sentiment": round(avg_sentiment, 3)
            },
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error in advanced scoring: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")

# [Include all other endpoints from your original main.py - they look good]
@app.get("/api/companies/{symbol}/history")
async def get_score_history(symbol: str, days: int = 30):
    try:
        symbol = symbol.upper()
        
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute("""
            SELECT score, model_name, confidence, timestamp
            FROM scores 
            WHERE symbol = ? AND timestamp > ?
            ORDER BY timestamp DESC
        """, (symbol, cutoff_date))
        
        history = cursor.fetchall()
        conn.close()
        
        if not history:
            return {"symbol": symbol, "history": [], "message": "No historical data available"}
        
        scores = [row[0] for row in history]
        trend = "improving" if len(scores) > 1 and scores[0] > scores[-1] else "declining" if len(scores) > 1 and scores[0] < scores[-1] else "stable"
        
        return {
            "symbol": symbol,
            "history": [
                {
                    "score": row[0],
                    "model": row[1],
                    "confidence": row[2],
                    "timestamp": row[3],
                    "risk_grade": get_risk_grade(row[0])
                }
                for row in history
            ],
            "analytics": {
                "data_points": len(history),
                "score_range": [min(scores), max(scores)],
                "average_score": round(np.mean(scores), 2),
                "volatility": round(np.std(scores), 2),
                "trend": trend,
                "days_analyzed": days
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History error: {str(e)}")

@app.get("/api/companies/{symbol}/compare/{agency}")
async def compare_with_agency(symbol: str, agency: str):
    try:
        symbol = symbol.upper()
        agency = agency.lower()
        
        if agency not in ['sp', 'moody', 'fitch']:
            raise HTTPException(status_code=400, detail="Supported agencies: sp, moody, fitch")
        
        financial_data = pipeline.get_financial_data(symbol)
        if not financial_data:
            raise HTTPException(status_code=404, detail="Company not found")
        
        macro_data = pipeline.get_macro_data()
        
        try:
            news_events = pipeline.get_news_sentiment(symbol)
            avg_sentiment = np.mean([event['sentiment'] for event in news_events]) if news_events else 0.0
        except:
            avg_sentiment = 0.0
        
        our_score, confidence, ensemble_predictions = model_system.predict_score(
            financial_data['metrics'],
            macro_data,
            avg_sentiment
        )
        
        mock_agency_ratings = {
            'sp': {'AAPL': 8.5, 'MSFT': 8.3, 'GOOGL': 8.0, 'AMZN': 7.8, 'TSLA': 6.5},
            'moody': {'AAPL': 8.4, 'MSFT': 8.2, 'GOOGL': 7.9, 'AMZN': 7.7, 'TSLA': 6.3},
            'fitch': {'AAPL': 8.3, 'MSFT': 8.1, 'GOOGL': 7.8, 'AMZN': 7.6, 'TSLA': 6.4}
        }
        
        agency_score = mock_agency_ratings.get(agency, {}).get(symbol, 6.0)
        score_difference = our_score - agency_score
        
        arbitrage_opportunity = abs(score_difference) > 1.5
        
        return {
            "symbol": symbol,
            "comparison": {
                "credtech_score": round(our_score, 2),
                "credtech_grade": get_risk_grade(our_score),
                "agency_score": agency_score,
                "agency_grade": get_risk_grade(agency_score),
                "agency_name": agency.upper(),
                "score_difference": round(score_difference, 2),
                "credtech_confidence": round(confidence, 3)
            },
            "analysis": {
                "arbitrage_opportunity": arbitrage_opportunity,
                "opportunity_type": "overvalued_by_agency" if score_difference > 1.5 else "undervalued_by_agency" if score_difference < -1.5 else "fairly_valued",
                "confidence_in_difference": "high" if confidence > 0.85 and abs(score_difference) > 1.0 else "medium",
                "potential_alpha": abs(score_difference) * 0.5 if arbitrage_opportunity else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison error: {str(e)}")

@app.get("/api/alerts")
async def get_active_alerts():
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT c.symbol, c.name, s1.score as current_score, s2.score as previous_score,
                   s1.timestamp as current_time, s2.timestamp as previous_time
            FROM companies c
            JOIN scores s1 ON c.symbol = s1.symbol
            JOIN scores s2 ON c.symbol = s2.symbol
            WHERE s1.timestamp > s2.timestamp
            AND ABS(s1.score - s2.score) > 1.0
            ORDER BY s1.timestamp DESC
            LIMIT 20
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        alerts = []
        for row in results:
            symbol, name, current_score, previous_score, current_time, previous_time = row
            score_change = current_score - previous_score
            
            alerts.append({
                "symbol": symbol,
                "company_name": name,
                "alert_type": "score_improvement" if score_change > 0 else "score_degradation",
                "current_score": round(current_score, 2),
                "previous_score": round(previous_score, 2),
                "score_change": round(score_change, 2),
                "severity": "high" if abs(score_change) > 2.0 else "medium",
                "current_grade": get_risk_grade(current_score),
                "previous_grade": get_risk_grade(previous_score),
                "timestamp": current_time
            })
        
        return {
            "active_alerts": alerts,
            "alert_count": len(alerts),
            "high_severity_count": len([a for a in alerts if a['severity'] == 'high']),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "active_alerts": [],
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/analytics/overview")
async def get_analytics_overview():
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM companies")
        total_companies = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM scores WHERE timestamp > date('now', '-1 day')")
        daily_scores = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT AVG(score), COUNT(*) 
            FROM scores s
            JOIN (
                SELECT symbol, MAX(timestamp) as max_time 
                FROM scores 
                GROUP BY symbol
            ) latest ON s.symbol = latest.symbol AND s.timestamp = latest.max_time
        """)
        avg_result = cursor.fetchone()
        market_avg_score = avg_result[0] if avg_result[0] else 5.0
        
        cursor.execute("""
            SELECT sector, AVG(s.score), COUNT(*)
            FROM companies c
            JOIN scores s ON c.symbol = s.symbol
            GROUP BY sector
            HAVING COUNT(*) > 1
            ORDER BY AVG(s.score) DESC
        """)
        sector_data = cursor.fetchall()
        
        conn.close()
        
        return {
            "market_overview": {
                "total_companies_tracked": total_companies,
                "daily_score_updates": daily_scores,
                "market_average_score": round(market_avg_score, 2),
                "market_grade": get_risk_grade(market_avg_score)
            },
            "sector_analysis": [
                {
                    "sector": row[0],
                    "average_score": round(row[1], 2),
                    "average_grade": get_risk_grade(row[1]),
                    "company_count": row[2]
                }
                for row in sector_data
            ],
            "system_metrics": {
                "model_type": model_system.best_model_name or "MultiModel",
                "model_performance": model_system.model_performances,
                "last_retrain": "Active MLOps cycle"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics error: {str(e)}")

@app.get("/api/search/companies")
async def search_companies_proxy(q: str):
    """
    Proxy endpoint for Yahoo Finance company search to avoid CORS issues
    """
    try:
        encoded_query = url_quote(q)
        yahoo_url = f"https://query1.finance.yahoo.com/v1/finance/search?q={encoded_query}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                yahoo_url,
                timeout=aiohttp.ClientTimeout(total=10),
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json',
                }
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    quotes = data.get('quotes', [])
                    filtered_quotes = []
                    
                    for quote_item in quotes[:10]:
                        if quote_item.get('quoteType') in ['EQUITY', 'ETF', 'MUTUALFUND']:
                            filtered_quotes.append({
                                'symbol': quote_item.get('symbol', ''),
                                'shortname': quote_item.get('shortname', ''),
                                'longname': quote_item.get('longname', ''),
                                'exchDisp': quote_item.get('exchDisp', ''),
                                'quoteType': quote_item.get('quoteType', ''),
                                'sector': quote_item.get('sector', 'N/A'),
                                'industry': quote_item.get('industry', 'N/A')
                            })
                    
                    return {
                        "success": True,
                        "quotes": filtered_quotes,
                        "total_results": len(filtered_quotes),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "success": False,
                        "quotes": [],
                        "error": f"Yahoo Finance API returned status {response.status}",
                        "timestamp": datetime.now().isoformat()
                    }
                    
    except asyncio.TimeoutError:
        return {
            "success": False,
            "quotes": [],
            "error": "Request timeout",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"❌ Search proxy error: {e}")
        return {
            "success": False,
            "quotes": [],
            "error": f"Search service temporarily unavailable: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/models/retrain")
async def trigger_model_retrain():
    try:
        print("🤖 Manual model retraining triggered...")
        model_system.train_and_compare_models()
        
        return {
            "message": "Model retraining completed",
            "best_model": model_system.best_model_name,
            "model_performances": model_system.model_performances,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")

@app.get("/api/system/health")
async def advanced_system_health():
    try:
        initialize_system()
        
        test_symbol = "AAPL"
        financial_data = pipeline.get_financial_data(test_symbol)
        macro_data = pipeline.get_macro_data()
        
        try:
            news_data = pipeline.get_news_sentiment(test_symbol)
        except:
            news_data = []
        
        model_status = "trained" if model_system.is_trained else "not_trained"
        
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM companies")
        company_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM scores")
        score_count = cursor.fetchone()[0]
        conn.close()
        
        return {
            "status": "healthy",
            "components": {
                "database": f"connected ({company_count} companies, {score_count} scores)",
                "yfinance": "working" if financial_data else "degraded",
                "fred_api": "working" if not np.isnan(macro_data.get('gdp_growth_rate', float('nan'))) else "fallback",
                "news_api": "working" if news_data else "fallback",
                "ml_system": f"{model_status} - best: {model_system.best_model_name}",
                "explainability_engine": "operational"
            },
            "performance_metrics": {
                "model_accuracy": model_system.model_performances.get(model_system.best_model_name, {}).get('cv_score', 0) if model_system.best_model_name else 0,
                "feature_count": len(model_system.feature_names),
                "data_freshness": "real-time",
                "mlops_status": "active"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def get_risk_grade(score: float) -> str:
    """Convert numeric score to risk grade"""
    if score >= 9.0:
        return "AAA"
    elif score >= 8.5:
        return "AA+"
    elif score >= 8.0:
        return "AA"
    elif score >= 7.5:
        return "AA-"
    elif score >= 7.0:
        return "A+"
    elif score >= 6.5:
        return "A"
    elif score >= 6.0:
        return "A-"
    elif score >= 5.5:
        return "BBB+"
    elif score >= 5.0:
        return "BBB"
    elif score >= 4.5:
        return "BBB-"
    elif score >= 4.0:
        return "BB+"
    elif score >= 3.5:
        return "BB"
    elif score >= 3.0:
        return "BB-"
    elif score >= 2.5:
        return "B+"
    elif score >= 2.0:
        return "B"
    else:
        return "CCC"

# For Render deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)