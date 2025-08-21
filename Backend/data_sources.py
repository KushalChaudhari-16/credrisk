# data_sources.py
import yfinance as yf
import requests
from fredapi import Fred
from textblob import TextBlob
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import time
import os
from dotenv import load_dotenv

load_dotenv()

class DataPipeline:
    def __init__(self):
        self.fred = Fred(api_key=os.getenv('FRED_API_KEY', 'demo_key'))
        self.news_api_key = os.getenv('NEWS_API_KEY', 'demo_key')
    
    def get_financial_data(self, symbol: str) -> Dict[str, Any]:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet  
            cash_flow = ticker.cashflow
            
            metrics = {}
            
            if not financials.empty and len(financials.columns) > 0:
                latest_financials = financials.iloc[:, 0]
                total_revenue = self.safe_get_value(latest_financials, 'Total Revenue')
                net_income = self.safe_get_value(latest_financials, 'Net Income')
                
                metrics.update({
                    'total_revenue': total_revenue,
                    'gross_profit': self.safe_get_value(latest_financials, 'Gross Profit'),
                    'operating_income': self.safe_get_value(latest_financials, 'Operating Income'),
                    'net_income': net_income,
                    'ebitda': self.safe_get_value(latest_financials, 'EBITDA')
                })
                
                if total_revenue > 0:
                    metrics['profit_margin'] = net_income / total_revenue
                else:
                    metrics['profit_margin'] = info.get('profitMargins', 0)
            
            if not balance_sheet.empty and len(balance_sheet.columns) > 0:
                latest_balance = balance_sheet.iloc[:, 0]
                
                total_debt = self.safe_get_value(latest_balance, 'Total Debt')
                if total_debt == 0:
                    short_debt = self.safe_get_value(latest_balance, 'Short Long Term Debt')
                    long_debt = self.safe_get_value(latest_balance, 'Long Term Debt')
                    total_debt = short_debt + long_debt
                
                shareholders_equity = self.safe_get_value(latest_balance, 'Stockholders Equity')
                if shareholders_equity == 0:
                    shareholders_equity = self.safe_get_value(latest_balance, 'Total Stockholder Equity')
                if shareholders_equity == 0:
                    shareholders_equity = self.safe_get_value(latest_balance, 'Common Stock Equity')
                
                current_assets = self.safe_get_value(latest_balance, 'Current Assets')
                current_liabilities = self.safe_get_value(latest_balance, 'Current Liabilities')
                
                metrics.update({
                    'total_debt': total_debt,
                    'total_equity': shareholders_equity,
                    'current_assets': current_assets,
                    'current_liabilities': current_liabilities
                })
                
                if shareholders_equity > 0:
                    metrics['debt_to_equity'] = total_debt / shareholders_equity
                else:
                    metrics['debt_to_equity'] = 5.0
                    
                if current_liabilities > 0:
                    metrics['current_ratio'] = current_assets / current_liabilities
                else:
                    metrics['current_ratio'] = 1.0
            
            if not cash_flow.empty and len(cash_flow.columns) > 0:
                latest_cashflow = cash_flow.iloc[:, 0]
                operating_cf = self.safe_get_value(latest_cashflow, 'Operating Cash Flow')
                free_cf = self.safe_get_value(latest_cashflow, 'Free Cash Flow')
                
                metrics.update({
                    'operating_cash_flow': operating_cf,
                    'free_cash_flow': free_cf
                })
            
            metrics.update({
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': self.validate_ratio(info.get('trailingPE', 0)),
                'pb_ratio': self.validate_ratio(info.get('priceToBook', 0)),
                'roe': self.validate_percentage(info.get('returnOnEquity', 0)),
                'roa': self.validate_percentage(info.get('returnOnAssets', 0)),
                'beta': max(0.1, min(3.0, info.get('beta', 1.0)))
            })
            
            if 'profit_margin' not in metrics:
                metrics['profit_margin'] = self.validate_percentage(info.get('profitMargins', 0))
            
            return {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'metrics': metrics
            }
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def safe_get_value(self, data_series, key):
        try:
            value = data_series.get(key, 0)
            if pd.isna(value) or np.isnan(value):
                return 0
            return float(value)
        except:
            return 0
    
    def validate_ratio(self, value):
        if pd.isna(value) or np.isnan(value) or value <= 0:
            return 15.0
        return min(200, max(1, float(value)))
    
    def validate_percentage(self, value):
        if pd.isna(value) or np.isnan(value):
            return 0.1
        return max(-1.0, min(1.0, float(value)))
    
    def get_macro_data(self) -> Dict[str, float]:
        try:
            gdp_data = self.fred.get_series('GDPC1', limit=8)
            if len(gdp_data) >= 2:
                gdp_growth = ((gdp_data.iloc[-1] / gdp_data.iloc[-2]) - 1) * 100
            else:
                gdp_growth = 2.5
            
            interest_rate = self.fred.get_series('FEDFUNDS', limit=1).iloc[-1]
            
            cpi_data = self.fred.get_series('CPIAUCSL', limit=12)
            if len(cpi_data) >= 12:
                inflation_rate = ((cpi_data.iloc[-1] / cpi_data.iloc[-12]) - 1) * 100
            else:
                inflation_rate = 3.2
            
            unemployment = self.fred.get_series('UNRATE', limit=1).iloc[-1]
            
            return {
                'gdp_growth_rate': float(gdp_growth),
                'federal_funds_rate': float(interest_rate),
                'inflation_rate': float(inflation_rate),
                'unemployment_rate': float(unemployment)
            }
            
        except Exception as e:
            print(f"FRED API Error: {e}")
            return {
                'gdp_growth_rate': 2.5,
                'federal_funds_rate': 5.25,
                'inflation_rate': 3.2,
                'unemployment_rate': 3.8
            }
    
    def get_news_sentiment(self, symbol: str) -> List[Dict[str, Any]]:
        try:
            company_name = self.get_company_name(symbol)
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'"{symbol}" OR "{company_name}" finance earnings debt',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 15,
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                return self.generate_mock_news(symbol)
            
            data = response.json()
            news_events = []
            
            for article in data.get('articles', [])[:10]:
                headline = article.get('title', '')
                description = article.get('description', '') or ''
                text = f"{headline} {description}"
                
                if not text.strip():
                    continue
                
                sentiment = TextBlob(text).sentiment.polarity
                event_type = self.classify_event(text)
                risk_impact = self.calculate_risk_impact(sentiment, event_type)
                
                news_events.append({
                    'headline': headline,
                    'sentiment': float(sentiment),
                    'event_type': event_type,
                    'risk_impact': float(risk_impact)
                })
            
            if not news_events:
                return self.generate_mock_news(symbol)
            
            return news_events
            
        except Exception as e:
            print(f"News API Error for {symbol}: {e}")
            return self.generate_mock_news(symbol)
    
    def generate_mock_news(self, symbol: str) -> List[Dict[str, Any]]:
        mock_headlines = [
            f"{symbol} reports quarterly earnings",
            f"{symbol} announces strategic partnership",
            f"{symbol} stock analysis by analysts",
            f"{symbol} market performance review",
            f"{symbol} financial outlook positive"
        ]
        
        return [{
            'headline': headline,
            'sentiment': np.random.normal(0.1, 0.2),
            'event_type': 'earnings',
            'risk_impact': 0.1
        } for headline in mock_headlines]
    
    def get_company_name(self, symbol: str) -> str:
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info.get('longName', symbol)
        except:
            return symbol
    
    def classify_event(self, text: str) -> str:
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['bankrupt', 'default', 'debt restructur', 'financial distress']):
            return 'debt_stress'
        elif any(word in text_lower for word in ['earnings', 'profit', 'revenue', 'quarterly']):
            return 'earnings'
        elif any(word in text_lower for word in ['acquisition', 'merger', 'deal', 'buyout']):
            return 'corporate_action'
        elif any(word in text_lower for word in ['lawsuit', 'investigation', 'scandal', 'regulatory']):
            return 'legal_risk'
        else:
            return 'general'
    
    def calculate_risk_impact(self, sentiment: float, event_type: str) -> float:
        base_impact = abs(sentiment)
        
        multipliers = {
            'debt_stress': 2.5,
            'legal_risk': 2.0,
            'earnings': 1.5,
            'corporate_action': 1.2,
            'general': 0.8
        }
        
        return base_impact * multipliers.get(event_type, 1.0)

