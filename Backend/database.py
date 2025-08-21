# database.py
import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

class Database:
    def __init__(self, db_path: str = "credtech.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                symbol TEXT PRIMARY KEY,
                name TEXT,
                sector TEXT,
                market_cap REAL,
                last_updated TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                score REAL,
                model_name TEXT,
                confidence REAL,
                timestamp TEXT,
                FOREIGN KEY(symbol) REFERENCES companies(symbol)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS financials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                metric_name TEXT,
                value REAL,
                source TEXT,
                period TEXT,
                timestamp TEXT,
                FOREIGN KEY(symbol) REFERENCES companies(symbol)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                headline TEXT,
                sentiment REAL,
                event_type TEXT,
                risk_impact REAL,
                timestamp TEXT,
                FOREIGN KEY(symbol) REFERENCES companies(symbol)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def insert_company(self, symbol: str, name: str, sector: str, market_cap: float):
        conn = self.get_connection()
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT OR REPLACE INTO companies 
            (symbol, name, sector, market_cap, last_updated)
            VALUES (?, ?, ?, ?, ?)
        """, (symbol, name, sector, market_cap, timestamp))
        
        conn.commit()
        conn.close()
    
    def insert_financial_data(self, symbol: str, metrics: Dict[str, Any], source: str, period: str):
        conn = self.get_connection()
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        
        for metric_name, value in metrics.items():
            if value is not None and not (isinstance(value, float) and str(value) == 'nan'):
                try:
                    cursor.execute("""
                        INSERT INTO financials 
                        (symbol, metric_name, value, source, period, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (symbol, metric_name, float(value), source, period, timestamp))
                except (ValueError, TypeError):
                    continue
        
        conn.commit()
        conn.close()
    
    def insert_news_event(self, symbol: str, headline: str, sentiment: float, 
                         event_type: str, risk_impact: float):
        conn = self.get_connection()
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO news_events 
            (symbol, headline, sentiment, event_type, risk_impact, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (symbol, headline, sentiment, event_type, risk_impact, timestamp))
        
        conn.commit()
        conn.close()
    
    def get_latest_financials(self, symbol: str) -> Dict[str, float]:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT metric_name, value FROM financials 
            WHERE symbol = ? 
            ORDER BY timestamp DESC
        """, (symbol,))
        
        results = cursor.fetchall()
        conn.close()
        
        return {metric: value for metric, value in results}

