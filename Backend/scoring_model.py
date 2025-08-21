# scoring_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any
import pickle
import os

class CreditScoringModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
    def prepare_features(self, financial_metrics: Dict[str, float], 
                        macro_data: Dict[str, float],
                        news_sentiment: float = 0.0) -> np.ndarray:
        
        financial_features = [
            'debt_to_equity', 'current_ratio', 'roe', 'roa', 'profit_margin',
            'pe_ratio', 'pb_ratio', 'beta', 'market_cap', 'operating_cash_flow'
        ]
        
        macro_features = [
            'gdp_growth_rate', 'federal_funds_rate', 'inflation_rate', 'unemployment_rate'
        ]
        
        features = []
        feature_names = []
        
        for feature in financial_features:
            value = financial_metrics.get(feature, 0)
            if isinstance(value, (int, float)) and not np.isnan(value):
                features.append(float(value))
            else:
                features.append(0.0)
            feature_names.append(feature)
        
        for feature in macro_features:
            value = macro_data.get(feature, 0)
            if not np.isnan(value):
                features.append(float(value))
            else:
                features.append(0.0)
            feature_names.append(feature)
        
        features.append(float(news_sentiment) if not np.isnan(news_sentiment) else 0.0)
        feature_names.append('news_sentiment')
        
        self.feature_names = feature_names
        return np.array(features).reshape(1, -1)
    
    def generate_realistic_training_data(self, n_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(42)
        
        debt_to_equity = np.random.gamma(1.5, 0.8, n_samples)
        debt_to_equity = np.clip(debt_to_equity, 0.1, 8.0)
        
        current_ratio = np.random.gamma(2.5, 0.6, n_samples)
        current_ratio = np.clip(current_ratio, 0.3, 4.0)
        
        roe = np.random.normal(0.12, 0.15, n_samples)
        roe = np.clip(roe, -0.5, 0.6)
        
        roa = np.random.normal(0.06, 0.08, n_samples)
        roa = np.clip(roa, -0.3, 0.3)
        
        profit_margin = np.random.normal(0.08, 0.12, n_samples)
        profit_margin = np.clip(profit_margin, -0.4, 0.4)
        
        pe_ratio = np.random.gamma(2, 8, n_samples)
        pe_ratio = np.clip(pe_ratio, 5, 80)
        
        pb_ratio = np.random.gamma(1.5, 1.2, n_samples)
        pb_ratio = np.clip(pb_ratio, 0.5, 10)
        
        beta = np.random.normal(1.0, 0.4, n_samples)
        beta = np.clip(beta, 0.2, 2.5)
        
        market_cap = np.random.lognormal(20, 2.5, n_samples)
        operating_cash_flow = np.random.normal(5e8, 1e9, n_samples)
        
        gdp_growth = np.random.normal(2.2, 1.5, n_samples)
        interest_rate = np.random.normal(4.5, 2.0, n_samples)
        inflation = np.random.normal(3.1, 1.8, n_samples)
        unemployment = np.random.normal(4.2, 1.5, n_samples)
        
        news_sentiment = np.random.normal(0.05, 0.25, n_samples)
        
        X = np.column_stack([
            debt_to_equity, current_ratio, roe, roa, profit_margin,
            pe_ratio, pb_ratio, beta, market_cap, operating_cash_flow,
            gdp_growth, interest_rate, inflation, unemployment, news_sentiment
        ])
        
        base_score = 5.0
        debt_impact = -1.2 * np.log1p(debt_to_equity)
        liquidity_impact = 0.8 * np.log1p(current_ratio)
        profitability_impact = 8.0 * roe + 6.0 * roa + 4.0 * profit_margin
        valuation_impact = -0.02 * (pe_ratio - 20) - 0.1 * (pb_ratio - 2)
        risk_impact = -0.3 * (beta - 1.0)
        macro_impact = 0.15 * gdp_growth - 0.1 * (interest_rate - 3) - 0.05 * (inflation - 2.5)
        sentiment_impact = 2.0 * news_sentiment
        
        score = (base_score + debt_impact + liquidity_impact + profitability_impact + 
                valuation_impact + risk_impact + macro_impact + sentiment_impact)
        
        noise = np.random.normal(0, 0.3, n_samples)
        y = np.clip(score + noise, 1.0, 10.0)
        
        return X, y
    
    def train_model(self):
        X_train, y_train = self.generate_realistic_training_data()
        
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
        
        self.save_model()
        
        train_score = self.model.score(X_scaled, y_train)
        print(f"Model trained with R² score: {train_score:.3f}")
    
    def predict_score(self, financial_metrics: Dict[str, float], 
                     macro_data: Dict[str, float],
                     news_sentiment: float = 0.0) -> Tuple[float, float]:
        
        if not self.is_trained:
            self.train_model()
        
        features = self.prepare_features(financial_metrics, macro_data, news_sentiment)
        features_scaled = self.scaler.transform(features)
        
        score = self.model.predict(features_scaled)[0]
        
        trees_predictions = np.array([tree.predict(features_scaled)[0] for tree in self.model.estimators_])
        std_dev = np.std(trees_predictions)
        confidence = max(0.65, min(0.95, 1.0 - (std_dev / 3.0)))
        
        return float(np.clip(score, 1.0, 10.0)), float(confidence)
    
    def get_feature_importance(self, financial_metrics: Dict[str, float], 
                              macro_data: Dict[str, float],
                              news_sentiment: float = 0.0) -> List[Dict[str, Any]]:
        
        if not self.is_trained:
            return []
        
        features = self.prepare_features(financial_metrics, macro_data, news_sentiment)
        feature_importance = self.model.feature_importances_
        
        importance_data = []
        for i, (name, importance) in enumerate(zip(self.feature_names, feature_importance)):
            value = features[0][i] if i < len(features[0]) else 0
            impact = self.calculate_feature_impact(name, value)
            
            importance_data.append({
                'factor': name,
                'value': float(value),
                'importance': float(importance),
                'impact': float(impact),
                'description': self.get_factor_description(name, value)
            })
        
        return sorted(importance_data, key=lambda x: x['importance'], reverse=True)
    
    def calculate_feature_impact(self, feature_name: str, value: float) -> float:
        impact_rules = {
            'debt_to_equity': -0.8 if value > 3.0 else -0.3 if value > 1.5 else 0.2,
            'current_ratio': 0.4 if value > 1.8 else 0.1 if value > 1.2 else -0.4,
            'roe': 0.6 if value > 0.20 else 0.2 if value > 0.10 else -0.3,
            'roa': 0.4 if value > 0.10 else 0.1 if value > 0.05 else -0.2,
            'profit_margin': 0.5 if value > 0.15 else 0.1 if value > 0.05 else -0.3,
            'pe_ratio': -0.2 if value > 40 else 0.0 if value > 15 else -0.1,
            'news_sentiment': value * 1.5
        }
        return impact_rules.get(feature_name, 0.0)
    
    def get_factor_description(self, feature_name: str, value: float) -> str:
        descriptions = {
            'debt_to_equity': f"{'High risk' if value > 3 else 'Moderate' if value > 1.5 else 'Conservative'} leverage ({value:.2f})",
            'current_ratio': f"{'Strong' if value > 1.8 else 'Adequate' if value > 1.2 else 'Concerning'} liquidity ({value:.2f})",
            'roe': f"{'Excellent' if value > 0.20 else 'Good' if value > 0.10 else 'Weak'} profitability ({value:.1%})",
            'roa': f"{'High' if value > 0.10 else 'Average' if value > 0.05 else 'Low'} asset efficiency ({value:.1%})",
            'profit_margin': f"{'Strong' if value > 0.15 else 'Moderate' if value > 0.05 else 'Weak'} margins ({value:.1%})",
            'pe_ratio': f"{'Expensive' if value > 40 else 'Fair' if value > 15 else 'Cheap'} valuation ({value:.1f}x)",
            'news_sentiment': f"{'Positive' if value > 0.1 else 'Neutral' if value > -0.1 else 'Negative'} sentiment ({value:.2f})",
            'beta': f"{'High' if value > 1.5 else 'Moderate' if value > 0.8 else 'Low'} volatility ({value:.2f})"
        }
        return descriptions.get(feature_name, f"{feature_name}: {value:.2f}")
    
    def save_model(self):
        os.makedirs('models', exist_ok=True)
        with open('models/credit_model.pkl', 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained
            }, f)
    
    def load_model(self):
        try:
            with open('models/credit_model.pkl', 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.feature_names = data['feature_names']
                self.is_trained = data['is_trained']
                print("Model loaded successfully")
        except FileNotFoundError:
            print("No saved model found. Training new model...")
            self.train_model()

