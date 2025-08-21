# multi_model_system.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from typing import Dict, List, Tuple, Any, Optional
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MultiModelCreditScoring:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=150, random_state=42, max_depth=12),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=8),
            'logistic': LogisticRegression(random_state=42, max_iter=1000)
        }
        self.scalers = {name: StandardScaler() for name in self.models.keys()}
        self.feature_names = []
        self.best_model_name = None
        self.model_performances = {}
        self.is_trained = False
        
    def prepare_features(self, financial_metrics: Dict[str, float], 
                        macro_data: Dict[str, float],
                        news_sentiment: float = 0.0) -> np.ndarray:
        
        financial_features = [
            'debt_to_equity', 'current_ratio', 'roe', 'roa', 'profit_margin',
            'pe_ratio', 'pb_ratio', 'beta', 'market_cap', 'operating_cash_flow',
            'free_cash_flow', 'total_debt', 'total_equity'
        ]
        
        macro_features = [
            'gdp_growth_rate', 'federal_funds_rate', 'inflation_rate', 'unemployment_rate'
        ]
        
        features = []
        feature_names = []
        
        for feature in financial_features:
            value = financial_metrics.get(feature, 0)
            if isinstance(value, (int, float)) and not np.isnan(value) and np.isfinite(value):
                features.append(float(value))
            else:
                features.append(0.0)
            feature_names.append(feature)
        
        for feature in macro_features:
            value = macro_data.get(feature, 0)
            if isinstance(value, (int, float)) and not np.isnan(value) and np.isfinite(value):
                features.append(float(value))
            else:
                features.append(0.0)
            feature_names.append(feature)
        
        sentiment = news_sentiment if isinstance(news_sentiment, (int, float)) and not np.isnan(news_sentiment) else 0.0
        features.append(float(sentiment))
        feature_names.append('news_sentiment')
        
        self.feature_names = feature_names
        return np.array(features).reshape(1, -1)
    
    def generate_enhanced_training_data(self, n_samples: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(42)
        
        debt_to_equity = np.random.gamma(1.8, 0.9, n_samples)
        debt_to_equity = np.clip(debt_to_equity, 0.05, 10.0)
        
        current_ratio = np.random.gamma(2.2, 0.7, n_samples)
        current_ratio = np.clip(current_ratio, 0.2, 5.0)
        
        roe = np.random.normal(0.14, 0.18, n_samples)
        roe = np.clip(roe, -0.6, 0.7)
        
        roa = np.random.normal(0.07, 0.09, n_samples)
        roa = np.clip(roa, -0.4, 0.4)
        
        profit_margin = np.random.normal(0.09, 0.14, n_samples)
        profit_margin = np.clip(profit_margin, -0.5, 0.5)
        
        pe_ratio = np.random.gamma(2.5, 9, n_samples)
        pe_ratio = np.clip(pe_ratio, 3, 100)
        
        pb_ratio = np.random.gamma(1.8, 1.4, n_samples)
        pb_ratio = np.clip(pb_ratio, 0.3, 12)
        
        beta = np.random.normal(1.1, 0.5, n_samples)
        beta = np.clip(beta, 0.1, 3.0)
        
        market_cap = np.random.lognormal(21, 2.8, n_samples)
        operating_cash_flow = np.random.normal(8e8, 1.5e9, n_samples)
        free_cash_flow = np.random.normal(6e8, 1.2e9, n_samples)
        
        total_debt = np.random.lognormal(22, 1.5, n_samples)
        total_equity = np.random.lognormal(23, 1.8, n_samples)
        
        gdp_growth = np.random.normal(2.4, 1.8, n_samples)
        interest_rate = np.random.normal(4.8, 2.2, n_samples)
        inflation = np.random.normal(3.3, 2.0, n_samples)
        unemployment = np.random.normal(4.1, 1.8, n_samples)
        
        news_sentiment = np.random.normal(0.03, 0.28, n_samples)
        
        X = np.column_stack([
            debt_to_equity, current_ratio, roe, roa, profit_margin,
            pe_ratio, pb_ratio, beta, market_cap, operating_cash_flow,
            free_cash_flow, total_debt, total_equity,
            gdp_growth, interest_rate, inflation, unemployment, news_sentiment
        ])
        
        base_score = 5.2
        debt_impact = -1.4 * np.log1p(debt_to_equity)
        liquidity_impact = 0.9 * np.log1p(current_ratio)
        profitability_impact = 9.0 * roe + 7.0 * roa + 5.0 * profit_margin
        valuation_impact = -0.025 * (pe_ratio - 18) - 0.12 * (pb_ratio - 1.8)
        risk_impact = -0.35 * (beta - 1.0)
        size_impact = 0.3 * np.log1p(market_cap / 1e9)
        cashflow_impact = 0.8 * np.log1p(np.maximum(operating_cash_flow, 1e6) / 1e9)
        macro_impact = 0.18 * gdp_growth - 0.12 * (interest_rate - 3.5) - 0.08 * (inflation - 2.8)
        sentiment_impact = 2.2 * news_sentiment
        
        score = (base_score + debt_impact + liquidity_impact + profitability_impact + 
                valuation_impact + risk_impact + size_impact + cashflow_impact + 
                macro_impact + sentiment_impact)
        
        noise = np.random.normal(0, 0.25, n_samples)
        y = np.clip(score + noise, 0.5, 10.0)
        
        return X, y
    
    def train_and_compare_models(self):
        X_train, y_train = self.generate_enhanced_training_data()
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            if name == 'logistic':
                y_binary = (y_train > 6.0).astype(int)
                X_scaled = self.scalers[name].fit_transform(X_train)
                model.fit(X_scaled, y_binary)
                cv_scores = cross_val_score(model, X_scaled, y_binary, cv=tscv, scoring='accuracy')
                self.model_performances[name] = {
                    'cv_score': np.mean(cv_scores),
                    'cv_std': np.std(cv_scores),
                    'score_type': 'accuracy'
                }
            else:
                X_scaled = self.scalers[name].fit_transform(X_train)
                model.fit(X_scaled, y_train)
                cv_scores = cross_val_score(model, X_scaled, y_train, cv=tscv, scoring='r2')
                
                y_pred = model.predict(X_scaled)
                mse = mean_squared_error(y_train, y_pred)
                mae = mean_absolute_error(y_train, y_pred)
                
                self.model_performances[name] = {
                    'cv_score': np.mean(cv_scores),
                    'cv_std': np.std(cv_scores),
                    'mse': mse,
                    'mae': mae,
                    'score_type': 'r2'
                }
            
            print(f"{name} CV Score: {self.model_performances[name]['cv_score']:.4f} ± {self.model_performances[name]['cv_std']:.4f}")
        
        regression_models = {k: v for k, v in self.model_performances.items() if k != 'logistic'}
        self.best_model_name = max(regression_models.keys(), key=lambda k: regression_models[k]['cv_score'])
        
        print(f"\nBest Model: {self.best_model_name} with CV Score: {self.model_performances[self.best_model_name]['cv_score']:.4f}")
        
        self.is_trained = True
        self.save_models()
    
    def predict_score(self, financial_metrics: Dict[str, float], 
                     macro_data: Dict[str, float],
                     news_sentiment: float = 0.0) -> Tuple[float, float, Dict]:
        
        if not self.is_trained:
            self.train_and_compare_models()
        
        features = self.prepare_features(financial_metrics, macro_data, news_sentiment)
        
        best_model = self.models[self.best_model_name]
        best_scaler = self.scalers[self.best_model_name]
        
        features_scaled = best_scaler.transform(features)
        
        if hasattr(best_model, 'predict'):
            score = best_model.predict(features_scaled)[0]
        else:
            score = 5.0
        
        if hasattr(best_model, 'estimators_'):
            trees_predictions = np.array([tree.predict(features_scaled)[0] for tree in best_model.estimators_])
            std_dev = np.std(trees_predictions)
            confidence = max(0.7, min(0.98, 1.0 - (std_dev / 2.5)))
        else:
            confidence = 0.85
        
        ensemble_predictions = {}
        for name, model in self.models.items():
            if name != 'logistic':
                try:
                    scaler = self.scalers[name]
                    scaled_features = scaler.transform(features)
                    pred = model.predict(scaled_features)[0]
                    ensemble_predictions[name] = float(np.clip(pred, 0.5, 10.0))
                except:
                    ensemble_predictions[name] = 5.0
        
        return float(np.clip(score, 0.5, 10.0)), float(confidence), ensemble_predictions
    
    def get_feature_importance(self, financial_metrics: Dict[str, float], 
                              macro_data: Dict[str, float],
                              news_sentiment: float = 0.0) -> List[Dict[str, Any]]:
        
        if not self.is_trained or not self.best_model_name:
            return []
        
        features = self.prepare_features(financial_metrics, macro_data, news_sentiment)
        best_model = self.models[self.best_model_name]
        
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = best_model.feature_importances_
        elif hasattr(best_model, 'coef_'):
            feature_importance = np.abs(best_model.coef_[0])
        else:
            feature_importance = np.ones(len(self.feature_names)) / len(self.feature_names)
        
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
        if not isinstance(value, (int, float)) or np.isnan(value):
            return 0.0
            
        impact_rules = {
            'debt_to_equity': -0.9 if value > 3.5 else -0.4 if value > 1.8 else 0.3,
            'current_ratio': 0.5 if value > 2.0 else 0.2 if value > 1.3 else -0.5,
            'roe': 0.7 if value > 0.25 else 0.3 if value > 0.12 else -0.4,
            'roa': 0.5 if value > 0.12 else 0.2 if value > 0.06 else -0.3,
            'profit_margin': 0.6 if value > 0.18 else 0.2 if value > 0.08 else -0.4,
            'pe_ratio': -0.3 if value > 50 else 0.1 if value < 12 else 0.0,
            'pb_ratio': -0.2 if value > 5 else 0.1 if value < 1 else 0.0,
            'beta': -0.4 if value > 1.8 else 0.2 if value < 0.7 else 0.0,
            'news_sentiment': value * 1.8
        }
        return impact_rules.get(feature_name, 0.0)
    
    def get_factor_description(self, feature_name: str, value: float) -> str:
        if not isinstance(value, (int, float)) or np.isnan(value):
            return f"{feature_name}: N/A"
            
        descriptions = {
            'debt_to_equity': f"{'Very High' if value > 5 else 'High' if value > 3 else 'Moderate' if value > 1.5 else 'Conservative'} leverage ({value:.2f})",
            'current_ratio': f"{'Excellent' if value > 2.5 else 'Strong' if value > 1.8 else 'Adequate' if value > 1.2 else 'Weak'} liquidity ({value:.2f})",
            'roe': f"{'Outstanding' if value > 0.25 else 'Excellent' if value > 0.18 else 'Good' if value > 0.12 else 'Average' if value > 0.08 else 'Weak'} profitability ({value:.1%})",
            'roa': f"{'Excellent' if value > 0.15 else 'High' if value > 0.10 else 'Average' if value > 0.05 else 'Low'} asset efficiency ({value:.1%})",
            'profit_margin': f"{'Exceptional' if value > 0.20 else 'Strong' if value > 0.15 else 'Moderate' if value > 0.08 else 'Weak'} margins ({value:.1%})",
            'pe_ratio': f"{'Overvalued' if value > 40 else 'Expensive' if value > 25 else 'Fair' if value > 15 else 'Undervalued'} ({value:.1f}x)",
            'pb_ratio': f"{'Premium' if value > 3 else 'Fair' if value > 1.5 else 'Discount'} valuation ({value:.2f}x)",
            'news_sentiment': f"{'Very Positive' if value > 0.3 else 'Positive' if value > 0.1 else 'Neutral' if value > -0.1 else 'Negative' if value > -0.3 else 'Very Negative'} ({value:.2f})",
            'beta': f"{'Very High' if value > 2.0 else 'High' if value > 1.5 else 'Moderate' if value > 0.8 else 'Low'} volatility ({value:.2f})",
            'market_cap': f"Market cap ${value/1e9:.1f}B" if value > 1e9 else f"${value/1e6:.1f}M"
        }
        return descriptions.get(feature_name, f"{feature_name}: {value:.2f}")
    
    def save_models(self):
        os.makedirs('models', exist_ok=True)
        with open('models/multi_model_system.pkl', 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers,
                'feature_names': self.feature_names,
                'best_model_name': self.best_model_name,
                'model_performances': self.model_performances,
                'is_trained': self.is_trained
            }, f)
    
    def load_models(self):
        try:
            with open('models/multi_model_system.pkl', 'rb') as f:
                data = pickle.load(f)
                self.models = data['models']
                self.scalers = data['scalers']
                self.feature_names = data['feature_names']
                self.best_model_name = data['best_model_name']
                self.model_performances = data['model_performances']
                self.is_trained = data['is_trained']
                print("Multi-model system loaded successfully")
        except FileNotFoundError:
            print("No saved models found. Training new multi-model system...")
            self.train_and_compare_models()
