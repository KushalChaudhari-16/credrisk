# innovation_features.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import statistics
from textblob import TextBlob

class RatingAgencyLagDetector:
    def __init__(self):
        self.agency_ratings = {
            'AAA': 10.0, 'AA+': 9.0, 'AA': 8.5, 'AA-': 8.0,
            'A+': 7.5, 'A': 7.0, 'A-': 6.5,
            'BBB+': 6.0, 'BBB': 5.5, 'BBB-': 5.0,
            'BB+': 4.5, 'BB': 4.0, 'BB-': 3.5,
            'B+': 3.0, 'B': 2.5, 'B-': 2.0,
            'CCC': 1.5
        }
        
    def detect_arbitrage_opportunities(self, our_score: float, agency_rating: str, 
                                     confidence: float) -> Dict[str, Any]:
        
        if agency_rating not in self.agency_ratings:
            return {"alert_type": "no_data", "message": "No agency rating available"}
        
        agency_score = self.agency_ratings[agency_rating]
        score_difference = our_score - agency_score
        
        if abs(score_difference) > 1.5 and confidence > 0.80:
            if score_difference > 0:
                alert_type = "UPGRADE_OPPORTUNITY"
                message = f"Our model rates the credit {abs(score_difference):.1f} points higher than {agency_rating}"
                trade_direction = "Long Credit / Short Protection"
            else:
                alert_type = "DOWNGRADE_RISK"
                message = f"Our model rates the credit {abs(score_difference):.1f} points lower than {agency_rating}"
                trade_direction = "Short Credit / Long Protection"
            
            profit_potential = self.estimate_profit_potential(abs(score_difference), confidence)
            
            return {
                "alert_type": alert_type,
                "message": message,
                "score_difference": round(score_difference, 2),
                "confidence": confidence,
                "trade_direction": trade_direction,
                "profit_potential": profit_potential,
                "risk_level": "medium" if confidence > 0.85 else "high"
            }
        
        return {"alert_type": "no_arbitrage", "message": "No significant rating divergence detected"}
    
    def estimate_profit_potential(self, score_diff: float, confidence: float) -> Dict[str, float]:
        base_spread_change = score_diff * 25
        confidence_multiplier = confidence
        
        expected_return = base_spread_change * confidence_multiplier
        risk_adjusted_return = expected_return * 0.7
        
        return {
            "expected_spread_change_bps": round(base_spread_change, 1),
            "expected_return_pct": round(expected_return / 100, 2),
            "risk_adjusted_return_pct": round(risk_adjusted_return / 100, 2)
        }

class FinancialStressContagionPredictor:
    def __init__(self, database):
        self.db = database
        self.sector_correlations = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL'],
            'Financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
            'Healthcare': ['JNJ', 'PFE', 'UNH'],
            'Energy': ['XOM', 'CVX', 'COP']
        }
        
    def calculate_contagion_risk(self, stressed_company: str, 
                               stress_score: float) -> Dict[str, Any]:
        
        company_sector = self.get_company_sector(stressed_company)
        if not company_sector:
            return {"contagion_risk": "unknown", "affected_companies": []}
        
        sector_companies = self.sector_correlations.get(company_sector, [])
        contagion_effects = []
        
        base_contagion_multiplier = max(0.1, (10 - stress_score) / 10)
        
        for company in sector_companies:
            if company != stressed_company:
                correlation_strength = np.random.uniform(0.3, 0.8)
                expected_impact = stress_score * base_contagion_multiplier * correlation_strength
                
                contagion_effects.append({
                    "company": company,
                    "expected_score_impact": round(-expected_impact, 2),
                    "correlation_strength": round(correlation_strength, 2),
                    "contagion_probability": min(0.95, base_contagion_multiplier + 0.2)
                })
        
        overall_risk = "high" if base_contagion_multiplier > 0.6 else "medium" if base_contagion_multiplier > 0.3 else "low"
        
        network_effect = {
            "stressed_company": stressed_company,
            "sector": company_sector,
            "contagion_risk_level": overall_risk,
            "affected_companies": sorted(contagion_effects, key=lambda x: abs(x['expected_score_impact']), reverse=True)[:5],
            "network_stability_score": round(10 - (base_contagion_multiplier * 10), 1),
            "monitoring_recommendation": "Increase monitoring frequency" if overall_risk in ["high", "medium"] else "Standard monitoring"
        }
        
        return network_effect
    
    def get_company_sector(self, symbol: str) -> str:
        for sector, companies in self.sector_correlations.items():
            if symbol in companies:
                return sector
        return None
    
    def generate_stress_scenario(self, base_scenario: Dict[str, float]) -> Dict[str, Any]:
        stressed_scenarios = {}
        
        for company, base_score in base_scenario.items():
            stress_multipliers = {
                "mild_stress": 0.9,
                "moderate_stress": 0.75,
                "severe_stress": 0.6
            }
            
            company_scenarios = {}
            for scenario_name, multiplier in stress_multipliers.items():
                stressed_score = base_score * multiplier
                company_scenarios[scenario_name] = {
                    "stressed_score": round(stressed_score, 2),
                    "score_change": round(stressed_score - base_score, 2),
                    "grade_change": self.calculate_grade_impact(base_score, stressed_score)
                }
            
            stressed_scenarios[company] = company_scenarios
        
        return {
            "scenario_analysis": stressed_scenarios,
            "systemic_risk_indicators": self.calculate_systemic_risk(stressed_scenarios),
            "generated_at": datetime.now().isoformat()
        }
    
    def calculate_grade_impact(self, original_score: float, stressed_score: float) -> str:
        grade_diff = abs(original_score - stressed_score) // 0.5
        if grade_diff >= 2:
            return "multi_notch_downgrade"
        elif grade_diff >= 1:
            return "single_notch_downgrade"
        else:
            return "stable"
    
    def calculate_systemic_risk(self, scenarios: Dict) -> Dict[str, Any]:
        total_companies = len(scenarios)
        severe_stress_count = sum(1 for company_data in scenarios.values() 
                                if company_data['severe_stress']['stressed_score'] < 4.0)
        
        systemic_risk_ratio = severe_stress_count / total_companies if total_companies > 0 else 0
        
        return {
            "systemic_risk_level": "high" if systemic_risk_ratio > 0.4 else "medium" if systemic_risk_ratio > 0.2 else "low",
            "companies_at_risk": severe_stress_count,
            "total_companies": total_companies,
            "risk_concentration": round(systemic_risk_ratio, 2)
        }

class EarningsCallSentimentDivergenceDetector:
    def __init__(self):
        self.sentiment_thresholds = {
            'very_positive': 0.5,
            'positive': 0.1,
            'neutral': 0.0,
            'negative': -0.1,
            'very_negative': -0.5
        }
    
    def analyze_sentiment_divergence(self, earnings_text: str, 
                                   financial_metrics: Dict[str, float]) -> Dict[str, Any]:
        
        if not earnings_text:
            return {"divergence_detected": False, "reason": "No earnings data available"}
        
        management_sentiment = self.extract_management_sentiment(earnings_text)
        financial_performance = self.calculate_financial_performance_score(financial_metrics)
        
        sentiment_score = management_sentiment['compound_sentiment']
        performance_score = financial_performance['normalized_score']
        
        divergence_magnitude = abs(sentiment_score - performance_score)
        
        if divergence_magnitude > 0.4:
            divergence_type = "positive_bias" if sentiment_score > performance_score else "negative_bias"
            
            warning_level = "high" if divergence_magnitude > 0.7 else "medium"
            
            return {
                "divergence_detected": True,
                "divergence_type": divergence_type,
                "divergence_magnitude": round(divergence_magnitude, 3),
                "warning_level": warning_level,
                "management_sentiment": management_sentiment,
                "financial_performance": financial_performance,
                "early_warning_signal": self.generate_early_warning(divergence_type, divergence_magnitude),
                "recommended_action": self.recommend_action(divergence_type, warning_level)
            }
        
        return {
            "divergence_detected": False,
            "sentiment_alignment": "good",
            "management_sentiment": management_sentiment,
            "financial_performance": financial_performance
        }
    
    def extract_management_sentiment(self, text: str) -> Dict[str, Any]:
        blob = TextBlob(text)
        
        sentences = blob.sentences
        sentiment_scores = [sentence.sentiment.polarity for sentence in sentences]
        
        positive_phrases = ['strong growth', 'excellent performance', 'optimistic', 'confident', 'robust']
        negative_phrases = ['challenging', 'difficult', 'headwinds', 'uncertainty', 'weakness']
        
        positive_count = sum(1 for phrase in positive_phrases if phrase in text.lower())
        negative_count = sum(1 for phrase in negative_phrases if phrase in text.lower())
        
        compound_sentiment = statistics.mean(sentiment_scores) if sentiment_scores else 0.0
        
        return {
            "compound_sentiment": round(compound_sentiment, 3),
            "positive_indicators": positive_count,
            "negative_indicators": negative_count,
            "sentiment_consistency": round(1 - statistics.stdev(sentiment_scores), 3) if len(sentiment_scores) > 1 else 1.0,
            "dominant_tone": self.classify_sentiment(compound_sentiment)
        }
    
    def calculate_financial_performance_score(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        key_metrics = {
            'revenue_growth': metrics.get('revenue_growth', 0),
            'profit_margin': metrics.get('profit_margin', 0),
            'roe': metrics.get('roe', 0),
            'eps_growth': metrics.get('eps_growth', 0)
        }
        
        performance_indicators = []
        for metric, value in key_metrics.items():
            if metric == 'revenue_growth':
                score = 1.0 if value > 0.10 else 0.5 if value > 0.03 else -0.5 if value < -0.05 else 0.0
            elif metric == 'profit_margin':
                score = 1.0 if value > 0.15 else 0.5 if value > 0.05 else -0.5 if value < 0 else 0.0
            elif metric == 'roe':
                score = 1.0 if value > 0.15 else 0.5 if value > 0.08 else -0.5 if value < 0.03 else 0.0
            elif metric == 'eps_growth':
                score = 1.0 if value > 0.15 else 0.5 if value > 0.05 else -0.5 if value < -0.10 else 0.0
            else:
                score = 0.0
            
            performance_indicators.append(score)
        
        normalized_score = statistics.mean(performance_indicators)
        
        return {
            "normalized_score": round(normalized_score, 3),
            "individual_metrics": key_metrics,
            "performance_level": self.classify_performance(normalized_score)
        }
    
    def classify_sentiment(self, score: float) -> str:
        if score > 0.5: return "very_positive"
        elif score > 0.1: return "positive"
        elif score > -0.1: return "neutral"
        elif score > -0.5: return "negative"
        else: return "very_negative"
    
    def classify_performance(self, score: float) -> str:
        if score > 0.7: return "excellent"
        elif score > 0.3: return "good"
        elif score > -0.3: return "average"
        elif score > -0.7: return "poor"
        else: return "very_poor"
    
    def generate_early_warning(self, divergence_type: str, magnitude: float) -> Dict[str, str]:
        if divergence_type == "positive_bias":
            return {
                "signal_type": "management_overconfidence",
                "description": "Management sentiment appears overly optimistic relative to financial performance",
                "implication": "Potential for disappointing future results or guidance revisions"
            }
        else:
            return {
                "signal_type": "management_pessimism",
                "description": "Management tone is more negative than financial metrics suggest",
                "implication": "Possible conservative guidance or hidden operational challenges"
            }
    
    def recommend_action(self, divergence_type: str, warning_level: str) -> str:
        if warning_level == "high":
            return f"Immediate review recommended - significant {divergence_type} detected"
        else:
            return f"Monitor closely - moderate {divergence_type} requires attention"

