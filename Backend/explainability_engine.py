
# explainability_engine.py
from typing import Dict, List, Any, Tuple
import numpy as np
from datetime import datetime, timedelta
import sqlite3

class ExplainabilityEngine:
    def __init__(self, database):
        self.db = database
        
    def generate_detailed_explanation(self, symbol: str, score: float, 
                                    feature_importance: List[Dict], 
                                    financial_metrics: Dict[str, float],
                                    macro_data: Dict[str, float],
                                    news_sentiment: float,
                                    ensemble_predictions: Dict[str, float]) -> Dict[str, Any]:
        
        explanation = {
            'executive_summary': self.generate_executive_summary(symbol, score, feature_importance[:3]),
            'feature_breakdown': self.analyze_feature_contributions(feature_importance),
            'trend_analysis': self.analyze_trends(symbol, score),
            'risk_assessment': self.assess_risk_factors(feature_importance, financial_metrics),
            'comparative_analysis': self.compare_with_peers(symbol, score, financial_metrics),
            'model_consensus': self.analyze_model_consensus(ensemble_predictions),
            'confidence_factors': self.explain_confidence_factors(feature_importance),
            'action_items': self.generate_action_items(feature_importance, financial_metrics)
        }
        
        return explanation
    
    def generate_executive_summary(self, symbol: str, score: float, 
                                 top_factors: List[Dict]) -> str:
        
        grade = self.get_risk_grade(score)
        risk_level = "low" if score >= 7.0 else "moderate" if score >= 5.0 else "high"
        
        key_drivers = []
        for factor in top_factors:
            impact_direction = "supporting" if factor['impact'] > 0 else "constraining"
            key_drivers.append(f"{factor['factor'].replace('_', ' ')} ({impact_direction})")
        
        drivers_text = ", ".join(key_drivers[:2])
        
        summary = f"""
        {symbol} receives a credit score of {score:.1f}/10 ({grade}), indicating {risk_level} credit risk. 
        The assessment is primarily driven by {drivers_text}. 
        This score reflects the company's current financial health, market position, and macroeconomic environment.
        """
        
        return summary.strip()
    
    def analyze_feature_contributions(self, feature_importance: List[Dict]) -> Dict[str, Any]:
        positive_factors = [f for f in feature_importance if f['impact'] > 0.1]
        negative_factors = [f for f in feature_importance if f['impact'] < -0.1]
        neutral_factors = [f for f in feature_importance if -0.1 <= f['impact'] <= 0.1]
        
        return {
            'strengths': {
                'count': len(positive_factors),
                'factors': positive_factors[:5],
                'total_positive_impact': sum(f['impact'] for f in positive_factors)
            },
            'weaknesses': {
                'count': len(negative_factors),
                'factors': negative_factors[:5],
                'total_negative_impact': sum(f['impact'] for f in negative_factors)
            },
            'neutral': {
                'count': len(neutral_factors),
                'factors': neutral_factors[:3]
            }
        }
    
    def analyze_trends(self, symbol: str, current_score: float) -> Dict[str, Any]:
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT score, timestamp FROM scores 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT 10
            """, (symbol,))
            
            historical_scores = cursor.fetchall()
            conn.close()
            
            if len(historical_scores) < 2:
                return {
                    'trend_direction': 'insufficient_data',
                    'trend_magnitude': 0,
                    'volatility': 0,
                    'data_points': len(historical_scores)
                }
            
            scores = [row[0] for row in historical_scores]
            recent_trend = scores[0] - scores[-1] if len(scores) > 1 else 0
            volatility = np.std(scores) if len(scores) > 2 else 0
            
            trend_direction = 'improving' if recent_trend > 0.2 else 'declining' if recent_trend < -0.2 else 'stable'
            
            return {
                'trend_direction': trend_direction,
                'trend_magnitude': abs(recent_trend),
                'volatility': volatility,
                'short_term_avg': np.mean(scores[:3]) if len(scores) >= 3 else current_score,
                'long_term_avg': np.mean(scores) if len(scores) > 5 else current_score,
                'data_points': len(scores)
            }
            
        except Exception as e:
            return {
                'trend_direction': 'error',
                'trend_magnitude': 0,
                'volatility': 0,
                'data_points': 0,
                'error': str(e)
            }
    
    def assess_risk_factors(self, feature_importance: List[Dict], 
                          financial_metrics: Dict[str, float]) -> Dict[str, Any]:
        
        risk_factors = []
        
        debt_to_equity = financial_metrics.get('debt_to_equity', 0)
        current_ratio = financial_metrics.get('current_ratio', 1)
        roe = financial_metrics.get('roe', 0)
        
        if debt_to_equity > 3.0:
            risk_factors.append({
                'factor': 'High Leverage',
                'severity': 'high',
                'description': f'Debt-to-equity ratio of {debt_to_equity:.2f} indicates excessive leverage',
                'impact': 'Increases default risk and interest burden'
            })
        
        if current_ratio < 1.2:
            risk_factors.append({
                'factor': 'Liquidity Concerns',
                'severity': 'medium' if current_ratio > 0.8 else 'high',
                'description': f'Current ratio of {current_ratio:.2f} suggests potential liquidity issues',
                'impact': 'May struggle to meet short-term obligations'
            })
        
        if roe < 0.05:
            risk_factors.append({
                'factor': 'Poor Profitability',
                'severity': 'high' if roe < 0 else 'medium',
                'description': f'ROE of {roe:.1%} indicates weak profit generation',
                'impact': 'Limited ability to generate returns for shareholders'
            })
        
        return {
            'high_risk_factors': [r for r in risk_factors if r['severity'] == 'high'],
            'medium_risk_factors': [r for r in risk_factors if r['severity'] == 'medium'],
            'overall_risk_level': self.calculate_overall_risk(risk_factors),
            'risk_score': self.calculate_risk_score(financial_metrics)
        }
    
    def compare_with_peers(self, symbol: str, score: float, 
                          financial_metrics: Dict[str, float]) -> Dict[str, Any]:
        
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT c.symbol, s.score, c.sector 
                FROM companies c 
                JOIN scores s ON c.symbol = s.symbol 
                WHERE c.sector = (SELECT sector FROM companies WHERE symbol = ?) 
                AND c.symbol != ?
                ORDER BY s.timestamp DESC
            """, (symbol, symbol))
            
            peer_data = cursor.fetchall()
            conn.close()
            
            if not peer_data:
                return {'comparison_available': False}
            
            peer_scores = [row[1] for row in peer_data]
            sector_avg = np.mean(peer_scores) if peer_scores else 5.0
            percentile = (sum(1 for s in peer_scores if s < score) / len(peer_scores)) * 100 if peer_scores else 50
            
            return {
                'comparison_available': True,
                'sector_average': sector_avg,
                'percentile_rank': percentile,
                'outperformance': score - sector_avg,
                'peer_count': len(peer_data),
                'ranking': f"Top {100-percentile:.0f}%" if percentile > 75 else f"Bottom {percentile:.0f}%" if percentile < 25 else "Middle tier"
            }
            
        except Exception as e:
            return {'comparison_available': False, 'error': str(e)}
    
    def analyze_model_consensus(self, ensemble_predictions: Dict[str, float]) -> Dict[str, Any]:
        if not ensemble_predictions:
            return {'consensus_available': False}
        
        scores = list(ensemble_predictions.values())
        consensus_score = np.mean(scores)
        disagreement = np.std(scores)
        
        model_agreement = 'high' if disagreement < 0.3 else 'medium' if disagreement < 0.7 else 'low'
        
        return {
            'consensus_available': True,
            'consensus_score': consensus_score,
            'model_disagreement': disagreement,
            'agreement_level': model_agreement,
            'individual_predictions': ensemble_predictions,
            'range': max(scores) - min(scores) if scores else 0
        }
    
    def explain_confidence_factors(self, feature_importance: List[Dict]) -> Dict[str, Any]:
        data_quality_score = self.calculate_data_quality_score(feature_importance)
        feature_stability = self.calculate_feature_stability(feature_importance)
        
        confidence_factors = []
        
        if data_quality_score > 0.8:
            confidence_factors.append("High data completeness and quality")
        elif data_quality_score < 0.5:
            confidence_factors.append("Limited data availability affects confidence")
        
        if feature_stability > 0.7:
            confidence_factors.append("Consistent feature patterns increase reliability")
        
        non_zero_features = len([f for f in feature_importance if f['value'] != 0])
        if non_zero_features > 10:
            confidence_factors.append("Rich feature set supports robust prediction")
        
        return {
            'data_quality_score': data_quality_score,
            'feature_stability': feature_stability,
            'confidence_drivers': confidence_factors,
            'overall_confidence_level': 'high' if data_quality_score > 0.7 and feature_stability > 0.6 else 'medium'
        }
    
    def generate_action_items(self, feature_importance: List[Dict], 
                            financial_metrics: Dict[str, float]) -> List[Dict[str, str]]:
        
        action_items = []
        
        debt_to_equity = financial_metrics.get('debt_to_equity', 0)
        current_ratio = financial_metrics.get('current_ratio', 1)
        roe = financial_metrics.get('roe', 0)
        
        if debt_to_equity > 2.5:
            action_items.append({
                'priority': 'high',
                'category': 'debt_management',
                'action': 'Consider debt reduction strategies or refinancing',
                'rationale': 'High leverage increases financial risk'
            })
        
        if current_ratio < 1.3:
            action_items.append({
                'priority': 'medium',
                'category': 'liquidity_management',
                'action': 'Improve working capital management',
                'rationale': 'Strengthen short-term financial flexibility'
            })
        
        if roe < 0.10:
            action_items.append({
                'priority': 'high',
                'category': 'profitability',
                'action': 'Focus on operational efficiency and margin improvement',
                'rationale': 'Enhance return on equity to attract investors'
            })
        
        negative_sentiment = any(f['factor'] == 'news_sentiment' and f['value'] < -0.1 
                                for f in feature_importance)
        
        if negative_sentiment:
            action_items.append({
                'priority': 'medium',
                'category': 'reputation_management',
                'action': 'Address negative market sentiment through improved communication',
                'rationale': 'Negative sentiment may impact credit access and costs'
            })
        
        return action_items
    
    def calculate_data_quality_score(self, feature_importance: List[Dict]) -> float:
        if not feature_importance:
            return 0.0
        
        non_zero_features = len([f for f in feature_importance if f['value'] != 0])
        total_features = len(feature_importance)
        
        return min(1.0, non_zero_features / total_features)
    
    def calculate_feature_stability(self, feature_importance: List[Dict]) -> float:
        if not feature_importance:
            return 0.0
        
        stability_score = 0.8
        
        for feature in feature_importance[:5]:
            if abs(feature['impact']) > 1.0:
                stability_score -= 0.1
        
        return max(0.0, min(1.0, stability_score))
    
    def calculate_overall_risk(self, risk_factors: List[Dict]) -> str:
        high_risk_count = len([r for r in risk_factors if r['severity'] == 'high'])
        medium_risk_count = len([r for r in risk_factors if r['severity'] == 'medium'])
        
        if high_risk_count >= 2:
            return 'high'
        elif high_risk_count == 1 or medium_risk_count >= 3:
            return 'medium'
        else:
            return 'low'
    
    def calculate_risk_score(self, financial_metrics: Dict[str, float]) -> float:
        base_risk = 5.0
        
        debt_to_equity = financial_metrics.get('debt_to_equity', 1)
        current_ratio = financial_metrics.get('current_ratio', 1)
        roe = financial_metrics.get('roe', 0.1)
        
        risk_adjustments = 0
        if debt_to_equity > 3:
            risk_adjustments += 2
        elif debt_to_equity > 2:
            risk_adjustments += 1
        
        if current_ratio < 1:
            risk_adjustments += 1.5
        elif current_ratio < 1.2:
            risk_adjustments += 0.5
        
        if roe < 0:
            risk_adjustments += 2
        elif roe < 0.05:
            risk_adjustments += 1
        
        return max(1.0, min(10.0, base_risk + risk_adjustments))
    
    def get_risk_grade(self, score: float) -> str:
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