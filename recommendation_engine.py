import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

class RecommendationEngine:
    def __init__(self, tidb_client):
        """Initialize the recommendation engine with TiDB client."""
        self.tidb_client = tidb_client
        self.logger = logging.getLogger(__name__)
        
        # Weight configurations for different factors
        self.weights = {
            'cost': 0.25,
            'delivery_performance': 0.30,
            'quality': 0.20,
            'lead_time': 0.15,
            'risk': 0.10
        }
    
    def calculate_supplier_score(self, supplier_data, performance_data):
        """Calculate comprehensive supplier score based on multiple factors."""
        try:
            # Base supplier metrics
            rating = float(supplier_data.get('rating', 0))
            lead_time = float(supplier_data.get('lead_time_days', 0))
            price = float(supplier_data.get('price_per_unit', 0))
            
            # Performance metrics
            on_time_delivery = float(performance_data.get('on_time_delivery_rate', 0))
            quality_score = float(performance_data.get('quality_score', 0))
            communication = float(performance_data.get('communication_rating', 0))
            defect_rate = float(performance_data.get('defect_rate', 0))
            cost_variance = float(performance_data.get('cost_variance_percent', 0))
            
            # Normalize metrics to 0-1 scale
            normalized_rating = rating / 5.0
            normalized_lead_time = max(0, 1 - (lead_time / 60))  # Prefer shorter lead times
            normalized_price = max(0, 1 - (price / 100))  # Prefer lower prices
            normalized_defect_rate = max(0, 1 - (defect_rate * 100))  # Prefer lower defect rates
            normalized_cost_variance = max(0, 1 - (cost_variance / 10))  # Prefer lower variance
            
            # Calculate weighted score
            score = (
                self.weights['cost'] * normalized_price +
                self.weights['delivery_performance'] * on_time_delivery +
                self.weights['quality'] * (quality_score / 5.0) +
                self.weights['lead_time'] * normalized_lead_time +
                self.weights['risk'] * (1 - normalized_defect_rate - normalized_cost_variance) / 2
            )
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            self.logger.error(f"Error calculating supplier score: {e}")
            return 0.0
    
    def calculate_risk_score(self, supplier_data, performance_data, historical_data):
        """Calculate risk score for a supplier."""
        try:
            risk_factors = []
            
            # Contract risk
            contract_end = supplier_data.get('contract_end_date')
            if contract_end:
                days_to_expiry = (datetime.strptime(contract_end, '%Y-%m-%d') - datetime.now()).days
                if days_to_expiry < 30:
                    risk_factors.append(0.8)  # High risk
                elif days_to_expiry < 90:
                    risk_factors.append(0.4)  # Medium risk
                else:
                    risk_factors.append(0.1)  # Low risk
            
            # Performance risk
            on_time_delivery = float(performance_data.get('on_time_delivery_rate', 0))
            if on_time_delivery < 0.8:
                risk_factors.append(0.6)
            elif on_time_delivery < 0.9:
                risk_factors.append(0.3)
            else:
                risk_factors.append(0.1)
            
            # Quality risk
            defect_rate = float(performance_data.get('defect_rate', 0))
            if defect_rate > 0.05:
                risk_factors.append(0.7)
            elif defect_rate > 0.02:
                risk_factors.append(0.4)
            else:
                risk_factors.append(0.1)
            
            # Financial risk (cost variance)
            cost_variance = float(performance_data.get('cost_variance_percent', 0))
            if cost_variance > 5:
                risk_factors.append(0.6)
            elif cost_variance > 2:
                risk_factors.append(0.3)
            else:
                risk_factors.append(0.1)
            
            # Location risk
            location = supplier_data.get('location', '').lower()
            high_risk_locations = ['china', 'india', 'vietnam']  # Simplified for demo
            if any(loc in location for loc in high_risk_locations):
                risk_factors.append(0.3)
            else:
                risk_factors.append(0.1)
            
            # Calculate average risk score
            if risk_factors:
                avg_risk = sum(risk_factors) / len(risk_factors)
                return min(1.0, avg_risk)
            else:
                return 0.5  # Default medium risk
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {e}")
            return 0.5
    
    def calculate_reorder_quantity(self, product_name, target_quantity, supplier_id=None, months=6):
        """Calculate optimal reorder quantity based on historical data."""
        try:
            # Get historical purchase data
            trends = self.tidb_client.get_purchase_trends(
                supplier_id=supplier_id, 
                product_category=None, 
                months=months
            )
            
            if not trends:
                # No historical data, use target quantity with safety stock
                safety_stock_factor = 1.2
                return int(target_quantity * safety_stock_factor)
            
            # Calculate average monthly demand
            total_quantity = sum(trend['total_quantity'] for trend in trends)
            avg_monthly_demand = total_quantity / len(trends)
            
            # Calculate demand variability
            quantities = [trend['total_quantity'] for trend in trends]
            demand_std = np.std(quantities)
            
            # Safety stock calculation (simplified)
            safety_stock = demand_std * 1.5  # 1.5 standard deviations
            
            # Economic order quantity (simplified)
            eoq = np.sqrt((2 * avg_monthly_demand * 100) / 5)  # Assuming $100 ordering cost, $5 holding cost
            
            # Recommended quantity
            recommended_qty = max(
                int(avg_monthly_demand + safety_stock),
                int(eoq),
                target_quantity
            )
            
            return recommended_qty
            
        except Exception as e:
            self.logger.error(f"Error calculating reorder quantity: {e}")
            return target_quantity
    
    def get_supplier_recommendations(self, product_name, quantity, category=None, budget=None):
        """Get top supplier recommendations for a procurement request."""
        try:
            # Create query vector based on requirements
            query_vector = self._create_query_vector(product_name, quantity, category)
            
            # Search for similar suppliers
            similar_suppliers = self.tidb_client.vector_search_suppliers(
                query_vector, category=category, limit=20
            )
            
            recommendations = []
            
            for supplier in similar_suppliers:
                # Get performance data
                performance = self.tidb_client.get_supplier_performance(supplier['supplier_id'])
                
                if not performance:
                    continue
                
                # Calculate scores
                supplier_score = self.calculate_supplier_score(supplier, performance)
                risk_score = self.calculate_risk_score(supplier, performance, None)
                
                # Calculate reorder quantity
                reorder_qty = self.calculate_reorder_quantity(
                    product_name, quantity, supplier['supplier_id']
                )
                
                # Calculate total cost
                unit_cost = float(supplier.get('price_per_unit', 0))
                total_cost = unit_cost * reorder_qty
                
                # Check budget constraint
                if budget and total_cost > budget:
                    continue
                
                # Check minimum order quantity
                min_order = int(supplier.get('min_order_quantity', 0))
                if reorder_qty < min_order:
                    reorder_qty = min_order
                    total_cost = unit_cost * reorder_qty
                
                recommendation = {
                    'supplier_id': supplier['supplier_id'],
                    'supplier_name': supplier['supplier_name'],
                    'category': supplier['category'],
                    'rating': supplier['rating'],
                    'lead_time_days': supplier['lead_time_days'],
                    'unit_cost': unit_cost,
                    'recommended_quantity': reorder_qty,
                    'total_cost': total_cost,
                    'supplier_score': supplier_score,
                    'risk_score': risk_score,
                    'on_time_delivery_rate': performance.get('on_time_delivery_rate', 0),
                    'quality_score': performance.get('quality_score', 0),
                    'location': supplier['location'],
                    'payment_terms': supplier['payment_terms']
                }
                
                recommendations.append(recommendation)
            
            # Sort by supplier score (descending)
            recommendations.sort(key=lambda x: x['supplier_score'], reverse=True)
            
            return recommendations[:10]  # Return top 10
            
        except Exception as e:
            self.logger.error(f"Error getting supplier recommendations: {e}")
            return []
    
    def _create_query_vector(self, product_name, quantity, category):
        """Create a query vector for vector search."""
        try:
            # Simple feature vector based on requirements
            # In a real implementation, this could be more sophisticated
            
            # Normalize quantity (assuming typical range 1-10000)
            normalized_quantity = min(1.0, quantity / 10000)
            
            # Category encoding (simplified)
            category_weights = {
                'electronics': [0.8, 0.2, 0.1, 0.3],
                'industrial': [0.3, 0.8, 0.2, 0.1],
                'automotive': [0.2, 0.1, 0.8, 0.4],
                'chemicals': [0.1, 0.3, 0.2, 0.8],
                'logistics': [0.4, 0.1, 0.1, 0.2],
                'general': [0.5, 0.5, 0.5, 0.5]
            }
            
            category_vector = category_weights.get(category.lower(), [0.5, 0.5, 0.5, 0.5])
            
            # Combine features
            query_vector = category_vector + [normalized_quantity]
            
            return np.array(query_vector)
            
        except Exception as e:
            self.logger.error(f"Error creating query vector: {e}")
            return np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    
    def analyze_market_trends(self, product_category=None, months=12):
        """Analyze market trends for procurement insights."""
        try:
            trends = self.tidb_client.get_purchase_trends(
                product_category=product_category, 
                months=months
            )
            
            if not trends:
                return {
                    'total_spent': 0,
                    'avg_unit_cost': 0,
                    'total_orders': 0,
                    'cost_trend': 'stable',
                    'demand_trend': 'stable'
                }
            
            # Calculate trends
            total_spent = sum(trend['total_spent'] for trend in trends)
            avg_unit_cost = sum(trend['avg_unit_cost'] for trend in trends) / len(trends)
            total_orders = sum(trend['order_count'] for trend in trends)
            
            # Analyze cost trend
            costs = [trend['avg_unit_cost'] for trend in trends]
            if len(costs) >= 2:
                cost_change = (costs[-1] - costs[0]) / costs[0]
                if cost_change > 0.1:
                    cost_trend = 'increasing'
                elif cost_change < -0.1:
                    cost_trend = 'decreasing'
                else:
                    cost_trend = 'stable'
            else:
                cost_trend = 'stable'
            
            # Analyze demand trend
            quantities = [trend['total_quantity'] for trend in trends]
            if len(quantities) >= 2:
                demand_change = (quantities[-1] - quantities[0]) / quantities[0]
                if demand_change > 0.1:
                    demand_trend = 'increasing'
                elif demand_change < -0.1:
                    demand_trend = 'decreasing'
                else:
                    demand_trend = 'stable'
            else:
                demand_trend = 'stable'
            
            return {
                'total_spent': total_spent,
                'avg_unit_cost': avg_unit_cost,
                'total_orders': total_orders,
                'cost_trend': cost_trend,
                'demand_trend': demand_trend,
                'monthly_data': trends
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market trends: {e}")
            return {}
    
    def generate_procurement_report(self, recommendations, market_analysis):
        """Generate a comprehensive procurement report."""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_recommendations': len(recommendations),
                    'total_cost_range': {
                        'min': min([r['total_cost'] for r in recommendations]) if recommendations else 0,
                        'max': max([r['total_cost'] for r in recommendations]) if recommendations else 0
                    },
                    'avg_risk_score': sum([r['risk_score'] for r in recommendations]) / len(recommendations) if recommendations else 0
                },
                'top_recommendations': recommendations[:3],
                'market_analysis': market_analysis,
                'insights': self._generate_insights(recommendations, market_analysis)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating procurement report: {e}")
            return {}
    
    def _generate_insights(self, recommendations, market_analysis):
        """Generate insights from recommendations and market analysis."""
        insights = []
        
        try:
            if not recommendations:
                insights.append("No suitable suppliers found for the given criteria.")
                return insights
            
            # Cost insights
            costs = [r['total_cost'] for r in recommendations]
            avg_cost = sum(costs) / len(costs)
            min_cost = min(costs)
            
            if min_cost < avg_cost * 0.8:
                insights.append(f"Significant cost savings available: {min_cost:.2f} vs average {avg_cost:.2f}")
            
            # Risk insights
            high_risk_suppliers = [r for r in recommendations if r['risk_score'] > 0.7]
            if high_risk_suppliers:
                insights.append(f"Warning: {len(high_risk_suppliers)} suppliers have high risk scores")
            
            # Performance insights
            top_performers = [r for r in recommendations if r['supplier_score'] > 0.8]
            if top_performers:
                insights.append(f"{len(top_performers)} suppliers have excellent performance scores")
            
            # Market insights
            if market_analysis.get('cost_trend') == 'increasing':
                insights.append("Market costs are trending upward - consider locking in prices")
            elif market_analysis.get('cost_trend') == 'decreasing':
                insights.append("Market costs are trending downward - consider delaying purchases")
            
            if market_analysis.get('demand_trend') == 'increasing':
                insights.append("Market demand is increasing - secure supply early")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
            return ["Unable to generate insights due to data processing error"]
