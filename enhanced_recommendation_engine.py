import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
import logging
import json
from typing import List, Dict, Any, Optional
from ai_enhanced_client import AIEnhancedClient

class EnhancedRecommendationEngine:
    def __init__(self, tidb_client, ai_client: AIEnhancedClient = None):
        """Initialize the enhanced recommendation engine with AI capabilities."""
        self.tidb_client = tidb_client
        self.ai_client = ai_client or AIEnhancedClient()
        self.logger = logging.getLogger(__name__)
        
        # Weight configurations for different factors
        self.weights = {
            'cost': 0.25,
            'delivery_performance': 0.30,
            'quality': 0.20,
            'lead_time': 0.15,
            'risk': 0.10
        }
        
        # AI enhancement weights
        self.ai_weights = {
            'vector_search': 0.4,
            'gpt_analysis': 0.4,
            'mcp_insights': 0.2
        }
    
    def get_all_suppliers_data(self) -> List[Dict[str, Any]]:
        """Get all suppliers data from TiDB for AI analysis."""
        try:
            if not self.tidb_client.connection:
                if not self.tidb_client.connect():
                    return []
            
            with self.tidb_client.connection.cursor() as cursor:
                cursor.execute("""
                SELECT s.*, 
                       COALESCE(dp.on_time_delivery_rate, 0) as on_time_delivery_rate,
                       COALESCE(dp.quality_score, 0) as quality_score,
                       COALESCE(dp.overall_performance_score, 0) as overall_performance_score,
                       COALESCE(dp.defect_rate, 0) as defect_rate
                FROM suppliers s
                LEFT JOIN delivery_performance dp ON s.supplier_id = dp.supplier_id
            """)
                suppliers = cursor.fetchall()
                
                # Convert to list of dictionaries
                suppliers_data = []
                for supplier in suppliers:
                    supplier_dict = {}
                    for key, value in supplier.items():
                        # Handle different data types
                        if isinstance(value, (int, float)):
                            supplier_dict[key] = float(value)
                        elif isinstance(value, (datetime, date)):
                            supplier_dict[key] = value.isoformat()
                        else:
                            supplier_dict[key] = str(value) if value is not None else ""
                    suppliers_data.append(supplier_dict)
                
                return suppliers_data
                
        except Exception as e:
            self.logger.error(f"Error getting suppliers data: {e}")
            return []
    
    def enhanced_supplier_search(self, query: str, category: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Enhanced supplier search using vector search, full-text search, and AI analysis."""
        try:
            # Get all suppliers data
            suppliers_data = self.get_all_suppliers_data()
            
            if not suppliers_data:
                self.logger.warning("No suppliers data available")
                return []
            
            # Filter by category if specified
            if category:
                suppliers_data = [s for s in suppliers_data if s.get('category', '').lower() == category.lower()]
            
            # Check if AI client and vector search are available
            if not hasattr(self, 'ai_client') or self.ai_client is None:
                self.logger.warning("AI client not available")
                return suppliers_data[:limit]  # Fallback to basic data
            
            # Check if vector search is available
            if not hasattr(self.ai_client, 'index') or self.ai_client.index is None:
                self.logger.warning("Vector index not available, using full-text search only")
                search_results = self.ai_client.full_text_search(query, suppliers_data, k=limit)
            else:
                # Perform hybrid search using AI client
                search_results = self.ai_client.hybrid_search(query, suppliers_data, k=limit)
            
            # Add performance data to results
            enhanced_results = []
            for result in search_results:
                # Get detailed performance data
                performance = self.tidb_client.get_supplier_performance(result.get('supplier_id'))
                if performance:
                    result.update({
                        'on_time_delivery_rate': performance.get('on_time_delivery_rate', 0),
                        'quality_score': performance.get('quality_score', 0),
                        'overall_performance_score': performance.get('overall_performance_score', 0),
                        'defect_rate': performance.get('defect_rate', 0)
                    })
                
                enhanced_results.append(result)
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Error in enhanced supplier search: {e}")
            # Fallback: return basic supplier data
            suppliers_data = self.get_all_suppliers_data()
            if category:
                suppliers_data = [s for s in suppliers_data if s.get('category', '').lower() == category.lower()]
            return suppliers_data[:limit]
    
    def ai_enhanced_scoring(self, supplier_data: Dict[str, Any], performance_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate AI-enhanced scores for suppliers."""
        try:
            # Traditional scoring
            traditional_score = self.calculate_supplier_score(supplier_data, performance_data)
            risk_score = self.calculate_risk_score(supplier_data, performance_data, None)
            
            # AI-enhanced scoring
            ai_scores = {
                'traditional_score': traditional_score,
                'risk_score': risk_score,
                'vector_similarity': supplier_data.get('similarity_score', 0),
                'text_match': supplier_data.get('text_match_score', 0),
                'combined_search_score': supplier_data.get('combined_score', 0)
            }
            
            # Calculate weighted AI score
            ai_score = (
                traditional_score * 0.4 +
                (1 - risk_score) * 0.2 +
                ai_scores['vector_similarity'] * 0.2 +
                ai_scores['combined_search_score'] * 0.2
            )
            
            ai_scores['ai_enhanced_score'] = min(1.0, max(0.0, ai_score))
            
            return ai_scores
            
        except Exception as e:
            self.logger.error(f"Error in AI-enhanced scoring: {e}")
            return {
                'traditional_score': 0.0,
                'risk_score': 0.5,
                'ai_enhanced_score': 0.0
            }
    
    def get_ai_enhanced_recommendations(self, product_name: str, quantity: int, 
                                      category: str = None, budget: float = None) -> Dict[str, Any]:
        """Get AI-enhanced supplier recommendations using GPT-3.5 Turbo and vector search."""
        try:
            # Create comprehensive query
            query = f"procurement request for {quantity} units of {product_name}"
            if category:
                query += f" in {category} category"
            if budget:
                query += f" with budget ${budget}"
            
            # Get suppliers data
            suppliers_data = self.get_all_suppliers_data()
            
            if not suppliers_data:
                return {"error": "No suppliers data available"}
            success = True
            if self.ai_client.index is None:
                self.logger.info("Building vector index with %d suppliers", len(suppliers_data))
                success = self.ai_client.build_vector_index(suppliers_data)
            if not success:
                self.logger.warning("Vector index build failed, falling back to text search")
            # Perform enhanced analysis using AI client
            ai_analysis = self.ai_client.enhanced_procurement_analysis(query, suppliers_data)
            if "error" in ai_analysis:
                self.logger.warning("AI analysis failed: %s", ai_analysis.get("error"))
            
            # Get search results
            search_results = self.enhanced_supplier_search(query, category, limit=15)
            
            # Calculate AI-enhanced scores for each supplier
            enhanced_recommendations = []
            for supplier in search_results:
                # Get performance data
                performance = self.tidb_client.get_supplier_performance(supplier.get('supplier_id'))
                
                # Calculate AI-enhanced scores
                ai_scores = self.ai_enhanced_scoring(supplier, performance or {})
                
                # Calculate reorder quantity
                reorder_qty = self.calculate_reorder_quantity(product_name, quantity, supplier.get('supplier_id'))
                
                # Calculate total cost
                unit_cost = float(supplier.get('price_per_unit', 0))
                total_cost = unit_cost * reorder_qty
                
                # Check budget constraint
                if budget and total_cost > budget:
                    continue
                
                # Create enhanced recommendation
                recommendation = {
                    'supplier_id': supplier.get('supplier_id'),
                    'supplier_name': supplier.get('supplier_name'),
                    'category': supplier.get('category'),
                    'rating': supplier.get('rating'),
                    'lead_time_days': supplier.get('lead_time_days'),
                    'unit_cost': unit_cost,
                    'recommended_quantity': reorder_qty,
                    'total_cost': total_cost,
                    'location': supplier.get('location'),
                    'payment_terms': supplier.get('payment_terms'),
                    'ai_scores': ai_scores,
                    'supplier_score': ai_scores.get('ai_enhanced_score', 0),
                    'risk_score': ai_scores.get('risk_score', 0),
                    'search_scores': {
                        'vector_similarity': supplier.get('similarity_score', 0),
                        'text_match': supplier.get('text_match_score', 0),
                        'combined_score': supplier.get('combined_score', 0)
                    },
                    'performance_metrics': {
                        'on_time_delivery_rate': supplier.get('on_time_delivery_rate', 0),
                        'quality_score': supplier.get('quality_score', 0),
                        'overall_performance_score': supplier.get('overall_performance_score', 0),
                        'defect_rate': supplier.get('defect_rate', 0)
                    }
                }
                
                enhanced_recommendations.append(recommendation)
            
            # Sort by AI-enhanced score
            enhanced_recommendations.sort(key=lambda x: x['ai_scores']['ai_enhanced_score'], reverse=True)
            
            # Get market analysis
            market_analysis = self.analyze_market_trends(category)
            
            # Combine results
            comprehensive_results = {
                'recommendations': enhanced_recommendations[:10],
                'ai_analysis': ai_analysis,
                'market_analysis': market_analysis,
                'query_info': {
                    'product_name': product_name,
                    'quantity': quantity,
                    'category': category,
                    'budget': budget,
                    'total_recommendations': len(enhanced_recommendations)
                },
                'ai_models_used': {
                    'vector_search': 'sentence-transformers',
                    'llm': self.ai_client.openai_model,
                    'mcp': bool(self.ai_client.mcp_server_url)
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return comprehensive_results
            
        except Exception as e:
            self.logger.error(f"Error in AI-enhanced recommendations: {e}")
            return {"error": str(e)}
    
    def calculate_supplier_score(self, supplier_data: Dict[str, Any], performance_data: Dict[str, Any]) -> float:
        """Calculate comprehensive supplier score (inherited from original engine)."""
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
            normalized_lead_time = max(0, 1 - (lead_time / 60))
            normalized_price = max(0, 1 - (price / 100))
            normalized_defect_rate = max(0, 1 - (defect_rate * 100))
            normalized_cost_variance = max(0, 1 - (cost_variance / 10))
            
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
    
    def calculate_risk_score(self, supplier_data: Dict[str, Any], performance_data: Dict[str, Any], historical_data: Any) -> float:
        """Calculate risk score (inherited from original engine)."""
        try:
            risk_factors = []
            
            # Contract risk
            contract_end = supplier_data.get('contract_end_date')
            if contract_end:
                if isinstance(contract_end, str):
                    contract_end_dt = datetime.strptime(contract_end, '%Y-%m-%d').date()
                elif isinstance(contract_end, datetime):
                    contract_end_dt = contract_end.date()
                elif isinstance(contract_end, date):
                    contract_end_dt = contract_end
                else:
                    contract_end_dt = None
                if contract_end_dt:
                    days_to_expiry = (contract_end_dt - datetime.now().date()).days
                    if days_to_expiry < 30:
                        risk_factors.append(0.8)
                    elif days_to_expiry < 90:
                        risk_factors.append(0.4)
                    else:
                        risk_factors.append(0.1)
            
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
            
            # Financial risk
            cost_variance = float(performance_data.get('cost_variance_percent', 0))
            if cost_variance > 5:
                risk_factors.append(0.6)
            elif cost_variance > 2:
                risk_factors.append(0.3)
            else:
                risk_factors.append(0.1)
            
            # Location risk
            location = supplier_data.get('location', '').lower()
            high_risk_locations = ['china', 'india', 'vietnam']
            if any(loc in location for loc in high_risk_locations):
                risk_factors.append(0.3)
            else:
                risk_factors.append(0.1)
            
            # Calculate average risk score
            if risk_factors:
                avg_risk = sum(risk_factors) / len(risk_factors)
                return min(1.0, avg_risk)
            else:
                return 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {e}")
            return 0.5
    
    def calculate_reorder_quantity(self, product_name: str, target_quantity: int, supplier_id: str = None, months: int = 6) -> int:
        """Calculate optimal reorder quantity (inherited from original engine)."""
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
            
            # Safety stock calculation
            safety_stock = demand_std * 1.5
            
            # Economic order quantity
            eoq = np.sqrt((2 * avg_monthly_demand * 100) / 5)
            
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
    
    def analyze_market_trends(self, product_category: str = None, months: int = 12) -> Dict[str, Any]:
        """Analyze market trends (inherited from original engine)."""
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
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the engine state."""
        debug_info = {
            'tidb_client_connected': self.tidb_client.connection is not None if self.tidb_client else False,
            'ai_client_configured': self.ai_client is not None,
            'suppliers_data_count': len(self.get_all_suppliers_data()),
            'methods_available': [method for method in dir(self) if not method.startswith('_')]
        }
        
        # Add AI client debug info if available
        if self.ai_client:
            debug_info.update({
                'ai_client_openai_ready': self.ai_client.openai_client is not None,
                'ai_client_vector_ready': self.ai_client.index is not None if hasattr(self.ai_client, 'index') else False,
                'ai_client_supplier_data_count': len(self.ai_client.supplier_data) if hasattr(self.ai_client, 'supplier_data') else 0
            })
        
        return debug_info