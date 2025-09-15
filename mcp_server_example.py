#!/usr/bin/env python3
"""
Simple MCP (Model Context Protocol) Server Example
This demonstrates how to run an MCP server that can be called by the AI-enhanced procurement system.
"""

from flask import Flask, request, jsonify
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class MCPServer:
    def __init__(self):
        """Initialize MCP server with procurement-specific analysis capabilities."""
        self.api_key = os.getenv('MCP_API_KEY', 'mcp-procurement-2024-secret-key-xyz123')
        
    def analyze_procurement_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze procurement data and provide additional insights."""
        try:
            query = data.get('query', '')
            suppliers_count = data.get('suppliers_count', 0)
            search_results = data.get('search_results', [])
            
            # Perform additional analysis
            analysis = {
                "mcp_insights": {
                    "market_analysis": self._analyze_market_trends(suppliers_count),
                    "risk_assessment": self._assess_supply_chain_risk(search_results),
                    "cost_optimization": self._suggest_cost_optimization(search_results),
                    "strategic_recommendations": self._generate_strategic_insights(query, search_results)
                },
                "supplier_metrics": self._calculate_supplier_metrics(search_results),
                "procurement_trends": self._analyze_procurement_trends(),
                "timestamp": datetime.now().isoformat(),
                "mcp_version": "1.0.0"
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in procurement analysis: {e}")
            return {"error": str(e)}
    
    def _analyze_market_trends(self, suppliers_count: int) -> Dict[str, Any]:
        """Analyze market trends based on supplier availability."""
        return {
            "market_competition": "high" if suppliers_count > 10 else "moderate" if suppliers_count > 5 else "low",
            "supplier_diversity": "excellent" if suppliers_count > 15 else "good" if suppliers_count > 8 else "limited",
            "market_stability": "stable",
            "trend_direction": "increasing competition"
        }
    
    def _assess_supply_chain_risk(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess supply chain risk based on search results."""
        if not search_results:
            return {"overall_risk": "unknown", "risk_factors": []}
        
        risk_factors = []
        total_risk_score = 0
        
        for supplier in search_results:
            # Analyze location risk
            location = supplier.get('location', '').lower()
            if 'china' in location:
                risk_factors.append("Geopolitical risk (China)")
                total_risk_score += 0.3
            elif 'russia' in location:
                risk_factors.append("High geopolitical risk (Russia)")
                total_risk_score += 0.5
            
            # Analyze lead time risk
            lead_time = supplier.get('lead_time_days', 0)
            if lead_time > 60:
                risk_factors.append("Long lead time risk")
                total_risk_score += 0.2
            elif lead_time > 30:
                risk_factors.append("Moderate lead time risk")
                total_risk_score += 0.1
        
        # Normalize risk score
        avg_risk = total_risk_score / len(search_results) if search_results else 0
        
        return {
            "overall_risk": "high" if avg_risk > 0.4 else "medium" if avg_risk > 0.2 else "low",
            "risk_score": round(avg_risk, 2),
            "risk_factors": list(set(risk_factors)),
            "mitigation_strategies": [
                "Diversify supplier base",
                "Establish backup suppliers",
                "Monitor geopolitical developments",
                "Implement just-in-time inventory"
            ]
        }
    
    def _suggest_cost_optimization(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Suggest cost optimization strategies."""
        if not search_results:
            return {"suggestions": [], "potential_savings": 0}
        
        prices = [s.get('price_per_unit', 0) for s in search_results if s.get('price_per_unit', 0) > 0]
        
        if not prices:
            return {"suggestions": ["Price data not available"], "potential_savings": 0}
        
        min_price = min(prices)
        max_price = max(prices)
        avg_price = sum(prices) / len(prices)
        
        potential_savings = (max_price - min_price) / max_price * 100 if max_price > 0 else 0
        
        suggestions = []
        if potential_savings > 20:
            suggestions.append("Significant cost savings available through supplier negotiation")
        if potential_savings > 10:
            suggestions.append("Consider bulk purchasing for better pricing")
        if len(search_results) > 5:
            suggestions.append("Multiple suppliers available - leverage for competitive pricing")
        
        return {
            "suggestions": suggestions,
            "potential_savings": round(potential_savings, 1),
            "price_range": f"${min_price:.2f} - ${max_price:.2f}",
            "average_price": f"${avg_price:.2f}"
        }
    
    def _generate_strategic_insights(self, query: str, search_results: List[Dict[str, Any]]) -> List[str]:
        """Generate strategic procurement insights."""
        insights = []
        
        if "electronics" in query.lower():
            insights.append("Consider long-term contracts for electronics due to supply chain volatility")
            insights.append("Monitor semiconductor supply chain developments")
        
        if "raw materials" in query.lower():
            insights.append("Implement commodity hedging strategies")
            insights.append("Consider local sourcing to reduce transportation costs")
        
        if len(search_results) > 10:
            insights.append("Strong supplier competition - leverage for better terms")
        elif len(search_results) < 3:
            insights.append("Limited supplier options - consider developing new supplier relationships")
        
        insights.append("Implement supplier performance monitoring and KPIs")
        insights.append("Consider sustainability criteria in supplier selection")
        
        return insights
    
    def _calculate_supplier_metrics(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate key supplier performance metrics."""
        if not search_results:
            return {"metrics": {}, "summary": "No supplier data available"}
        
        metrics = {
            "total_suppliers": len(search_results),
            "avg_rating": sum(s.get('rating', 0) for s in search_results) / len(search_results),
            "avg_lead_time": sum(s.get('lead_time_days', 0) for s in search_results) / len(search_results),
            "locations": list(set(s.get('location', 'Unknown') for s in search_results)),
            "categories": list(set(s.get('category', 'Unknown') for s in search_results))
        }
        
        return {
            "metrics": metrics,
            "summary": f"Analyzed {len(search_results)} suppliers across {len(metrics['categories'])} categories"
        }
    
    def _analyze_procurement_trends(self) -> Dict[str, Any]:
        """Analyze procurement trends (mock data for demonstration)."""
        return {
            "quarterly_spend": {
                "Q1": 1250000,
                "Q2": 1380000,
                "Q3": 1420000,
                "Q4": 1560000
            },
            "top_categories": ["Electronics", "Raw Materials", "Packaging", "Services"],
            "supplier_churn_rate": "8.5%",
            "cost_savings_achieved": "12.3%",
            "trend_analysis": "Increasing spend with focus on strategic sourcing"
        }

# Initialize MCP server
mcp_server = MCPServer()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "MCP Procurement Server",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """Main analysis endpoint."""
    try:
        # Check API key if configured
        auth_header = request.headers.get('Authorization', '')
        if mcp_server.api_key and mcp_server.api_key != 'mcp-procurement-2024-secret-key-xyz123':
            if not auth_header.startswith('Bearer ') or auth_header[7:] != mcp_server.api_key:
                return jsonify({"error": "Unauthorized"}), 401
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Perform analysis
        result = mcp_server.analyze_procurement_data(data)
        
        logger.info(f"Analysis completed for query: {data.get('query', 'Unknown')}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/suppliers', methods=['GET'])
def get_supplier_insights():
    """Get general supplier insights."""
    try:
        insights = {
            "market_overview": "Global supplier market showing increased competition",
            "trending_categories": ["Sustainable Materials", "Local Sourcing", "Digital Services"],
            "risk_alerts": ["Supply chain disruptions in Asia", "Rising transportation costs"],
            "opportunities": ["New suppliers entering market", "Technology-driven cost savings"]
        }
        return jsonify(insights)
        
    except Exception as e:
        logger.error(f"Error in supplier insights: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.getenv('MCP_SERVER_PORT', 3000))
    
    print(f"🚀 Starting MCP Server on port {port}")
    print(f"📊 Health check: http://localhost:{port}/health")
    print(f"🔍 Analysis endpoint: http://localhost:{port}/analyze")
    print(f"📈 Supplier insights: http://localhost:{port}/suppliers")
    print(f"🔑 API Key: {mcp_server.api_key if mcp_server.api_key != 'your-mcp-api-key' else 'Not configured'}")
    print("\n" + "="*50)
    
    app.run(host='0.0.0.0', port=port, debug=True)
