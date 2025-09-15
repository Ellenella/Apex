# MCP (Model Context Protocol) Setup Guide

## What is MCP?

Model Context Protocol (MCP) is a protocol that enables AI applications to connect to external data sources and tools. In our procurement system, MCP provides additional analysis capabilities, market insights, and strategic recommendations beyond what GPT-3.5 Turbo and vector search can provide alone.

## MCP Architecture in Our System

```
┌─────────────────┐    HTTP/REST    ┌─────────────────┐
│   Streamlit     │ ──────────────► │   MCP Server    │
│   Application   │                 │   (Flask)       │
└─────────────────┘                 └─────────────────┘
         │                                   │
         │                                   │
         ▼                                   ▼
┌─────────────────┐                 ┌─────────────────┐
│  AI Enhanced    │                 │  Procurement    │
│  Client         │                 │  Analysis       │
└─────────────────┘                 └─────────────────┘
```

## How to Run MCP

### Option 1: Run the Built-in MCP Server (Recommended)

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**
   Copy `env_example.txt` to `.env` and update:
   ```bash
   cp env_example.txt .env
   ```
   
   Edit `.env` and set:
   ```env
   MCP_SERVER_URL=http://localhost:3000
   MCP_API_KEY=your-secret-api-key
   MCP_SERVER_PORT=3000
   ```

3. **Start the MCP Server**
   ```bash
   python mcp_server_example.py
   ```
   
   You should see:
   ```
   🚀 Starting MCP Server on port 3000
   📊 Health check: http://localhost:3000/health
   🔍 Analysis endpoint: http://localhost:3000/analyze
   📈 Supplier insights: http://localhost:3000/suppliers
   🔑 API Key: your-secret-api-key
   ```

4. **Test the MCP Server**
   ```bash
   # Health check
   curl http://localhost:3000/health
   
   # Test analysis endpoint
   curl -X POST http://localhost:3000/analyze \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your-secret-api-key" \
     -d '{
       "query": "Need 1000 units of electronic components",
       "suppliers_count": 15,
       "search_results": [
         {
           "supplier_name": "TechCorp",
           "location": "China",
           "lead_time_days": 45,
           "price_per_unit": 25.50
         }
       ]
     }'
   ```

5. **Run the Enhanced Streamlit App**
   ```bash
   streamlit run enhanced_app.py
   ```

### Option 2: Use an External MCP Server

If you have access to an external MCP server, update your `.env`:

```env
MCP_SERVER_URL=https://your-mcp-server.com/api
MCP_API_KEY=your-external-api-key
```

### Option 3: Run MCP Server in Background

For production use, you can run the MCP server as a background service:

```bash
# Using nohup (Linux/Mac)
nohup python mcp_server_example.py > mcp_server.log 2>&1 &

# Using Windows Task Manager or PowerShell
Start-Process python -ArgumentList "mcp_server_example.py" -WindowStyle Hidden
```

## MCP Server Endpoints

### 1. Health Check
- **URL**: `GET /health`
- **Purpose**: Verify server is running
- **Response**: Server status and version

### 2. Analysis Endpoint
- **URL**: `POST /analyze`
- **Purpose**: Main procurement analysis
- **Headers**: 
  - `Content-Type: application/json`
  - `Authorization: Bearer <api-key>` (optional)
- **Request Body**:
  ```json
  {
    "query": "procurement request description",
    "suppliers_count": 15,
    "search_results": [
      {
        "supplier_name": "Supplier Name",
        "category": "Category",
        "location": "Location",
        "rating": 4.5,
        "lead_time_days": 30,
        "price_per_unit": 25.00
      }
    ]
  }
  ```

### 3. Supplier Insights
- **URL**: `GET /suppliers`
- **Purpose**: Get general market insights
- **Response**: Market overview, trends, alerts

## MCP Analysis Features

The MCP server provides:

### 1. Market Analysis
- Market competition assessment
- Supplier diversity analysis
- Market stability evaluation
- Trend direction analysis

### 2. Risk Assessment
- Supply chain risk evaluation
- Geopolitical risk analysis
- Lead time risk assessment
- Risk mitigation strategies

### 3. Cost Optimization
- Price range analysis
- Potential savings calculation
- Bulk purchasing recommendations
- Competitive pricing strategies

### 4. Strategic Insights
- Category-specific recommendations
- Long-term contract suggestions
- Supplier relationship strategies
- Sustainability considerations

### 5. Supplier Metrics
- Performance calculations
- Location diversity analysis
- Category distribution
- Average ratings and lead times

## Integration with Streamlit App

The MCP server integrates seamlessly with the enhanced Streamlit application:

1. **AI Analysis Page**: When you submit a procurement query, the system calls the MCP server for additional insights
2. **AI Insights Page**: Displays MCP analysis results alongside GPT-3.5 Turbo analysis
3. **Enhanced Results**: Combines traditional scoring with MCP insights for comprehensive recommendations

## Troubleshooting

### Common Issues

1. **MCP Server Won't Start**
   ```bash
   # Check if port is in use
   netstat -an | grep 3000
   
   # Try different port
   export MCP_SERVER_PORT=3001
   python mcp_server_example.py
   ```

2. **Connection Refused**
   - Ensure MCP server is running
   - Check firewall settings
   - Verify URL in `.env` file

3. **Authentication Errors**
   - Verify API key in `.env`
   - Check Authorization header format
   - Ensure API key matches server configuration

4. **Timeout Errors**
   - Increase timeout in `ai_enhanced_client.py`
   - Check network connectivity
   - Monitor server performance

### Debug Mode

Enable debug logging:

```python
# In mcp_server_example.py
logging.basicConfig(level=logging.DEBUG)

# In ai_enhanced_client.py
logging.basicConfig(level=logging.DEBUG)
```

## Performance Considerations

1. **Server Resources**: MCP server uses minimal resources (~50MB RAM)
2. **Response Time**: Typical response time < 500ms
3. **Concurrent Requests**: Flask development server handles ~10 concurrent requests
4. **Production**: Use Gunicorn or uWSGI for production deployment

## Security

1. **API Key**: Always use a strong API key
2. **HTTPS**: Use HTTPS in production
3. **Rate Limiting**: Implement rate limiting for production use
4. **Input Validation**: Validate all input data

## Example MCP Response

```json
{
  "mcp_insights": {
    "market_analysis": {
      "market_competition": "high",
      "supplier_diversity": "excellent",
      "market_stability": "stable",
      "trend_direction": "increasing competition"
    },
    "risk_assessment": {
      "overall_risk": "medium",
      "risk_score": 0.25,
      "risk_factors": ["Geopolitical risk (China)", "Moderate lead time risk"],
      "mitigation_strategies": [
        "Diversify supplier base",
        "Establish backup suppliers",
        "Monitor geopolitical developments"
      ]
    },
    "cost_optimization": {
      "suggestions": [
        "Significant cost savings available through supplier negotiation",
        "Consider bulk purchasing for better pricing"
      ],
      "potential_savings": 15.2,
      "price_range": "$20.50 - $35.00",
      "average_price": "$27.75"
    },
    "strategic_recommendations": [
      "Consider long-term contracts for electronics due to supply chain volatility",
      "Monitor semiconductor supply chain developments",
      "Strong supplier competition - leverage for better terms"
    ]
  },
  "supplier_metrics": {
    "metrics": {
      "total_suppliers": 15,
      "avg_rating": 4.2,
      "avg_lead_time": 35.5,
      "locations": ["China", "USA", "Germany", "Japan"],
      "categories": ["Electronics", "Components", "Services"]
    },
    "summary": "Analyzed 15 suppliers across 3 categories"
  },
  "procurement_trends": {
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
  },
  "timestamp": "2024-01-15T10:30:00.000Z",
  "mcp_version": "1.0.0"
}
```

## Next Steps

1. **Customize Analysis**: Modify `mcp_server_example.py` to add your own analysis logic
2. **Add Data Sources**: Connect to external APIs for real-time market data
3. **Scale Up**: Deploy MCP server to cloud infrastructure
4. **Monitor**: Add logging and monitoring for production use

## Support

For issues with MCP integration:
1. Check the logs in `mcp_server.log`
2. Verify environment variables
3. Test endpoints manually with curl
4. Review the Flask debug output

