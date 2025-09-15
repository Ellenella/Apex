# AI-Enhanced Procurement Optimization Agent (TiDB + Streamlit + GPT-3.5 Turbo)

## Project Overview

This application is an **advanced multi-step AI agent** that leverages cutting-edge AI technologies to help procurement managers make **data-driven sourcing decisions**. Using **TiDB Serverless** for data storage, **vector search** for semantic similarity, **GPT-3.5 Turbo** for intelligent analysis, and **Model Context Protocol (MCP)** for enhanced insights, the agent identifies optimal suppliers, calculates reorder schedules, evaluates risk, and provides actionable recommendations.

This workflow **demonstrates a sophisticated multi-step automated process** with advanced AI capabilities, fully compatible with the TiDB AgentX hackathon requirements.

## 🤖 Advanced AI Features

### 1. **Vector Search & Semantic Similarity**
- **Sentence Transformers**: Uses `all-MiniLM-L6-v2` for high-quality embeddings
- **FAISS Index**: Fast similarity search for supplier matching
- **Hybrid Search**: Combines vector search with full-text search for optimal results
- **Semantic Understanding**: Finds suppliers based on meaning, not just keywords

### 2. **GPT-3.5 Turbo Integration**
- **Intelligent Analysis**: AI-powered supplier recommendations with reasoning
- **Risk Assessment**: Advanced risk evaluation using natural language understanding
- **Strategic Insights**: Automated generation of procurement strategies
- **Cost-Benefit Analysis**: AI-driven cost optimization recommendations

### 3. **Model Context Protocol (MCP)**
- **Enhanced Context**: Additional analysis through MCP server integration
- **External Insights**: Leverage external AI models and data sources
- **Scalable Architecture**: Modular design for easy MCP server integration

### 4. **Multi-Modal AI Scoring**
- **Traditional Scoring**: Rule-based supplier evaluation
- **AI-Enhanced Scoring**: Combines traditional metrics with AI insights
- **Vector Similarity Scoring**: Semantic matching scores
- **Hybrid Ranking**: Intelligent combination of all scoring methods

## Features

1. **Advanced Data Ingestion & Indexing**
   - Upload CSVs containing supplier details, purchase history, and delivery performance metrics
   - **Vector Embeddings**: Automatic generation of semantic embeddings using sentence transformers
   - **FAISS Indexing**: High-performance vector search capabilities
   - Store embeddings and metadata in **TiDB Serverless**

2. **Intelligent Search & Analysis**
   - **Vector Search**: Semantic similarity search using FAISS
   - **Full-Text Search**: Traditional keyword-based search
   - **Hybrid Search**: Combines both approaches for optimal results
   - **GPT-3.5 Turbo Analysis**: AI-powered insights and recommendations

3. **AI-Enhanced Recommendation Engine**
   - **Multi-Modal Scoring**: Traditional + AI + Vector similarity scoring
   - **GPT-3.5 Turbo Integration**: Intelligent supplier analysis and reasoning
   - **MCP Server Integration**: Additional context and insights
   - **Risk Assessment**: Advanced AI-powered risk evaluation

4. **External Tool Integrations**
   - Slack / Email Notifications: AI-enhanced alerts with detailed insights
   - Google Sheets / CSV Export: Comprehensive AI analysis results
   - Optional: Integrate with ERP for automated order creation

5. **Interactive Streamlit Dashboard**
   - **AI Configuration**: Toggle different AI capabilities
   - **Enhanced Visualizations**: AI vs traditional scoring comparisons
   - **GPT Insights**: Display AI-generated recommendations and reasoning
   - **Vector Search Results**: Show semantic similarity scores

## Multi-Step AI-Enhanced Workflow

```
1. User uploads CSVs → Supplier & Purchase History
        ↓
2. TiDB Serverless → Index and store embeddings / normalized data
        ↓
3. Vector Index Building → FAISS index creation with sentence transformers
        ↓
4. User query → Hybrid search (Vector + Full-text) in TiDB + FAISS
        ↓
5. AI Analysis → GPT-3.5 Turbo analysis + MCP server insights
        ↓
6. Enhanced Scoring → Multi-modal scoring (Traditional + AI + Vector)
        ↓
7. External Tools → AI-enhanced notifications and exports
        ↓
8. Streamlit Dashboard → Display AI insights and comparisons
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```env
# TiDB Configuration
TIDB_HOST=your-tidb-host.tidbcloud.com
TIDB_PORT=4000
TIDB_USER=your-username
TIDB_PASSWORD=your-password
TIDB_DATABASE=procurement_db

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-3.5-turbo

# MCP Configuration
MCP_SERVER_URL=http://localhost:3000
MCP_API_KEY=your-mcp-api-key

# External Tools (Optional)
SLACK_BOT_TOKEN=your-slack-bot-token
SLACK_CHANNEL_ID=your-channel-id
GOOGLE_SHEETS_CREDENTIALS_FILE=path/to/credentials.json
```

### 3. Set Up TiDB Database
Run the database setup script:
```bash
python setup_database.py
```

### 4. Run the Enhanced Application
```bash
streamlit run enhanced_app.py
```

## Usage

1. **Upload Data**: Upload CSV files with supplier and purchase data
2. **Configure AI**: Enable/disable GPT-3.5 Turbo, vector search, and MCP
3. **Enter Query**: Input procurement query (product, quantity, timeframe)
4. **Run AI Analysis**: Click "Run AI-Enhanced Analysis" to get intelligent recommendations
5. **Review Insights**: View GPT-3.5 Turbo analysis and vector search results
6. **Take Action**: Send AI-enhanced notifications or export results

## AI Models & Technologies

### Vector Search
- **Model**: `all-MiniLM-L6-v2` (Sentence Transformers)
- **Index**: FAISS (Facebook AI Similarity Search)
- **Dimension**: 384-dimensional embeddings
- **Performance**: Sub-second similarity search

### Language Model
- **Model**: GPT-3.5 Turbo (OpenAI)
- **Capabilities**: Natural language analysis, reasoning, recommendations
- **Output**: Structured JSON analysis with insights

### Model Context Protocol
- **Integration**: RESTful API calls to MCP server
- **Purpose**: Additional context and external insights
- **Extensibility**: Easy integration with custom MCP servers

## Sample Data

The project includes sample CSV files:
- `sample_data/suppliers.csv` - Supplier information
- `sample_data/purchase_history.csv` - Historical purchase data
- `sample_data/delivery_performance.csv` - Delivery metrics

## Project Structure

```
├── enhanced_app.py                    # AI-enhanced Streamlit application
├── ai_enhanced_client.py              # OpenAI, MCP, and vector search client
├── enhanced_recommendation_engine.py  # AI-enhanced recommendation logic
├── ti_db_client.py                    # TiDB connection and operations
├── external_tools.py                  # Slack, email, Google Sheets integration
├── setup_database.py                  # Database initialization
├── requirements.txt                   # Python dependencies
├── .env                              # Environment variables
├── sample_data/                      # Sample CSV files
│   ├── suppliers.csv
│   ├── purchase_history.csv
│   └── delivery_performance.csv
└── README.md                         # This file
```

## AI-Enhanced Capabilities

### Vector Search Features
- **Semantic Similarity**: Find suppliers based on meaning, not just keywords
- **Fast Retrieval**: Sub-second search using FAISS index
- **Hybrid Ranking**: Combine vector and text search scores
- **Contextual Matching**: Understand supplier capabilities and specialties

### GPT-3.5 Turbo Features
- **Intelligent Recommendations**: AI-powered supplier selection with reasoning
- **Risk Assessment**: Advanced risk evaluation using natural language
- **Strategic Insights**: Automated procurement strategy generation
- **Cost Analysis**: AI-driven cost optimization recommendations

### MCP Integration Features
- **External Context**: Additional insights from MCP servers
- **Scalable Architecture**: Easy integration with multiple MCP servers
- **Enhanced Analysis**: Combine internal and external AI capabilities

## Hackathon Submission Notes

- **TiDB Cloud account email**
- **Public GitHub repo** with code, requirements.txt, sample data, and README
- **Demo video (3–5 min)** showing AI-enhanced features:
  - Vector search capabilities
  - GPT-3.5 Turbo analysis
  - MCP server integration
  - AI-enhanced recommendations
  - Advanced visualizations

## Contributing

This project is designed for the TiDB AgentX hackathon and demonstrates advanced AI capabilities. Feel free to extend the functionality and improve the AI algorithms.

## AI Model Requirements

- **OpenAI API Key**: Required for GPT-3.5 Turbo integration
- **Sentence Transformers**: Automatically downloaded on first use
- **FAISS**: CPU version included for vector search
- **MCP Server**: Optional for additional insights

## Performance Considerations

- **Vector Search**: Optimized with FAISS for sub-second response times
- **GPT-3.5 Turbo**: Rate-limited by OpenAI API
- **Memory Usage**: FAISS index loaded in memory for fast access
- **Scalability**: Designed to handle thousands of suppliers efficiently
