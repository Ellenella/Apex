# Quick Start Guide - AI-Enhanced Procurement Optimization Agent

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
1. Copy `env_example.txt` to `.env`
2. Update the configuration with your credentials:
```env
# TiDB Configuration
TIDB_HOST=your-tidb-host.tidbcloud.com
TIDB_PORT=4000
TIDB_USER=your-username
TIDB_PASSWORD=your-password
TIDB_DATABASE=procurement_db

# OpenAI Configuration (Required for GPT-3.5 Turbo)
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-3.5-turbo

# MCP Configuration (Optional)
MCP_SERVER_URL=http://localhost:3000
MCP_API_KEY=your-mcp-api-key
```

### 3. Set Up TiDB Database
```bash
python setup_database.py
```

### 4. Run the AI-Enhanced Application
```bash
streamlit run enhanced_app.py
```

### 5. Use the AI-Enhanced Application
1. **Dashboard**: View key metrics and AI capabilities status
2. **Data Upload**: Upload your CSV files and build vector index
3. **AI Analysis**: Configure AI settings and run enhanced analysis
4. **Enhanced Results**: View AI-powered recommendations and insights
5. **AI Insights**: Explore GPT-3.5 Turbo analysis and vector search results
6. **External Tools**: Send AI-enhanced notifications or export results

## 🤖 AI Capabilities Setup

### OpenAI GPT-3.5 Turbo
1. Get your OpenAI API key from [OpenAI Platform](https://platform.openai.com/)
2. Add to `.env`:
```env
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-3.5-turbo
```

### Vector Search (Automatic)
- **Sentence Transformers**: Automatically downloaded on first use
- **FAISS Index**: Built automatically when you upload supplier data
- **No additional setup required**

### MCP Server (Optional)
1. Set up your MCP server (e.g., on localhost:3000)
2. Add to `.env`:
```env
MCP_SERVER_URL=http://localhost:3000
MCP_API_KEY=your-mcp-api-key
```

## 📊 Sample Data Included

The project includes sample CSV files in the `sample_data/` directory:
- `suppliers.csv` - 10 sample suppliers with ratings and performance data
- `purchase_history.csv` - 15 historical purchase records
- `delivery_performance.csv` - Supplier performance metrics

## 🔧 Optional External Tools Setup

### Slack Notifications
1. Create a Slack app and get your bot token
2. Add to `.env`:
```env
SLACK_BOT_TOKEN=xoxb-your-token
SLACK_CHANNEL_ID=C1234567890
```

### Email Notifications
1. Use Gmail with app password or configure your SMTP server
2. Add to `.env`:
```env
EMAIL_USER=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
EMAIL_RECIPIENTS=user1@company.com,user2@company.com
```

### Google Sheets Export
1. Create a Google Cloud project and enable Google Sheets API
2. Download service account credentials JSON file
3. Add to `.env`:
```env
GOOGLE_SHEETS_CREDENTIALS_FILE=path/to/credentials.json
```

## 🎯 AI-Enhanced Demo Workflow

1. **Upload Data**: Use the sample CSV files or upload your own
2. **Build Vector Index**: Automatically creates FAISS index for semantic search
3. **Configure AI**: Enable GPT-3.5 Turbo, vector search, and MCP
4. **Run AI Analysis**: Enter "Microcontroller Board" as product, quantity 500
5. **View AI Results**: See AI-enhanced recommendations with reasoning
6. **Explore Insights**: Review GPT-3.5 Turbo analysis and vector search results
7. **Export Results**: Send AI-enhanced notifications or export to CSV/Google Sheets

## 🤖 AI Features to Explore

### Vector Search
- **Semantic Similarity**: Find suppliers based on meaning
- **Fast Retrieval**: Sub-second search using FAISS
- **Hybrid Ranking**: Combine vector and text search

### GPT-3.5 Turbo Analysis
- **Intelligent Recommendations**: AI-powered supplier selection
- **Risk Assessment**: Advanced risk evaluation
- **Strategic Insights**: Automated procurement strategies

### MCP Integration
- **External Context**: Additional insights from MCP servers
- **Enhanced Analysis**: Combine internal and external AI

## 🐛 Troubleshooting

### AI Model Issues
- **OpenAI API Errors**: Check your API key and billing status
- **Vector Search Issues**: Ensure sentence-transformers is installed
- **MCP Connection**: Verify MCP server URL and API key

### Connection Issues
- Verify TiDB credentials in `.env`
- Check network connectivity to TiDB host
- Ensure database exists and is accessible

### Import Errors
- Install all dependencies: `pip install -r requirements.txt`
- Check Python version (3.8+ recommended)
- Verify CSV file formats match expected schema

### External Tools Not Working
- Check configuration in `.env`
- Verify API credentials and permissions
- Review logs for specific error messages

## 📈 Next Steps

1. **Customize AI Weights**: Modify scoring weights in `enhanced_recommendation_engine.py`
2. **Extend MCP Integration**: Add custom MCP servers for specialized analysis
3. **Optimize Vector Search**: Fine-tune embedding model and search parameters
4. **Scale AI Models**: Deploy to production with proper monitoring

## 🏆 Hackathon Ready

This AI-enhanced project is designed for the TiDB AgentX hackathon and includes:
- ✅ **Advanced Vector Search** with FAISS and sentence transformers
- ✅ **GPT-3.5 Turbo Integration** for intelligent analysis
- ✅ **MCP Server Integration** for enhanced context
- ✅ **Multi-Modal AI Scoring** combining traditional and AI methods
- ✅ **AI-Enhanced Visualizations** and insights
- ✅ **Comprehensive Documentation** and sample data
- ✅ **Production-ready AI Architecture**

## 🚀 Performance Tips

- **Vector Search**: FAISS index provides sub-second response times
- **GPT-3.5 Turbo**: Use rate limiting for production deployments
- **Memory Usage**: FAISS index loaded in memory for optimal performance
- **Scalability**: Designed to handle thousands of suppliers efficiently

Good luck with your AI-enhanced hackathon submission! 🤖✨
