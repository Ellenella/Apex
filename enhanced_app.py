import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import json

# Import our enhanced modules
from ti_db_client import TiDBClient
from enhanced_recommendation_engine import EnhancedRecommendationEngine
from ai_enhanced_client import AIEnhancedClient
from external_tools import ExternalTools

# Configure page
st.set_page_config(
    page_title="AI-Enhanced Procurement Optimization Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'tidb_client' not in st.session_state:
    st.session_state.tidb_client = None
if 'ai_client' not in st.session_state:
    st.session_state.ai_client = None
if 'enhanced_engine' not in st.session_state:
    st.session_state.enhanced_engine = None
if 'external_tools' not in st.session_state:
    st.session_state.external_tools = None
if 'enhanced_recommendations' not in st.session_state:
    st.session_state.enhanced_recommendations = {}
if 'ai_analysis' not in st.session_state:
    st.session_state.ai_analysis = {}

def initialize_components():
    """Initialize enhanced components with AI capabilities."""
    if st.session_state.tidb_client is None:
        st.session_state.tidb_client = TiDBClient()
        if not st.session_state.tidb_client.connect():
            st.error("Failed to connect to TiDB. Please check your configuration.")
            return False
    
    if st.session_state.ai_client is None:
        st.session_state.ai_client = AIEnhancedClient()
        # Auto-rebuild vector index from DB if suppliers exist
        try:
            if st.session_state.tidb_client and st.session_state.tidb_client.connection:
                with st.session_state.tidb_client.connection.cursor() as cursor:
                    cursor.execute("SELECT * FROM suppliers")
                    suppliers = cursor.fetchall()
                if suppliers:
                    st.session_state.ai_client.build_vector_index(suppliers)
        except Exception as e:
            st.warning(f"Could not rebuild vector index on startup: {e}")
    
    if st.session_state.enhanced_engine is None:
        st.session_state.enhanced_engine = EnhancedRecommendationEngine(
            st.session_state.tidb_client, 
            st.session_state.ai_client
        )
    
    if st.session_state.external_tools is None:
        st.session_state.external_tools = ExternalTools()
    
    return True

def main():
    """Main application function."""
    
    # Header
    st.title("🤖 AI-Enhanced Procurement Optimization Agent")
    st.markdown("**Advanced multi-step AI agent with vector search, GPT-3.5 Turbo, and MCP integration**")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Initialize components
        if not initialize_components():
            st.stop()
        
        # Connection status
        if st.session_state.tidb_client and st.session_state.tidb_client.connection:
            st.success("✅ Connected to TiDB")
        else:
            st.error("❌ TiDB connection failed")
            st.stop()
        
        # AI Status
        ai_status = "✅ Configured" if st.session_state.ai_client.openai_client else "⚠️ OpenAI not configured"
        st.info(f"🤖 AI Capabilities: {ai_status}")
        
        # Navigation
        st.header("📋 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Dashboard", "📊 Data Upload", "🔍 AI Analysis", "📈 Enhanced Results", "🤖 AI Insights", "🔗 External Tools", "Debugger"]
        )
    
    # Page routing
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📊 Data Upload":
        show_data_upload()
    elif page == "🔍 AI Analysis":
        show_ai_analysis()
    elif page == "📈 Enhanced Results":
        show_enhanced_results()
    elif page == "🤖 AI Insights":
        show_ai_insights()
    elif page == "🔗 External Tools":
        show_external_tools()
    elif page == "Debugger":
        show_debugger()

def show_dashboard():
    """Show the enhanced dashboard."""
    st.header("🏠 AI-Enhanced Procurement Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        with st.session_state.tidb_client.connection.cursor() as cursor:
            # Total suppliers
            cursor.execute("SELECT COUNT(*) as count FROM suppliers")
            supplier_count = cursor.fetchone()['count']
            
            # Total purchases
            cursor.execute("SELECT COUNT(*) as count FROM purchase_history")
            purchase_count = cursor.fetchone()['count']
            
            # Total spend
            cursor.execute("SELECT SUM(total_cost) as total FROM purchase_history")
            total_spend = cursor.fetchone()['total'] or 0
            
            # Average supplier rating
            cursor.execute("SELECT AVG(rating) as avg_rating FROM suppliers")
            avg_rating = cursor.fetchone()['avg_rating'] or 0
        
        with col1:
            st.metric("Total Suppliers", supplier_count)
        
        with col2:
            st.metric("Total Purchases", purchase_count)
        
        with col3:
            st.metric("Total Spend", f"${total_spend:,.2f}")
        
        with col4:
            st.metric("Avg Supplier Rating", f"{avg_rating:.2f}")
    
    except Exception as e:
        st.error(f"Error loading dashboard metrics: {e}")
    
    # AI Capabilities Status
    st.subheader("🤖 AI Capabilities Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.ai_client.openai_client:
            st.success("✅ GPT-3.5 Turbo Available")
        else:
            st.warning("⚠️ GPT-3.5 Turbo Not Configured")
    
    with col2:
        if st.session_state.ai_client.index is not None:
            st.success("✅ Vector Search Ready")
        else:
            st.info("ℹ️ Vector Search Not Built")
    
    with col3:
        if st.session_state.ai_client.mcp_server_url:
            st.success("✅ MCP Server Connected")
        else:
            st.info("ℹ️ MCP Server Not Configured")
    
    # Recent activity
    st.subheader("📈 Recent Activity")
    
    try:
        # Get recent purchases
        with st.session_state.tidb_client.connection.cursor() as cursor:
            cursor.execute("""
                SELECT ph.*, s.supplier_name 
                FROM purchase_history ph 
                JOIN suppliers s ON ph.supplier_id = s.supplier_id 
                ORDER BY ph.purchase_date DESC 
                LIMIT 10
            """)
            recent_purchases = cursor.fetchall()
        
        if recent_purchases:
            df_recent = pd.DataFrame(recent_purchases)
            st.dataframe(
                df_recent[['purchase_date', 'supplier_name', 'product_name', 'total_cost']],
                use_container_width=True
            )
        else:
            st.info("No recent purchases found")
    
    except Exception as e:
        st.error(f"Error loading recent activity: {e}")

def show_data_upload():
    """Show data upload interface."""
    st.header("📊 Data Upload")
    
    # File upload section
    st.subheader("Upload CSV Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Suppliers Data")
        suppliers_file = st.file_uploader(
            "Upload suppliers CSV",
            type=['csv'],
            help="File should contain: supplier_id, supplier_name, category, rating, lead_time_days, min_order_quantity, price_per_unit, location, payment_terms, contract_end_date, risk_level"
        )
        
        if suppliers_file:
            try:
                df_suppliers = pd.read_csv(suppliers_file)
                st.success(f"✅ Loaded {len(df_suppliers)} supplier records")
                st.dataframe(df_suppliers.head(), use_container_width=True)
                
                if st.button("📥 Ingest Suppliers Data"):
                    with st.spinner("Ingesting suppliers data..."):
                        if st.session_state.tidb_client.ingest_suppliers(df_suppliers):
                            st.success("✅ Suppliers data ingested successfully!")
                            
                            # Build vector index
                            with st.spinner("Building vector index..."):
                                suppliers_data = df_suppliers.to_dict('records')
                                if st.session_state.ai_client.build_vector_index(suppliers_data):
                                    st.success("✅ Vector index built successfully!")
                                else:
                                    st.warning("⚠️ Vector index build failed")
                        else:
                            st.error("❌ Failed to ingest suppliers data")
            except Exception as e:
                st.error(f"Error reading suppliers file: {e}")
    
    with col2:
        st.markdown("### Purchase History Data")
        purchase_file = st.file_uploader(
            "Upload purchase history CSV",
            type=['csv'],
            help="File should contain: purchase_id, supplier_id, product_name, quantity, unit_cost, total_cost, purchase_date, delivery_date, status, product_category"
        )
        
        if purchase_file:
            try:
                df_purchase = pd.read_csv(purchase_file)
                st.success(f"✅ Loaded {len(df_purchase)} purchase records")
                st.dataframe(df_purchase.head(), use_container_width=True)
                
                if st.button("📥 Ingest Purchase History"):
                    with st.spinner("Ingesting purchase history..."):
                        if st.session_state.tidb_client.ingest_purchase_history(df_purchase):
                            st.success("✅ Purchase history ingested successfully!")
                        else:
                            st.error("❌ Failed to ingest purchase history")
            except Exception as e:
                st.error(f"Error reading purchase file: {e}")
    
    # Performance data upload
    st.subheader("Delivery Performance Data")
    performance_file = st.file_uploader(
        "Upload delivery performance CSV",
        type=['csv'],
        help="File should contain: supplier_id, on_time_delivery_rate, quality_score, communication_rating, response_time_hours, defect_rate, cost_variance_percent, overall_performance_score"
    )
    
    if performance_file:
        try:
            df_performance = pd.read_csv(performance_file)
            st.success(f"✅ Loaded {len(df_performance)} performance records")
            st.dataframe(df_performance.head(), use_container_width=True)
            
            if st.button("📥 Ingest Performance Data"):
                with st.spinner("Ingesting performance data..."):
                    if st.session_state.tidb_client.ingest_delivery_performance(df_performance):
                        st.success("✅ Performance data ingested successfully!")
                    else:
                        st.error("❌ Failed to ingest performance data")
        except Exception as e:
            st.error(f"Error reading performance file: {e}")

def show_ai_analysis():
    """Show AI-enhanced analysis interface."""
    st.header("🔍 AI-Enhanced Procurement Analysis")
    
    # Analysis parameters
    st.subheader("Analysis Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        product_name = st.text_input("Product Name", placeholder="e.g., Microcontroller Board")
        quantity = st.number_input("Required Quantity", min_value=1, value=100)
        category = st.selectbox(
            "Product Category",
            ["", "Electronics", "Industrial", "Automotive", "Chemicals", "Logistics", "General"]
        )
    
    with col2:
        budget = st.number_input("Budget Limit ($)", min_value=0, value=10000)
        analysis_type = st.selectbox(
            "Analysis Type",
            ["AI-Enhanced (GPT-3.5 + Vector Search)", "Traditional Rule-Based", "Hybrid Approach"]
        )
    
    # AI Configuration
    st.subheader("🤖 AI Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_gpt = st.checkbox("Use GPT-3.5 Turbo", value=True, disabled=not st.session_state.ai_client.openai_client)
        if not st.session_state.ai_client.openai_client:
            st.caption("⚠️ OpenAI not configured")
    
    with col2:
        use_vector_search = st.checkbox("Use Vector Search", value=True, disabled=st.session_state.ai_client.index is None)
        if st.session_state.ai_client.index is None:
            st.caption("⚠️ Vector index not built")
    
    with col3:
        use_mcp = st.checkbox("Use MCP Server", value=True, disabled=not st.session_state.ai_client.mcp_server_url)
        if not st.session_state.ai_client.mcp_server_url:
            st.caption("⚠️ MCP server not configured")
    
    # Run analysis button
    if st.button("🚀 Run AI-Enhanced Analysis", type="primary"):
        if not product_name:
            st.error("Please enter a product name")
            return
        
        with st.spinner("Running AI-enhanced procurement analysis..."):
            try:
                # Get AI-enhanced recommendations
                enhanced_results = st.session_state.enhanced_engine.get_ai_enhanced_recommendations(
                    product_name=product_name,
                    quantity=quantity,
                    category=category if category else None,
                    budget=budget if budget > 0 else None
                )
                
                # Store results in session state
                st.session_state.enhanced_recommendations = enhanced_results
                
                if "error" not in enhanced_results:
                    st.success(f"✅ AI analysis complete! Found {len(enhanced_results.get('recommendations', []))} recommendations")
                    
                    # Show quick summary
                    if enhanced_results.get('recommendations'):
                        st.subheader("📊 Quick Summary")
                        top_rec = enhanced_results['recommendations'][0]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Top Supplier", top_rec['supplier_name'])
                        with col2:
                            st.metric("Total Cost", f"${top_rec['total_cost']:,.2f}")
                        with col3:
                            st.metric("AI Score", f"{top_rec['ai_scores']['ai_enhanced_score']:.2f}")
                        with col4:
                            st.metric("Risk Score", f"{top_rec['ai_scores']['risk_score']:.2f}")
                else:
                    st.error(f"❌ Analysis failed: {enhanced_results['error']}")
                
            except Exception as e:
                st.error(f"Error running AI analysis: {e}")

def show_enhanced_results():
    """Show AI-enhanced analysis results."""
    st.header("📈 AI-Enhanced Analysis Results")
    
    if not st.session_state.enhanced_recommendations:
        st.info("No AI analysis results available. Please run an AI-enhanced analysis first.")
        return
    
    enhanced_results = st.session_state.enhanced_recommendations
    
    if "error" in enhanced_results:
        st.error(f"Analysis error: {enhanced_results['error']}")
        return
    
    # Results summary
    st.subheader("📊 Results Summary")
    
    recommendations = enhanced_results.get('recommendations', [])
    ai_analysis = enhanced_results.get('ai_analysis', {})
    market_analysis = enhanced_results.get('market_analysis', {})
    
    # AI Models Used
    st.markdown("### 🤖 AI Models Used")
    ai_models = enhanced_results.get('ai_models_used', {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Vector Search:** {ai_models.get('vector_search', 'N/A')}")
    with col2:
        st.info(f"**LLM:** {ai_models.get('llm', 'N/A')}")
    with col3:
        st.info(f"**MCP:** {'✅' if ai_models.get('mcp') else '❌'}")
    
    # Top recommendations table
    st.markdown("### 🏆 AI-Enhanced Supplier Recommendations")
    
    # Create a formatted dataframe for display
    display_data = []
    for i, rec in enumerate(recommendations, 1):
        display_data.append({
            'Rank': i,
            'Supplier': rec['supplier_name'],
            'Category': rec['category'],
            'AI Score': f"{rec['ai_scores']['ai_enhanced_score']:.2f}",
            'Traditional Score': f"{rec['ai_scores']['traditional_score']:.2f}",
            'Risk Score': f"{rec['ai_scores']['risk_score']:.2f}",
            'Cost': f"${rec['total_cost']:,.2f}",
            'Lead Time': f"{rec['lead_time_days']} days",
            'Location': rec['location']
        })
    
    df_display = pd.DataFrame(display_data)
    st.dataframe(df_display, use_container_width=True)
    
    # Visualizations
    st.subheader("📊 AI-Enhanced Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # AI Score vs Traditional Score
        ai_scores = [rec['ai_scores']['ai_enhanced_score'] for rec in recommendations]
        traditional_scores = [rec['ai_scores']['traditional_score'] for rec in recommendations]
        supplier_names = [rec['supplier_name'] for rec in recommendations]
        
        fig_comparison = go.Figure()
        fig_comparison.add_trace(go.Bar(
            x=supplier_names,
            y=ai_scores,
            name='AI-Enhanced Score',
            marker_color='lightblue'
        ))
        fig_comparison.add_trace(go.Bar(
            x=supplier_names,
            y=traditional_scores,
            name='Traditional Score',
            marker_color='lightcoral'
        ))
        fig_comparison.update_layout(
            title="AI vs Traditional Scoring Comparison",
            xaxis_title="Suppliers",
            yaxis_title="Score",
            barmode='group'
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    with col2:
        # Risk vs AI Score scatter plot
        risk_scores = [rec['ai_scores']['risk_score'] for rec in recommendations]
        fig_scatter = px.scatter(
            x=ai_scores,
            y=risk_scores,
            text=supplier_names,
            title="AI Score vs Risk Score",
            labels={'x': 'AI-Enhanced Score', 'y': 'Risk Score'}
        )
        fig_scatter.update_traces(textposition="top center")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Market analysis
    if market_analysis:
        st.subheader("📈 Market Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Market Spend", f"${market_analysis.get('total_spent', 0):,.2f}")
        
        with col2:
            st.metric("Cost Trend", market_analysis.get('cost_trend', 'Unknown'))
        
        with col3:
            st.metric("Demand Trend", market_analysis.get('demand_trend', 'Unknown'))

def show_ai_insights():
    """Show AI-generated insights and analysis."""
    st.header("🤖 AI-Generated Insights")
    
    if not st.session_state.enhanced_recommendations:
        st.info("No AI analysis results available. Please run an AI-enhanced analysis first.")
        return
    
    enhanced_results = st.session_state.enhanced_recommendations
    
    if "error" in enhanced_results:
        st.error(f"Analysis error: {enhanced_results['error']}")
        return
    
    ai_analysis = enhanced_results.get('ai_analysis', {})
    
    # GPT Analysis
    if ai_analysis.get('gpt_analysis'):
        st.subheader("🧠 GPT-3.5 Turbo Analysis")
        
        gpt_analysis = ai_analysis['gpt_analysis']
        
        if isinstance(gpt_analysis, dict):
            # Structured GPT analysis
            if 'top_recommendations' in gpt_analysis:
                st.markdown("### Top AI Recommendations")
                for i, rec in enumerate(gpt_analysis['top_recommendations'], 1):
                    with st.expander(f"{i}. {rec.get('supplier_name', 'Unknown')}"):
                        st.write(f"**Reasoning:** {rec.get('reasoning', 'N/A')}")
                        st.write(f"**Risk Level:** {rec.get('risk_level', 'N/A')}")
                        st.write(f"**Estimated Cost:** {rec.get('estimated_cost', 'N/A')}")
                        st.write(f"**Confidence Score:** {rec.get('confidence_score', 'N/A')}")
            
            if 'risk_assessment' in gpt_analysis:
                st.markdown("### Risk Assessment")
                st.info(gpt_analysis['risk_assessment'])
            
            if 'cost_analysis' in gpt_analysis:
                st.markdown("### Cost-Benefit Analysis")
                st.info(gpt_analysis['cost_analysis'])
            
            if 'strategic_insights' in gpt_analysis:
                st.markdown("### Strategic Insights")
                st.success(gpt_analysis['strategic_insights'])
        else:
            # Text-based GPT analysis
            st.info(gpt_analysis)
    
    # MCP Analysis
    if ai_analysis.get('mcp_analysis'):
        st.subheader("🔗 MCP Server Insights")
        mcp_analysis = ai_analysis['mcp_analysis']
        
        if isinstance(mcp_analysis, dict):
            for key, value in mcp_analysis.items():
                st.write(f"**{key.title()}:** {value}")
        else:
            st.info(str(mcp_analysis))
    
    # Search Results Analysis
    if ai_analysis.get('search_results'):
        st.subheader("🔍 Vector Search Analysis")
        
        search_results = ai_analysis['search_results']
        
        # Show top search results
        if search_results:
            st.markdown("### Top Vector Search Results")
            
            search_data = []
            for i, result in enumerate(search_results[:5], 1):
                search_data.append({
                    'Rank': i,
                    'Supplier': result.get('supplier_name', 'Unknown'),
                    'Similarity Score': f"{result.get('similarity_score', 0):.3f}",
                    'Category': result.get('category', 'Unknown'),
                    'Location': result.get('location', 'Unknown')
                })
            
            df_search = pd.DataFrame(search_data)
            st.dataframe(df_search, use_container_width=True)
    
    # Analysis Metadata
    st.subheader("📊 Analysis Metadata")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Analysis Timestamp:** {enhanced_results.get('analysis_timestamp', 'N/A')}")
        st.write(f"**Total Recommendations:** {len(enhanced_results.get('recommendations', []))}")
    
    with col2:
        st.write(f"**Query:** {enhanced_results.get('query_info', {}).get('product_name', 'N/A')}")
        st.write(f"**Quantity:** {enhanced_results.get('query_info', {}).get('quantity', 'N/A')}")

def show_external_tools():
    """Show external tools integration."""
    st.header("🔗 External Tools Integration")
    
    if not st.session_state.enhanced_recommendations:
        st.info("No AI analysis results available. Please run an AI-enhanced analysis first.")
        return
    
    enhanced_results = st.session_state.enhanced_recommendations
    
    if "error" in enhanced_results:
        st.error(f"Analysis error: {enhanced_results['error']}")
        return
    
    recommendations = enhanced_results.get('recommendations', [])
    market_analysis = enhanced_results.get('market_analysis', {})
    
    # Notification section
    st.subheader("📢 AI-Enhanced Notifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Slack Notifications")
        if st.button("📤 Send AI-Enhanced Slack Notification"):
            with st.spinner("Sending AI-enhanced Slack notification..."):
                success = False
                try:
                    # Create enhanced summary
                    summary = f"🤖 AI-Enhanced Procurement Analysis Complete!\n\n"
                    summary += f"Found {len(recommendations)} AI-optimized supplier recommendations.\n\n"
                    if recommendations:
                        top_rec = recommendations[0]
                        summary += f"**Top AI Recommendation:** {top_rec['supplier_name']}\n"
                        summary += f"**AI Score:** {top_rec['ai_scores']['ai_enhanced_score']:.2f}\n"
                        summary += f"**Total Cost:** ${top_rec['total_cost']:,.2f}\n"
                        summary += f"**Risk Score:** {top_rec['ai_scores']['risk_score']:.2f}\n\n"
                    summary += f"**AI Models Used:**\n"
                    ai_models = enhanced_results.get('ai_models_used', {})
                    summary += f"- Vector Search: {ai_models.get('vector_search', 'N/A')}\n"
                    summary += f"- LLM: {ai_models.get('llm', 'N/A')}\n"
                    summary += f"- MCP: {'✅' if ai_models.get('mcp') else '❌'}"
                    success = st.session_state.external_tools.send_slack_notification(
                        summary,
                        recommendations
                    )
                except Exception as e:
                    st.warning(f"Slack notification error: {e}")
                    success = False
                if success:
                    st.success("✅ AI-enhanced Slack notification sent!")
                else:
                    st.warning("⚠️ Slack notification failed (check configuration)")
    
    with col2:
        st.markdown("### Email Notifications")
        email_recipients = st.text_input(
            "Email Recipients (comma-separated)",
            placeholder="user1@company.com, user2@company.com"
        )
        if st.button("📧 Send AI-Enhanced Email"):
            if not email_recipients:
                st.error("Please enter email recipients")
                return
            with st.spinner("Sending AI-enhanced email..."):
                success = False
                try:
                    recipients = [email.strip() for email in email_recipients.split(',')]
                    # Create enhanced summary
                    summary = f"AI-Enhanced Procurement Analysis Results\n\n"
                    summary += f"Analysis completed using advanced AI models including GPT-3.5 Turbo and vector search.\n\n"
                    summary += f"Total recommendations: {len(recommendations)}\n"
                    success = st.session_state.external_tools.send_email_notification(
                        "🤖 AI-Enhanced Procurement Recommendations Available",
                        summary,
                        recipients,
                        recommendations
                    )
                except Exception as e:
                    st.warning(f"Email notification error: {e}")
                    success = False
                if success:
                    st.success("✅ AI-enhanced email notification sent!")
                else:
                    st.warning("⚠️ Email notification failed (check configuration)")
    
    # Export section
    st.subheader("📤 AI-Enhanced Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Google Sheets Export")
        
        if st.button("📊 Export AI Results to Google Sheets"):
            with st.spinner("Exporting AI-enhanced results to Google Sheets..."):
                # Prepare enhanced data for export
                export_data = []
                for rec in recommendations:
                    export_rec = {
                        'supplier_id': rec.get('supplier_id', ''),
                        'supplier_name': rec.get('supplier_name', ''),
                        'category': rec.get('category', ''),
                        'risk_score': rec['ai_scores']['risk_score'],
                        'total_cost': rec.get('total_cost', 0),
                        'lead_time_days': rec.get('lead_time_days', 0),
                        'location': rec.get('location', ''),
                        }
                    export_data.append(export_rec)
                
                sheet_url = st.session_state.external_tools.export_to_google_sheets(export_data)
                
                if sheet_url:
                    st.success("✅ AI-enhanced results exported to Google Sheets!")
                    st.markdown(f"[Open Google Sheet]({sheet_url})")
                else:
                    st.warning("⚠️ Google Sheets export failed (check configuration)")
    
    with col2:
        st.markdown("### CSV Export")
        
        if st.button("📄 Export AI Results to CSV"):
            with st.spinner("Exporting AI-enhanced results to CSV..."):
                # Prepare enhanced data for export
                export_data = []
                for rec in recommendations:
                    export_rec = {
                        'supplier_id': rec.get('supplier_id', ''),
                        'supplier_name': rec.get('supplier_name', ''),
                        'category': rec.get('category', ''),
                        'risk_score': rec['ai_scores']['risk_score'],
                        'total_cost': rec.get('total_cost', 0),
                        'lead_time_days': rec.get('lead_time_days', 0),
                        'location': rec.get('location', ''),
                       }
                    export_data.append(export_rec)
                
                csv_file = st.session_state.external_tools.export_to_csv(export_data)
                
                if csv_file:
                    st.success("✅ AI-enhanced results exported to CSV!")
                    
                    # Read and display the CSV for download
                    with open(csv_file, 'r') as f:
                        csv_data = f.read()
                    
                    st.download_button(
                        label="📥 Download AI-Enhanced CSV",
                        data=csv_data,
                        file_name=csv_file,
                        mime="text/csv"
                    )
                else:
                    st.error("❌ CSV export failed")

def show_debugger():
    # Debug section (can be hidden in production)
    with st.expander("🔧 Debug Information"):
        if st.session_state.enhanced_engine:
            debug_info = st.session_state.enhanced_engine.get_debug_info()
            st.write("Enhanced Engine Status:", debug_info)
            
            # Check AI client vector status
            if (st.session_state.enhanced_engine.ai_client and 
                hasattr(st.session_state.enhanced_engine.ai_client, 'index')):
                vector_status = {
                    'vector_index_built': st.session_state.enhanced_engine.ai_client.index is not None,
                    'suppliers_in_index': len(st.session_state.enhanced_engine.ai_client.supplier_data) 
                                        if hasattr(st.session_state.enhanced_engine.ai_client, 'supplier_data') else 0
                }
                st.write("Vector Index Status:", vector_status)
            
            # Check suppliers data
            suppliers_data = st.session_state.enhanced_engine.get_all_suppliers_data()
            st.write(f"Suppliers Data from TiDB: {len(suppliers_data)} records")
            if suppliers_data:
                st.write("Sample Supplier:", suppliers_data[0])
        

if __name__ == "__main__":
    main()
