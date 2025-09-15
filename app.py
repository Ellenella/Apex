import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import logging
import os

# Import our custom modules
from ti_db_client import TiDBClient
from recommendation_engine import RecommendationEngine
from external_tools import ExternalTools

# Configure page
st.set_page_config(
    page_title="AI-Driven Procurement Optimization Agent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'tidb_client' not in st.session_state:
    st.session_state.tidb_client = None
if 'recommendation_engine' not in st.session_state:
    st.session_state.recommendation_engine = None
if 'external_tools' not in st.session_state:
    st.session_state.external_tools = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'market_analysis' not in st.session_state:
    st.session_state.market_analysis = {}

def initialize_components():
    """Initialize TiDB client and other components."""
    if st.session_state.tidb_client is None:
        st.session_state.tidb_client = TiDBClient()
        if not st.session_state.tidb_client.connect():
            st.error("Failed to connect to TiDB. Please check your configuration.")
            return False
    
    if st.session_state.recommendation_engine is None:
        st.session_state.recommendation_engine = RecommendationEngine(st.session_state.tidb_client)
    
    if st.session_state.external_tools is None:
        st.session_state.external_tools = ExternalTools()
    
    return True

def main():
    """Main application function."""
    
    # Header
    st.title("🚀 AI-Driven Procurement Optimization Agent")
    st.markdown("**Multi-step AI agent for data-driven supplier selection and procurement optimization**")
    
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
        
        # Navigation
        st.header("📋 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Dashboard", "📊 Data Upload", "🔍 Analysis", "📈 Results", "🔗 External Tools"]
        )
    
    # Page routing
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📊 Data Upload":
        show_data_upload()
    elif page == "🔍 Analysis":
        show_analysis()
    elif page == "📈 Results":
        show_results()
    elif page == "🔗 External Tools":
        show_external_tools()

def show_dashboard():
    """Show the main dashboard."""
    st.header("🏠 Procurement Dashboard")
    
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

def show_analysis():
    """Show analysis interface."""
    st.header("🔍 Procurement Analysis")
    
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
        priority = st.selectbox(
            "Priority",
            ["Cost", "Quality", "Speed", "Balanced"]
        )
    
    # Run analysis button
    if st.button("🚀 Run Analysis", type="primary"):
        if not product_name:
            st.error("Please enter a product name")
            return
        
        with st.spinner("Running procurement analysis..."):
            try:
                # Get recommendations
                recommendations = st.session_state.recommendation_engine.get_supplier_recommendations(
                    product_name=product_name,
                    quantity=quantity,
                    category=category if category else None,
                    budget=budget if budget > 0 else None
                )
                
                # Get market analysis
                market_analysis = st.session_state.recommendation_engine.analyze_market_trends(
                    product_category=category if category else None
                )
                
                # Store results in session state
                st.session_state.recommendations = recommendations
                st.session_state.market_analysis = market_analysis
                
                st.success(f"✅ Analysis complete! Found {len(recommendations)} recommendations")
                
                # Show quick summary
                if recommendations:
                    st.subheader("📊 Quick Summary")
                    top_rec = recommendations[0]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Top Supplier", top_rec['supplier_name'])
                    with col2:
                        st.metric("Total Cost", f"${top_rec['total_cost']:,.2f}")
                    with col3:
                        st.metric("Risk Score", f"{top_rec['risk_score']:.2f}")
                
            except Exception as e:
                st.error(f"Error running analysis: {e}")

def show_results():
    """Show analysis results."""
    st.header("📈 Analysis Results")
    
    if not st.session_state.recommendations:
        st.info("No analysis results available. Please run an analysis first.")
        return
    
    # Results summary
    st.subheader("📊 Results Summary")
    
    recommendations = st.session_state.recommendations
    market_analysis = st.session_state.market_analysis
    
    # Top recommendations table
    st.markdown("### 🏆 Top Supplier Recommendations")
    
    # Create a formatted dataframe for display
    display_data = []
    for i, rec in enumerate(recommendations, 1):
        display_data.append({
            'Rank': i,
            'Supplier': rec['supplier_name'],
            'Category': rec['category'],
            'Score': f"{rec['supplier_score']:.2f}",
            'Risk': f"{rec['risk_score']:.2f}",
            'Cost': f"${rec['total_cost']:,.2f}",
            'Lead Time': f"{rec['lead_time_days']} days",
            'Location': rec['location']
        })
    
    df_display = pd.DataFrame(display_data)
    st.dataframe(df_display, use_container_width=True)
    
    # Visualizations
    st.subheader("📊 Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cost vs Score scatter plot
        fig_scatter = px.scatter(
            recommendations,
            x='supplier_score',
            y='total_cost',
            size='risk_score',
            color='category',
            hover_data=['supplier_name', 'lead_time_days'],
            title="Cost vs Score Analysis",
            labels={'supplier_score': 'Supplier Score', 'total_cost': 'Total Cost ($)'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Risk distribution
        risk_data = [rec['risk_score'] for rec in recommendations]
        fig_hist = px.histogram(
            x=risk_data,
            title="Risk Score Distribution",
            labels={'x': 'Risk Score', 'y': 'Count'},
            nbins=10
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
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
        
        # Market trends chart
        if 'monthly_data' in market_analysis and market_analysis['monthly_data']:
            monthly_data = market_analysis['monthly_data']
            df_monthly = pd.DataFrame(monthly_data)
            
            fig_trends = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Monthly Spend', 'Monthly Orders'),
                vertical_spacing=0.1
            )
            
            fig_trends.add_trace(
                go.Scatter(x=df_monthly['month'], y=df_monthly['total_spent'], name='Total Spend'),
                row=1, col=1
            )
            
            fig_trends.add_trace(
                go.Scatter(x=df_monthly['month'], y=df_monthly['order_count'], name='Order Count'),
                row=2, col=1
            )
            
            fig_trends.update_layout(height=500, title_text="Market Trends")
            st.plotly_chart(fig_trends, use_container_width=True)

def show_external_tools():
    """Show external tools integration."""
    st.header("🔗 External Tools Integration")
    
    if not st.session_state.recommendations:
        st.info("No recommendations available. Please run an analysis first.")
        return
    
    # Notification section
    st.subheader("📢 Notifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Slack Notifications")
        
        if st.button("📤 Send Slack Notification"):
            with st.spinner("Sending Slack notification..."):
                summary = st.session_state.external_tools.create_notification_summary(
                    st.session_state.recommendations,
                    st.session_state.market_analysis
                )
                
                success = st.session_state.external_tools.send_slack_notification(
                    summary,
                    st.session_state.recommendations
                )
                
                if success:
                    st.success("✅ Slack notification sent!")
                else:
                    st.warning("⚠️ Slack notification failed (check configuration)")
    
    with col2:
        st.markdown("### Email Notifications")
        
        email_recipients = st.text_input(
            "Email Recipients (comma-separated)",
            placeholder="user1@company.com, user2@company.com"
        )
        
        if st.button("📧 Send Email Notification"):
            if not email_recipients:
                st.error("Please enter email recipients")
                return
            
            with st.spinner("Sending email notification..."):
                recipients = [email.strip() for email in email_recipients.split(',')]
                summary = st.session_state.external_tools.create_notification_summary(
                    st.session_state.recommendations,
                    st.session_state.market_analysis
                )
                
                success = st.session_state.external_tools.send_email_notification(
                    "Procurement Recommendations Available",
                    summary,
                    recipients,
                    st.session_state.recommendations
                )
                
                if success:
                    st.success("✅ Email notification sent!")
                else:
                    st.warning("⚠️ Email notification failed (check configuration)")
    
    # Export section
    st.subheader("📤 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Google Sheets Export")
        
        if st.button("📊 Export to Google Sheets"):
            with st.spinner("Exporting to Google Sheets..."):
                sheet_url = st.session_state.external_tools.export_to_google_sheets(
                    st.session_state.recommendations
                )
                
                if sheet_url:
                    st.success("✅ Exported to Google Sheets!")
                    st.markdown(f"[Open Google Sheet]({sheet_url})")
                else:
                    st.warning("⚠️ Google Sheets export failed (check configuration)")
    
    with col2:
        st.markdown("### CSV Export")
        
        if st.button("📄 Export to CSV"):
            with st.spinner("Exporting to CSV..."):
                csv_file = st.session_state.external_tools.export_to_csv(
                    st.session_state.recommendations
                )
                
                if csv_file:
                    st.success("✅ Exported to CSV!")
                    
                    # Read and display the CSV for download
                    with open(csv_file, 'r') as f:
                        csv_data = f.read()
                    
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=csv_file,
                        mime="text/csv"
                    )
                else:
                    st.error("❌ CSV export failed")
    
    # Comprehensive alert
    st.subheader("🚨 Comprehensive Alert")
    
    alert_type = st.selectbox(
        "Alert Type",
        ["recommendation", "urgent", "analysis_complete"]
    )
    
    if st.button("🚀 Send Comprehensive Alert"):
        with st.spinner("Sending comprehensive alert..."):
            results = st.session_state.external_tools.send_procurement_alert(
                st.session_state.recommendations,
                st.session_state.market_analysis,
                alert_type
            )
            
            if 'error' not in results:
                st.success("✅ Comprehensive alert sent!")
                
                # Show results
                st.markdown("### Alert Results:")
                for channel, success in results.items():
                    status = "✅" if success else "❌"
                    st.write(f"{status} {channel.title()}")
            else:
                st.error(f"❌ Alert failed: {results['error']}")

if __name__ == "__main__":
    main()
