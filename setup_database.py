#!/usr/bin/env python3
"""
Database setup script for TiDB Procurement System
This script creates the necessary tables and loads sample data.
"""

import os
import pandas as pd
import logging
from dotenv import load_dotenv
from ti_db_client import TiDBClient

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main setup function."""
    logger.info("Starting TiDB database setup...")
    
    # Initialize TiDB client
    tidb_client = TiDBClient()
    
    # Test connection
    if not tidb_client.connect():
        logger.error("Failed to connect to TiDB. Please check your configuration.")
        return False
    
    logger.info("Successfully connected to TiDB")
    
    # Create tables
    logger.info("Creating database tables...")
    if not tidb_client.create_tables():
        logger.error("Failed to create tables")
        return False
    
    logger.info("Tables created successfully")
    
    # Load sample data
    logger.info("Loading sample data...")
    
    # Load suppliers data
    suppliers_file = "sample_data/suppliers.csv"
    if os.path.exists(suppliers_file):
        try:
            suppliers_df = pd.read_csv(suppliers_file)
            if tidb_client.ingest_suppliers(suppliers_df):
                logger.info(f"Loaded {len(suppliers_df)} supplier records")
            else:
                logger.error("Failed to load suppliers data")
        except Exception as e:
            logger.error(f"Error loading suppliers data: {e}")
    else:
        logger.warning(f"Suppliers file not found: {suppliers_file}")
    
    # Load purchase history data
    purchase_file = "sample_data/purchase_history.csv"
    if os.path.exists(purchase_file):
        try:
            purchase_df = pd.read_csv(purchase_file)
            if tidb_client.ingest_purchase_history(purchase_df):
                logger.info(f"Loaded {len(purchase_df)} purchase history records")
            else:
                logger.error("Failed to load purchase history data")
        except Exception as e:
            logger.error(f"Error loading purchase history data: {e}")
    else:
        logger.warning(f"Purchase history file not found: {purchase_file}")
    
    # Load delivery performance data
    performance_file = "sample_data/delivery_performance.csv"
    if os.path.exists(performance_file):
        try:
            performance_df = pd.read_csv(performance_file)
            if tidb_client.ingest_delivery_performance(performance_df):
                logger.info(f"Loaded {len(performance_df)} delivery performance records")
            else:
                logger.error("Failed to load delivery performance data")
        except Exception as e:
            logger.error(f"Error loading delivery performance data: {e}")
    else:
        logger.warning(f"Delivery performance file not found: {performance_file}")
    
    # Verify data loading
    logger.info("Verifying data loading...")
    try:
        with tidb_client.connection.cursor() as cursor:
            # Check suppliers
            cursor.execute("SELECT COUNT(*) as count FROM suppliers")
            supplier_count = cursor.fetchone()['count']
            logger.info(f"Suppliers in database: {supplier_count}")
            
            # Check purchase history
            cursor.execute("SELECT COUNT(*) as count FROM purchase_history")
            purchase_count = cursor.fetchone()['count']
            logger.info(f"Purchase history records in database: {purchase_count}")
            
            # Check delivery performance
            cursor.execute("SELECT COUNT(*) as count FROM delivery_performance")
            performance_count = cursor.fetchone()['count']
            logger.info(f"Delivery performance records in database: {performance_count}")
            
    except Exception as e:
        logger.error(f"Error verifying data: {e}")
    
    # Close connection
    tidb_client.disconnect()
    
    logger.info("Database setup completed successfully!")
    logger.info("You can now run the Streamlit application with: streamlit run app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Database setup completed successfully!")
        print("📊 Sample data loaded and ready for analysis")
        print("🚀 Run 'streamlit run app.py' to start the application")
    else:
        print("\n❌ Database setup failed!")
        print("Please check your TiDB configuration and try again")
