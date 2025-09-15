import os
import pymysql
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

class TiDBClient:
    def __init__(self):
        """Initialize TiDB connection using environment variables."""
        print("Database connection initialized")
        self.host = os.getenv('TIDB_HOST')
        self.port = int(os.getenv('TIDB_PORT', 4000))
        self.user = os.getenv('TIDB_USER')
        self.password = os.getenv('TIDB_PASSWORD')
        self.database = os.getenv('TIDB_DATABASE', 'test')
        self.ssl_ca=os.getenv("TIDB_CA")
        self.connection = None
        self.scaler = StandardScaler()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def connect(self):
        """Establish connection to TiDB."""
        try:
            ssl_config = None
            if self.ssl_ca and os.path.exists(self.ssl_ca):
                ssl_config = {'ca': self.ssl_ca}
            else:
                self.logger.warning("SSL CA certificate not found. Connection may fail.")

            self.connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                ssl=ssl_config,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            self.logger.info("Successfully connected to TiDB")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to TiDB: {e}")
            return False
    
    def disconnect(self):
        """Close TiDB connection."""
        if self.connection:
            self.connection.close()
            self.logger.info("Disconnected from TiDB")
    
    def create_tables(self):
        """Create necessary tables for the procurement system."""
        if not self.connection:
            if not self.connect():
                return False
        
        try:
            with self.connection.cursor() as cursor:
                # Suppliers table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS suppliers (
                        supplier_id VARCHAR(20) PRIMARY KEY,
                        supplier_name VARCHAR(255) NOT NULL,
                        category VARCHAR(100),
                        rating DECIMAL(3,2),
                        lead_time_days INT,
                        min_order_quantity INT,
                        price_per_unit DECIMAL(10,2),
                        location VARCHAR(100),
                        payment_terms VARCHAR(50),
                        contract_end_date DATE,
                        risk_level VARCHAR(20),
                        embedding_vector TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Purchase history table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS purchase_history (
                        purchase_id VARCHAR(20) PRIMARY KEY,
                        supplier_id VARCHAR(20),
                        product_name VARCHAR(255),
                        quantity INT,
                        unit_cost DECIMAL(10,2),
                        total_cost DECIMAL(12,2),
                        purchase_date DATE,
                        delivery_date DATE,
                        status VARCHAR(50),
                        product_category VARCHAR(100),
                        embedding_vector TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id)
                    )
                """)
                
                # Delivery performance table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS delivery_performance (
                        supplier_id VARCHAR(20) PRIMARY KEY,
                        on_time_delivery_rate DECIMAL(3,2),
                        quality_score DECIMAL(3,2),
                        communication_rating DECIMAL(3,2),
                        response_time_hours DECIMAL(5,2),
                        defect_rate DECIMAL(5,4),
                        cost_variance_percent DECIMAL(5,2),
                        overall_performance_score DECIMAL(3,2),
                        embedding_vector TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id)
                    )
                """)
                
                # Create indexes for better search performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_suppliers_category ON suppliers(category)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_suppliers_rating ON suppliers(rating)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_purchase_product ON purchase_history(product_name)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_purchase_category ON purchase_history(product_category)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_purchase_date ON purchase_history(purchase_date)")
                
                self.connection.commit()
                self.logger.info("Tables created successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            return False
    
    def create_embedding_vector(self, data_dict, vector_type='supplier'):
        """Create normalized embedding vectors for vector search."""
        if vector_type == 'supplier':
            # Create vector from supplier features
            features = [
                float(data_dict.get('rating', 0)),
                float(data_dict.get('lead_time_days', 0)),
                float(data_dict.get('min_order_quantity', 0)),
                float(data_dict.get('price_per_unit', 0))
            ]
        elif vector_type == 'purchase':
            # Create vector from purchase features
            features = [
                float(data_dict.get('quantity', 0)),
                float(data_dict.get('unit_cost', 0)),
                float(data_dict.get('total_cost', 0))
            ]
        elif vector_type == 'performance':
            # Create vector from performance features
            features = [
                float(data_dict.get('on_time_delivery_rate', 0)),
                float(data_dict.get('quality_score', 0)),
                float(data_dict.get('communication_rating', 0)),
                float(data_dict.get('response_time_hours', 0)),
                float(data_dict.get('defect_rate', 0)),
                float(data_dict.get('cost_variance_percent', 0)),
                float(data_dict.get('overall_performance_score', 0))
            ]
        else:
            return None
        
        # Normalize features
        features_array = np.array(features).reshape(1, -1)
        normalized_features = self.scaler.fit_transform(features_array)
        
        # Convert to string for storage
        return ','.join(map(str, normalized_features.flatten()))
    
    def ingest_suppliers(self, df, max_retries=3):
        """Ingest supplier data into TiDB with retry logic for lost connection errors."""
        attempt = 0
        while attempt < max_retries:
            if not self.connection:
                if not self.connect():
                    return False
            try:
                with self.connection.cursor() as cursor:
                    for _, row in df.iterrows():
                        # Create embedding vector
                        embedding = self.create_embedding_vector(row.to_dict(), 'supplier')
                        cursor.execute("""
                            INSERT INTO suppliers (
                                supplier_id, supplier_name, category, rating, lead_time_days,
                                min_order_quantity, price_per_unit, location, payment_terms,
                                contract_end_date, risk_level, embedding_vector
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE
                                supplier_name = VALUES(supplier_name),
                                category = VALUES(category),
                                rating = VALUES(rating),
                                lead_time_days = VALUES(lead_time_days),
                                min_order_quantity = VALUES(min_order_quantity),
                                price_per_unit = VALUES(price_per_unit),
                                location = VALUES(location),
                                payment_terms = VALUES(payment_terms),
                                contract_end_date = VALUES(contract_end_date),
                                risk_level = VALUES(risk_level),
                                embedding_vector = VALUES(embedding_vector)
                        """, (
                            row['supplier_id'], row['supplier_name'], row['category'],
                            row['rating'], row['lead_time_days'], row['min_order_quantity'],
                            row['price_per_unit'], row['location'], row['payment_terms'],
                            row['contract_end_date'], row['risk_level'], embedding
                        ))
                    self.connection.commit()
                    self.logger.info(f"Ingested {len(df)} supplier records")
                    return True
            except Exception as e:
                # Check for MySQL lost connection error code 2013
                if hasattr(e, 'args') and len(e.args) > 0 and '2013' in str(e.args[0]):
                    self.logger.warning(f"Lost connection to MySQL server during query. Retrying ({attempt+1}/{max_retries})...")
                    self.disconnect()
                    attempt += 1
                    continue
                else:
                    self.logger.error(f"Error ingesting suppliers: {e}")
                    return False
        self.logger.error("Failed to ingest suppliers after multiple retries due to lost connection.")
        return False
    
    def ingest_purchase_history(self, df):
        """Ingest purchase history data into TiDB."""
        if not self.connection:
            if not self.connect():
                return False
        
        try:
            with self.connection.cursor() as cursor:
                for _, row in df.iterrows():
                    # Create embedding vector
                    embedding = self.create_embedding_vector(row.to_dict(), 'purchase')
                    
                    cursor.execute("""
                        INSERT INTO purchase_history (
                            purchase_id, supplier_id, product_name, quantity, unit_cost,
                            total_cost, purchase_date, delivery_date, status,
                            product_category, embedding_vector
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            supplier_id = VALUES(supplier_id),
                            product_name = VALUES(product_name),
                            quantity = VALUES(quantity),
                            unit_cost = VALUES(unit_cost),
                            total_cost = VALUES(total_cost),
                            purchase_date = VALUES(purchase_date),
                            delivery_date = VALUES(delivery_date),
                            status = VALUES(status),
                            product_category = VALUES(product_category),
                            embedding_vector = VALUES(embedding_vector)
                    """, (
                        row['purchase_id'], row['supplier_id'], row['product_name'],
                        row['quantity'], row['unit_cost'], row['total_cost'],
                        row['purchase_date'], row['delivery_date'], row['status'],
                        row['product_category'], embedding
                    ))
                
                self.connection.commit()
                self.logger.info(f"Ingested {len(df)} purchase history records")
                return True
                
        except Exception as e:
            self.logger.error(f"Error ingesting purchase history: {e}")
            return False
    
    def ingest_delivery_performance(self, df):
        """Ingest delivery performance data into TiDB."""
        if not self.connection:
            if not self.connect():
                return False
        
        try:
            with self.connection.cursor() as cursor:
                for _, row in df.iterrows():
                    # Create embedding vector
                    embedding = self.create_embedding_vector(row.to_dict(), 'performance')
                    
                    cursor.execute("""
                        INSERT INTO delivery_performance (
                            supplier_id, on_time_delivery_rate, quality_score,
                            communication_rating, response_time_hours, defect_rate,
                            cost_variance_percent, overall_performance_score, embedding_vector
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            on_time_delivery_rate = VALUES(on_time_delivery_rate),
                            quality_score = VALUES(quality_score),
                            communication_rating = VALUES(communication_rating),
                            response_time_hours = VALUES(response_time_hours),
                            defect_rate = VALUES(defect_rate),
                            cost_variance_percent = VALUES(cost_variance_percent),
                            overall_performance_score = VALUES(overall_performance_score),
                            embedding_vector = VALUES(embedding_vector)
                    """, (
                        row['supplier_id'], row['on_time_delivery_rate'],
                        row['quality_score'], row['communication_rating'],
                        row['response_time_hours'], row['defect_rate'],
                        row['cost_variance_percent'], row['overall_performance_score'],
                        embedding
                    ))
                
                self.connection.commit()
                self.logger.info(f"Ingested {len(df)} delivery performance records")
                return True
                
        except Exception as e:
            self.logger.error(f"Error ingesting delivery performance: {e}")
            return False
    
    def vector_search_suppliers(self, query_vector, category=None, limit=10):
        """Perform vector search for similar suppliers."""
        if not self.connection:
            if not self.connect():
                return []
        
        try:
            with self.connection.cursor() as cursor:
                # Convert query vector to string format
                query_vector_str = ','.join(map(str, query_vector))
                
                # Build query with optional category filter
                sql = """
                    SELECT s.*, 
                           dp.on_time_delivery_rate,
                           dp.quality_score,
                           dp.overall_performance_score,
                           dp.embedding_vector as performance_vector
                    FROM suppliers s
                    LEFT JOIN delivery_performance dp ON s.supplier_id = dp.supplier_id
                """
                
                params = []
                if category:
                    sql += " WHERE s.category = %s"
                    params.append(category)
                
                sql += " ORDER BY ABS(s.embedding_vector - %s) LIMIT %s"
                params.extend([query_vector_str, limit])
                
                cursor.execute(sql, params)
                results = cursor.fetchall()
                
                # Calculate similarity scores
                for result in results:
                    if result['embedding_vector']:
                        stored_vector = np.array([float(x) for x in result['embedding_vector'].split(',')])
                        similarity = cosine_similarity([query_vector], [stored_vector])[0][0]
                        result['similarity_score'] = similarity
                    else:
                        result['similarity_score'] = 0.0
                
                # Sort by similarity score
                results.sort(key=lambda x: x['similarity_score'], reverse=True)
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error in vector search: {e}")
            return []
    
    def full_text_search(self, search_term, search_type='product', limit=10):
        """Perform full-text search on products or suppliers."""
        if not self.connection:
            if not self.connect():
                return []
        
        try:
            with self.connection.cursor() as cursor:
                if search_type == 'product':
                    sql = """
                        SELECT ph.*, s.supplier_name, s.rating, s.lead_time_days
                        FROM purchase_history ph
                        JOIN suppliers s ON ph.supplier_id = s.supplier_id
                        WHERE ph.product_name LIKE %s OR ph.product_category LIKE %s
                        ORDER BY ph.purchase_date DESC
                        LIMIT %s
                    """
                    search_pattern = f"%{search_term}%"
                    cursor.execute(sql, (search_pattern, search_pattern, limit))
                else:  # supplier search
                    sql = """
                        SELECT s.*, dp.overall_performance_score
                        FROM suppliers s
                        LEFT JOIN delivery_performance dp ON s.supplier_id = dp.supplier_id
                        WHERE s.supplier_name LIKE %s OR s.category LIKE %s
                        ORDER BY s.rating DESC
                        LIMIT %s
                    """
                    search_pattern = f"%{search_term}%"
                    cursor.execute(sql, (search_pattern, search_pattern, limit))
                
                return cursor.fetchall()
                
        except Exception as e:
            self.logger.error(f"Error in full-text search: {e}")
            return []
    
    def get_supplier_performance(self, supplier_id):
        """Get comprehensive performance data for a specific supplier."""
        if not self.connection:
            if not self.connect():
                return None
        
        try:
            with self.connection.cursor() as cursor:
                sql = """
                    SELECT s.*, dp.*,
                           COUNT(ph.purchase_id) as total_orders,
                           AVG(ph.total_cost) as avg_order_value,
                           SUM(ph.quantity) as total_quantity_purchased
                    FROM suppliers s
                    LEFT JOIN delivery_performance dp ON s.supplier_id = dp.supplier_id
                    LEFT JOIN purchase_history ph ON s.supplier_id = ph.supplier_id
                    WHERE s.supplier_id = %s
                    GROUP BY s.supplier_id
                """
                cursor.execute(sql, (supplier_id,))
                return cursor.fetchone()
                
        except Exception as e:
            self.logger.error(f"Error getting supplier performance: {e}")
            return None
    
    def get_purchase_trends(self, supplier_id=None, product_category=None, months=6):
        """Get purchase trends for analysis."""
        if not self.connection:
            if not self.connect():
                return []
        
        try:
            with self.connection.cursor() as cursor:
                sql = (
                    "SELECT "
                    "DATE_FORMAT(purchase_date, '%Y-%m') as month, "
                    "SUM(total_cost) as total_spent, "
                    "COUNT(*) as order_count, "
                    "AVG(unit_cost) as avg_unit_cost, "
                    "SUM(quantity) as total_quantity "
                    "FROM purchase_history "
                    "WHERE purchase_date >= DATE_SUB(CURDATE(), INTERVAL %s MONTH)"
                )
                params = [months]
                if supplier_id:
                    sql += " AND supplier_id = %s"
                    params.append(supplier_id)
                if product_category:
                    sql += " AND product_category = %s"
                    params.append(product_category)
                sql += " GROUP BY DATE_FORMAT(purchase_date, '%Y-%m') ORDER BY month"
                cursor.execute(sql, params)
                return cursor.fetchall()
        except Exception as e:
            self.logger.error(f"Error getting purchase trends: {e}")
            return []
