import os
import openai
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import json
import requests
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AIEnhancedClient:
    def __init__(self):
        """Initialize AI-enhanced client with OpenAI, MCP, and vector search capabilities."""
        self.logger = logging.getLogger(__name__)
        
        # OpenAI configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            self.logger.warning("OpenAI API key not configured")
        
        # MCP configuration
        self.mcp_server_url = os.getenv('MCP_SERVER_URL')
        self.mcp_api_key = os.getenv('MCP_API_KEY')
        
        # Vector search configuration
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_dimension = 384
        self.index = None
        self.supplier_embeddings = []
        self.supplier_data = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for text using sentence transformers."""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            return embeddings
        except Exception as e:
            self.logger.error(f"Error creating embeddings: {e}")
            return np.array([])
    
    def build_vector_index(self, suppliers_data: List[Dict[str, Any]]):
        """Build FAISS vector index for supplier data."""
        try:
            if not suppliers_data:
                self.logger.warning("No supplier data provided for vector index")
                return False
            
            # Create text representations for suppliers
            supplier_texts = []
            for supplier in suppliers_data:
                text = f"{supplier.get('supplier_name', '')} {supplier.get('category', '')} {supplier.get('location', '')} {supplier.get('payment_terms', '')}"
                supplier_texts.append(text)
            
            # Create embeddings
            embeddings = self.create_embeddings(supplier_texts)
            
            if len(embeddings) == 0:
                self.logger.error("Failed to create embeddings")
                return False
            
            # Initialize FAISS index
            self.index = faiss.IndexFlatIP(self.vector_dimension)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype('float32'))
            
            # Store data for retrieval
            self.supplier_embeddings = embeddings
            self.supplier_data = suppliers_data
            
            self.logger.info(f"Built vector index with {len(suppliers_data)} suppliers")
            return True
                
        except Exception as e:
            self.logger.error(f"Error building vector index: {e}")
            return False
    def vector_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform vector search using FAISS."""
        try:
            if self.index is None or len(self.supplier_data) == 0:
                self.logger.warning("Vector index not built or empty")
                return []
            
            # Create query embedding
            query_embedding = self.create_embeddings([query])
            
            if len(query_embedding) == 0:
                return []
            
            # Normalize query embedding
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding.astype('float32'), min(k, len(self.supplier_data)))
            
            # Return results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.supplier_data) and idx >= 0:  # Check for valid index
                    result = self.supplier_data[idx].copy()
                    result['similarity_score'] = float(score)
                    result['rank'] = i + 1
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in vector search: {e}")
            return []
    
    def full_text_search(self, query: str, suppliers_data: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
        """Perform full-text search on supplier data."""
        try:
            results = []
            query_lower = query.lower()
            
            for supplier in suppliers_data:
                score = 0
                
                # Search in supplier name
                if query_lower in supplier.get('supplier_name', '').lower():
                    score += 3
                
                # Search in category
                if query_lower in supplier.get('category', '').lower():
                    score += 2
                
                # Search in location
                if query_lower in supplier.get('location', '').lower():
                    score += 1
                
                # Search in payment terms
                if query_lower in supplier.get('payment_terms', '').lower():
                    score += 1
                
                if score > 0:
                    result = supplier.copy()
                    result['text_match_score'] = score
                    results.append(result)
            
            # Sort by score and return top k
            results.sort(key=lambda x: x['text_match_score'], reverse=True)
            return results[:k]
            
        except Exception as e:
            self.logger.error(f"Error in full-text search: {e}")
            return []
    
    def hybrid_search(self, query: str, suppliers_data: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector and full-text search."""
        try:
            # Vector search
            vector_results = self.vector_search(query, k)
            
            # Full-text search
            text_results = self.full_text_search(query, suppliers_data, k)
            
            # Combine and rank results
            combined_results = {}
            
            # Add vector search results
            for result in vector_results:
                supplier_id = result.get('supplier_id')
                if supplier_id:
                    combined_results[supplier_id] = {
                        'data': result,
                        'vector_score': result.get('similarity_score', 0),
                        'text_score': 0
                    }
            
            # Add text search results
            for result in text_results:
                supplier_id = result.get('supplier_id')
                if supplier_id:
                    if supplier_id in combined_results:
                        combined_results[supplier_id]['text_score'] = result.get('text_match_score', 0)
                    else:
                        combined_results[supplier_id] = {
                            'data': result,
                            'vector_score': 0,
                            'text_score': result.get('text_match_score', 0)
                        }
            
            # Calculate combined scores and sort
            final_results = []
            for supplier_id, scores in combined_results.items():
                combined_score = (scores['vector_score'] * 0.6) + (scores['text_score'] * 0.4)
                result = scores['data'].copy()
                result['combined_score'] = combined_score
                final_results.append(result)
            
            # Sort by combined score
            final_results.sort(key=lambda x: x['combined_score'], reverse=True)
            return final_results[:k]
            
        except Exception as e:
            self.logger.error(f"Error in hybrid search: {e}")
            return []
    
    def analyze_with_gpt(self, query: str, context_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze procurement query using GPT-3.5 Turbo."""
        try:
            if not self.openai_client:
                self.logger.warning("OpenAI client not configured")
                return {}
            
            # Prepare context
            context_text = "Available suppliers:\n"
            for i, supplier in enumerate(context_data[:5], 1):
                context_text += f"{i}. {supplier.get('supplier_name', 'Unknown')} "
                context_text += f"({supplier.get('category', 'Unknown')}, {supplier.get('location', 'Unknown')})\n"
                context_text += f"   Rating: {supplier.get('rating', 'N/A')}, "
                context_text += f"Lead Time: {supplier.get('lead_time_days', 'N/A')} days, "
                context_text += f"Price: ${supplier.get('price_per_unit', 'N/A')}\n"
            
            # Create prompt
            prompt = f"""
            You are an AI procurement analyst. Analyze the following procurement request and provide insights:

            Procurement Request: {query}

            {context_text}

            Please provide:
            1. Top 3 supplier recommendations with reasoning
            2. Risk assessment for each recommendation
            3. Cost-benefit analysis
            4. Strategic insights and recommendations

            Format your response as JSON with the following structure:
            {{
                "top_recommendations": [
                    {{
                        "supplier_name": "name",
                        "reasoning": "explanation",
                        "risk_level": "low/medium/high",
                        "estimated_cost": "amount",
                        "confidence_score": 0.0-1.0
                    }}
                ],
                "risk_assessment": "overall risk analysis",
                "cost_analysis": "cost-benefit summary",
                "strategic_insights": "strategic recommendations"
            }}
            """
            
            # Call GPT-3.5 Turbo
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are an expert procurement analyst. Provide detailed, actionable insights in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse response
            try:
                analysis = json.loads(response.choices[0].message.content)
                return analysis
            except json.JSONDecodeError:
                # Fallback: return structured text
                return {
                    "analysis": response.choices[0].message.content,
                    "model_used": self.openai_model
                }
                
        except Exception as e:
            self.logger.error(f"Error in GPT analysis: {e}")
            return {"error": str(e)}
    
    def call_mcp_server(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP server for additional analysis."""
        try:
            if not self.mcp_server_url:
                self.logger.warning("MCP server URL not configured")
                return {}
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.mcp_api_key}' if self.mcp_api_key else ''
            }
            
            response = requests.post(
                f"{self.mcp_server_url}/{endpoint}",
                json=data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"MCP server error: {response.status_code} - {response.text}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error calling MCP server: {e}")
            return {}
    
    def enhanced_procurement_analysis(self, query: str, suppliers_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive procurement analysis using all AI capabilities."""
        try:
            # Build vector index if not exists
            if self.index is None:
                self.build_vector_index(suppliers_data)
            
            # Perform hybrid search
            search_results = self.hybrid_search(query, suppliers_data, k=10)
            
            # Analyze with GPT-3.5 Turbo
            gpt_analysis = self.analyze_with_gpt(query, search_results)
            
            # Call MCP server for additional insights
            mcp_data = {
                "query": query,
                "suppliers_count": len(suppliers_data),
                "search_results": search_results[:5]
            }
            mcp_analysis = self.call_mcp_server("analyze", mcp_data)
            
            # Combine results
            comprehensive_analysis = {
                "query": query,
                "search_results": search_results,
                "gpt_analysis": gpt_analysis,
                "mcp_analysis": mcp_analysis,
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
                "ai_models_used": {
                    "vector_search": "sentence-transformers",
                    "llm": self.openai_model,
                    "mcp": bool(self.mcp_server_url)
                }
            }
            
            return comprehensive_analysis
            
        except Exception as e:
            self.logger.error(f"Error in enhanced procurement analysis: {e}")
            return {"error": str(e)}
    
    def generate_supplier_recommendations(self, product_name: str, quantity: int, 
                                        category: str = None, budget: float = None) -> Dict[str, Any]:
        """Generate AI-powered supplier recommendations."""
        try:
            # Create query
            query = f"Need {quantity} units of {product_name}"
            if category:
                query += f" in {category} category"
            if budget:
                query += f" with budget ${budget}"
            
            # Get all supplier data (this would come from TiDB in practice)
            # For now, we'll use a placeholder
            suppliers_data = []  # This should be populated from TiDB
            
            # Perform enhanced analysis
            analysis = self.enhanced_procurement_analysis(query, suppliers_data)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error generating supplier recommendations: {e}")
            return {"error": str(e)}
    def debug_vector_index(self):
        """Debug method to check vector index status."""
        status = {
            'index_exists': self.index is not None,
            'supplier_data_count': len(self.supplier_data),
            'supplier_embeddings_shape': self.supplier_embeddings.shape if hasattr(self.supplier_embeddings, 'shape') else 'No embeddings',
            'openai_configured': self.openai_client is not None,
            'mcp_configured': bool(self.mcp_server_url)
        }
        
        if self.supplier_data and len(self.supplier_data) > 0:
            status['sample_supplier'] = self.supplier_data[0]
        
        return status
