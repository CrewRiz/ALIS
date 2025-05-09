# memory/genetic_memory.py

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import google.generativeai as genai
from settings import SETTINGS
from base_classes import Rule
import uuid
import time
import json
from google.cloud import aiplatform

class LamarckianRule(Rule):
    def __init__(self, rule_text, parent_rules=None):
        super().__init__(rule_text)
        self.parent_rules = parent_rules or []
        self.acquired_traits = {}
        self.experience_log = []
        self.meta_state = {
            'attention_focus': 1.0,
            'self_awareness': 0.0,
            'subjective_value': 0.0
        }
        self.adaptation_history = []
        
    def adapt_from_experience(self, experience):
        """Modify rule structure based on experience"""
        self.experience_log.append(experience)
        
        # Update acquired traits
        for trait, value in experience.traits.items():
            if trait not in self.acquired_traits:
                self.acquired_traits[trait] = value
            else:
                self.acquired_traits[trait] = (
                    0.7 * self.acquired_traits[trait] + 
                    0.3 * value
                )
                
        # Modify rule structure
        self._modify_structure(experience)
        
        # Update meta state
        self._update_meta_state(experience)


class RAGMemory:
    """Memory system using Retrieval Augmented Generation principles with Gemini embeddings and Vertex AI Vector Search."""
    def __init__(self, 
                 dimension: int, 
                 google_api_key: Optional[str] = None,
                 project_id: Optional[str] = None,
                 location: str = "us-central1",
                 index_endpoint_name: Optional[str] = None,
                 index_name: Optional[str] = None,
                 deployed_index_id: Optional[str] = None):
        """
        Initialize RAGMemory with Vertex AI Vector Search.
        
        Args:
            dimension: Embedding dimension
            google_api_key: Google API key for Gemini
            project_id: Google Cloud project ID
            location: Google Cloud region
            index_endpoint_name: Full resource name of the Vertex AI Vector Search endpoint
            index_name: Full resource name of the Vertex AI Vector Search index
            deployed_index_id: ID of the deployed index
        """
        self.dimension = dimension
        self.logger = logging.getLogger(__name__)
        self.embedding_model_name = 'models/text-embedding-004'  # Gemini embedding model
        
        # Local memory store as fallback and cache
        self.memory_store: List[Dict[str, Any]] = []  # Stores {'embedding': vector, 'data': data_item, 'id': unique_id}
        
        # Configure Gemini API
        if not google_api_key:
            google_api_key = SETTINGS['api_keys'].get('google')

        if not google_api_key:
            self.logger.error("Google API Key not provided or found in settings for RAGMemory. Embeddings will fail.")
            self.genai_configured = False
        else:
            try:
                # Configure genai if not already configured elsewhere globally
                genai.configure(api_key=google_api_key)
                self.genai_configured = True
                self.logger.info(f"RAGMemory initialized with embedding model: {self.embedding_model_name}")
            except Exception as e:
                self.logger.error(f"Failed to configure genai for RAGMemory: {e}")
                self.genai_configured = False
        
        # Configure Vertex AI Vector Search
        self.project_id = project_id or SETTINGS.get('google_cloud', {}).get('project_id')
        self.location = location
        self.index_endpoint_name = index_endpoint_name or SETTINGS.get('google_cloud', {}).get('index_endpoint_name')
        self.index_name = index_name or SETTINGS.get('google_cloud', {}).get('index_name')
        self.deployed_index_id = deployed_index_id or SETTINGS.get('google_cloud', {}).get('deployed_index_id')
        
        self.vertex_ai_configured = False
        if self.project_id and self.index_endpoint_name and self.deployed_index_id:
            try:
                # Initialize Vertex AI
                aiplatform.init(project=self.project_id, location=self.location)
                
                # Get the index endpoint
                self.index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=self.index_endpoint_name)
                self.logger.info(f"Connected to Vertex AI Vector Search endpoint: {self.index_endpoint_name}")
                self.vertex_ai_configured = True
            except Exception as e:
                self.logger.error(f"Failed to initialize Vertex AI Vector Search: {e}")
                self.logger.warning("Falling back to local vector storage only.")
        else:
            self.logger.warning("Vertex AI Vector Search not configured. Using local vector storage only.")

    def _generate_embedding(self, text_content: str) -> Optional[List[float]]:
        """Generates an embedding for the given text content using Gemini."""
        if not self.genai_configured or not text_content:
            self.logger.warning("genai not configured or empty content for embedding.")
            return None
        try:
            result = genai.embed_content(
                model=self.embedding_model_name,
                content=text_content,
                task_type="RETRIEVAL_DOCUMENT"  # or RETRIEVAL_QUERY for queries
            )
            return result['embedding']
        except Exception as e:
            self.logger.error(f"Error generating embedding for content: '{text_content[:100]}...': {e}")
            return None

    def _prepare_text_for_embedding(self, data_item: Dict[str, Any]) -> str:
        """Prepares a single string from a data item for embedding."""
        texts = []
        if 'task' in data_item and isinstance(data_item['task'], str):
            texts.append(f"Task: {data_item['task']}")
        
        if 'patterns' in data_item and isinstance(data_item['patterns'], list):
            pattern_descs = [p.get('description', '') for p in data_item['patterns'] if isinstance(p, dict) and p.get('description')]
            if not pattern_descs:  # Fallback for Pattern objects
                 pattern_descs = [p.description for p in data_item['patterns'] if hasattr(p, 'description') and p.description]
            if pattern_descs:
                texts.append("Detected Patterns: " + "; ".join(pattern_descs))

        if 'rules' in data_item and isinstance(data_item['rules'], list):
            # Assuming rules might be Rule objects or dicts
            rule_texts = [r.get('rule_text', '') for r in data_item['rules'] if isinstance(r, dict) and r.get('rule_text')]
            if not rule_texts:  # Fallback for Rule objects
                rule_texts = [r.rule_text for r in data_item['rules'] if hasattr(r, 'rule_text') and r.rule_text]
            if rule_texts:
                texts.append("Generated Rules: " + "; ".join(rule_texts))
        
        if 'analysis' in data_item and isinstance(data_item.get('analysis'), dict):
            analysis_text = data_item['analysis'].get('analysis')  # From AnalysisAgent output
            if isinstance(analysis_text, str):
                 texts.append(f"Analysis Summary: {analysis_text}")
        elif 'analysis' in data_item and isinstance(data_item.get('analysis'), str):
             texts.append(f"Analysis Summary: {data_item.get('analysis')}")

        if 'results' in data_item and isinstance(data_item['results'], dict):
             web_summary = str(data_item['results'].get('summary', ''))[:200]  # Summarize web results if too long
             if web_summary:
                 texts.append(f"Web Results Summary: {web_summary}")
        elif 'results' in data_item and isinstance(data_item['results'], str):
            texts.append(f"Results: {str(data_item['results'])[:200]}")

        # Add timestamp if available
        if 'timestamp' in data_item:
            texts.append(f"Timestamp: {data_item['timestamp']}")

        return "\n".join(texts) if texts else "Empty_Content"

    def _add_to_vertex_ai(self, item_id: str, embedding: List[float], data_json: str) -> bool:
        """Add an embedding to Vertex AI Vector Search."""
        if not self.vertex_ai_configured:
            return False
        
        try:
            # Upsert the embedding into the Vector Search index
            response = self.index_endpoint.upsert_datapoints(
                deployed_index_id=self.deployed_index_id,
                datapoints=[{
                    "datapoint_id": item_id,
                    "feature_vector": embedding,
                    "restricts": {},  # Optional: Add restricts for filtering
                    "numeric_restricts": {},  # Optional: Add numeric restricts
                    "sparse_vector": {},  # Optional: Add sparse vector
                }]
            )
            
            # Store the full data in a Cloud Storage bucket or Firestore
            # This is necessary because Vector Search only stores the vectors, not the full data
            # For simplicity, we'll skip this step and rely on our local cache
            # In a production system, you'd store data_json in a persistent storage
            
            self.logger.info(f"Added item {item_id} to Vertex AI Vector Search")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add item to Vertex AI Vector Search: {e}")
            return False

    def add_memory(self, data_item: Dict[str, Any]) -> Optional[str]:
        """
        Adds a new item to the memory, including its embedding.
        
        Returns:
            str: The ID of the added memory item, or None if failed
        """
        if not isinstance(data_item, dict):
            self.logger.warning("Attempted to add non-dict item to RAGMemory.")
            return None

        # Generate a unique ID for this memory item
        item_id = str(uuid.uuid4())
        
        # Prepare text for embedding
        text_to_embed = self._prepare_text_for_embedding(data_item)
        if not text_to_embed or text_to_embed == "Empty_Content":
            self.logger.warning(f"No content generated for embedding from data_item: {list(data_item.keys())}")
            return None

        # Generate embedding
        embedding_vector = self._generate_embedding(text_to_embed)
        if not embedding_vector:
            self.logger.warning(f"Failed to generate embedding for data_item. Not adding to memory store.")
            return None
            
        if len(embedding_vector) != self.dimension:
            self.logger.warning(f"Generated embedding dimension {len(embedding_vector)} != configured dimension {self.dimension}.")
        
        # Add to local memory store
        memory_item = {
            'id': item_id,
            'embedding': np.array(embedding_vector), 
            'data': data_item,
            'timestamp': data_item.get('timestamp', time.time())
        }
        self.memory_store.append(memory_item)
        
        # Add to Vertex AI Vector Search if configured
        if self.vertex_ai_configured:
            # Serialize the data for storage
            data_json = json.dumps(data_item, default=str)  # default=str handles datetime objects
            success = self._add_to_vertex_ai(item_id, embedding_vector, data_json)
            if not success:
                self.logger.warning(f"Item {item_id} added to local store only, failed to add to Vertex AI.")
        
        self.logger.info(f"Added new item {item_id} to RAGMemory. Total items in local store: {len(self.memory_store)}")
        return item_id

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Computes cosine similarity between two numpy vectors."""
        if vec1 is None or vec2 is None:
            return 0.0
        # Ensure they are numpy arrays
        vec1 = np.asarray(vec1, dtype=float)
        vec2 = np.asarray(vec2, dtype=float)
        
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return dot_product / (norm_vec1 * norm_vec2)

    def _search_vertex_ai(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Search for similar vectors in Vertex AI Vector Search.
        
        Returns:
            List of tuples (item_id, similarity_score)
        """
        if not self.vertex_ai_configured:
            return []
        
        try:
            # Find nearest neighbors
            response = self.index_endpoint.find_neighbors(
                deployed_index_id=self.deployed_index_id,
                queries=[{
                    "datapoint": {
                        "feature_vector": query_embedding,
                        "restricts": {},  # Optional: Add restricts for filtering
                        "numeric_restricts": {},  # Optional: Add numeric restricts
                        "sparse_vector": {},  # Optional: Add sparse vector
                    },
                    "neighbor_count": top_k
                }]
            )
            
            # Extract results
            results = []
            if response and response.nearest_neighbors:
                for neighbor in response.nearest_neighbors[0].neighbors:
                    item_id = neighbor.datapoint.datapoint_id
                    # Note: Vertex AI returns distance, not similarity
                    # For cosine distance, similarity = 1 - distance
                    similarity = 1.0 - neighbor.distance
                    results.append((item_id, similarity))
            
            return results
        except Exception as e:
            self.logger.error(f"Failed to search Vertex AI Vector Search: {e}")
            return []

    def _local_search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors in local memory store.
        
        Returns:
            List of tuples (item_id, similarity_score, data_item)
        """
        similarities = []
        for item in self.memory_store:
            if item.get('embedding') is not None:
                similarity = self._cosine_similarity(query_embedding, item['embedding'])
                similarities.append((item['id'], similarity, item['data']))
        
        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_relevant_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves the top_k most relevant data items from memory for a given query."""
        if not query:
            return []
        
        # Generate embedding for the query
        query_embedding_list = self._generate_embedding(query)
        if not query_embedding_list:
            self.logger.warning("Failed to generate embedding for query.")
            return []
        
        query_embedding = np.array(query_embedding_list)
        if len(query_embedding) != self.dimension:
            self.logger.warning(f"Query embedding dim {len(query_embedding)} != RAGMemory dim {self.dimension}.")

        # Try Vertex AI Vector Search first
        relevant_items = []
        if self.vertex_ai_configured:
            try:
                # Get IDs and scores from Vertex AI
                vertex_results = self._search_vertex_ai(query_embedding_list, top_k)
                
                if vertex_results:
                    # Look up the full data items from our local cache
                    id_to_data = {item['id']: item['data'] for item in self.memory_store}
                    
                    for item_id, similarity in vertex_results:
                        if item_id in id_to_data:
                            self.logger.info(f"Retrieved item {item_id} from Vertex AI with similarity: {similarity:.4f}")
                            relevant_items.append(id_to_data[item_id])
                        else:
                            # If we don't have the item in our local cache, we'd need to fetch it from persistent storage
                            # This is where you'd add code to fetch from Cloud Storage, Firestore, etc.
                            self.logger.warning(f"Item {item_id} found in Vertex AI but not in local cache.")
                    
                    # If we found all items, return them
                    if len(relevant_items) == top_k:
                        return relevant_items
                    
                    # If we found some but not all, adjust top_k for local search
                    top_k = top_k - len(relevant_items)
            except Exception as e:
                self.logger.error(f"Error searching Vertex AI: {e}")
                # Fall back to local search
        
        # If Vertex AI search failed or didn't find enough items, use local search
        if len(relevant_items) < top_k:
            local_results = self._local_search(query_embedding, top_k)
            for _, similarity, data_item in local_results:
                self.logger.info(f"Retrieved item from local store with similarity: {similarity:.4f}")
                relevant_items.append(data_item)
        
        return relevant_items

    def clear_memory(self) -> bool:
        """Clear all memory items. Returns True if successful."""
        # Clear local memory
        self.memory_store = []
        
        # Clear Vertex AI index if configured
        # Note: This is a destructive operation and should be used with caution
        # In a production system, you might want to create a new index instead
        if self.vertex_ai_configured:
            try:
                # This is a placeholder - the actual API call depends on your Vertex AI setup
                # You might need to delete and recreate the index, or use a different API call
                self.logger.warning("Clearing Vertex AI Vector Search index is not implemented.")
                return False
            except Exception as e:
                self.logger.error(f"Failed to clear Vertex AI Vector Search index: {e}")
                return False
        
        return True
