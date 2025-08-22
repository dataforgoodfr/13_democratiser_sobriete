import os
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Qdrant configuration
QDRANT_URL = os.getenv(
    "QDRANT_URL", 
    "116919ed-8e07-47f6-8f24-a22527d5d520.europe-west3-0.gcp.cloud.qdrant.io"
)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "None")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "index_1")

class QdrantClient:
    """Client for connecting to Qdrant vector database"""
    
    def __init__(self, url: str = None, api_key: str = None, collection_name: str = None):
        """
        Initialize Qdrant client
        
        Args:
            url: Qdrant server URL
            api_key: Qdrant API key
            collection_name: Name of the collection to query
        """
        self.url = url or QDRANT_URL
        self.api_key = api_key or QDRANT_API_KEY
        self.collection_name = collection_name or QDRANT_COLLECTION
        
        try:
            from qdrant_client import QdrantClient as QdrantClientLib
            from qdrant_client.http import models as rest
            
            # Initialize the Qdrant client
            if self.api_key and self.api_key != "None":
                self.client = QdrantClientLib(
                    url=self.url,
                    api_key=self.api_key
                )
            else:
                self.client = QdrantClientLib(url=self.url)
                
            logger.info(f"Connected to Qdrant at {self.url}")
            
        except ImportError:
            raise ImportError(
                "Please install qdrant-client: 'pip install qdrant-client'"
            )
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {e}")
            raise
    
    def get_text_by_id(self, point_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve text content by point ID (openalex_id)
        
        Args:
            point_id: The point ID to retrieve (corresponds to openalex_id)
            
        Returns:
            Dictionary containing the point data or None if not found
        """
        try:
            from qdrant_client.http import models as rest
            
            # Retrieve the point by ID
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id]
            )
            
            if points and len(points) > 0:
                point = points[0]
                return {
                    'id': point.id,
                    'payload': point.payload,
                    'vector': point.vector
                }
            else:
                logger.warning(f"No point found with ID: {point_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving point {point_id}: {e}")
            return None
    
    def get_texts_by_ids(self, point_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve multiple text contents by point IDs
        
        Args:
            point_ids: List of point IDs to retrieve
            
        Returns:
            List of dictionaries containing point data
        """
        try:
            from qdrant_client.http import models as rest
            
            # Retrieve multiple points by IDs
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=point_ids
            )
            
            results = []
            for point in points:
                results.append({
                    'id': point.id,
                    'payload': point.payload,
                    'vector': point.vector
                })
            
            logger.info(f"Retrieved {len(results)} points from Qdrant")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving points: {e}")
            return []
    
    def search_by_text(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for texts by query text (requires embeddings)
        
        Args:
            query_text: Text to search for
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing search results
        """
        try:
            from qdrant_client.http import models as rest
            
            # This would require generating embeddings for the query text
            # For now, we'll return an empty list as this requires additional setup
            logger.warning("Text search requires embedding generation - not implemented yet")
            return []
            
        except Exception as e:
            logger.error(f"Error searching by text: {e}")
            return [] 