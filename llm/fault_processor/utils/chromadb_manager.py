import os
import sys
import threading
from typing import Dict, List, Optional, Any
import chromadb
from chromadb.config import Settings


_CHROMA_WRITE_LOCK = threading.RLock()


class ChromaDBManager:
    """Manage all ChromaDB vector database operations"""
    
    def __init__(self, 
                 chroma_base_path: str = "./chroma_db",
                 embedding_model=None):
        """
        Initialize ChromaDB manager
        
        Args:
            chroma_base_path: ChromaDB storage path
            embedding_model: Embedding model instance
        """
        self.chroma_base_path = chroma_base_path
        self.embedding_model = embedding_model
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client"""
        try:
            with _CHROMA_WRITE_LOCK:
                self.client = chromadb.PersistentClient(
                    path=self.chroma_base_path,
                    settings=Settings(allow_reset=True)
                )
        except Exception as e:
            raise Exception(f"Failed to initialize Chroma client: {e}")
    
    def store_chunks_to_collections(self, chunks_result: Dict, case_id: str,
                                    id_prefix: Optional[str] = None) -> Dict:
        """
        Store processed text chunks into three different ChromaDB collections
        
        Args:
            chunks_result: Dict containing events, symptoms, chains
            case_id: Case ID
            id_prefix: ChromaDB document ID prefix; defaults to case_id
            
        Returns:
            Storage result dict
        """
        if not self.embedding_model:
            return {"error": "Embedding model not set"}
        
        storage_results = {
            "events": {"status": "pending", "count": 0, "error": None},
            "symptoms": {"status": "pending", "count": 0, "error": None}, 
            "chains": {"status": "pending", "count": 0, "error": None}
        }
        
        document_id_prefix = id_prefix or case_id

        # Store event chunks
        storage_results["events"] = self._store_events(
            chunks_result.get('events', []), case_id, document_id_prefix
        )
        
        # Store symptom chunks
        storage_results["symptoms"] = self._store_symptoms(
            chunks_result.get('symptoms', []), case_id, document_id_prefix
        )
        
        # Store chain chunks
        storage_results["chains"] = self._store_chains(
            chunks_result.get('chains', []), case_id, document_id_prefix
        )
        
        return storage_results
    
    def search_chunks(self, query_text: str, collection_name: str, 
                     n_results: int = 3, filter_dict: Optional[Dict] = None) -> Dict:
        """
        Search for similar text chunks in specified collection
        
        Args:
            query_text: Query text
            collection_name: Collection name
            n_results: Number of results to return
            filter_dict: Metadata filter conditions
            
        Returns:
            Search result dict
        """
        if not self.embedding_model:
            return {"error": "Embedding model not set"}
        
        try:
            collection = self.client.get_collection(name=collection_name)
            
            # Generate query embeddings
            query_embedding = self.embedding_model.embed_documents([query_text])[0]
            
            # Execute query
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_dict if filter_dict else None
            )
            
            return results
            
        except Exception as e:
            return {"error": f"ChromaDB query failed: {e}"}
    
    def get_chunks_by_case_id(self, case_id: str) -> Dict:
        """
        Get all text chunks for a case by case_id
        
        Args:
            case_id: Case ID
            
        Returns:
            Dict containing events, symptoms and chains
        """
        result = {
            "events": [],
            "symptoms": [],
            "chains": []
        }
        
        # Get event chunks
        result["events"] = self._get_chunks_from_collection(
            "fault_events", case_id, ["root_cause_type", "severity", "start_time", "suspected_component"]
        )
        
        # Get symptom chunks
        result["symptoms"] = self._get_chunks_from_collection(
            "fault_symptoms", case_id, ["layer", "layer_display_name"]
        )
        
        # Get chain chunks
        result["chains"] = self._get_chunks_from_collection(
            "fault_chains", case_id, []
        )
        
        return result
    
    # Private helper methods
    def _store_events(self, events_chunks: List[Dict], case_id: str,
                      id_prefix: Optional[str] = None) -> Dict:
        """Store event chunks"""
        try:
            if not events_chunks:
                return {"status": "success", "count": 0, "error": "No event chunk data"}

            document_id_prefix = id_prefix or case_id
            
            # Prepare data
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(events_chunks):
                documents.append(chunk['page_content'])
                # Simplify metadata, ensure serializable
                metadata = {
                    'case_id': chunk['metadata']['case_id'],
                    'root_cause_type': chunk['metadata']['root_cause_type'],
                    'severity': chunk['metadata']['severity'],
                    'start_time': chunk['metadata']['start_time'],
                    'suspected_component': chunk['metadata']['suspected_component']
                }
                metadatas.append(metadata)
                ids.append(f"{document_id_prefix}_event_{i}")
            
            # Generate embeddings
            embeddings = self.embedding_model.embed_documents(documents)
            
            # Embedding can run in parallel; ChromaDB collection creation and writes serialized
            with _CHROMA_WRITE_LOCK:
                events_collection = self.client.get_or_create_collection(
                    name="fault_events",
                    metadata={"description": "Fault event chunk storage"}
                )
                events_collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                    ids=ids
                )
            
            return {"status": "success", "count": len(events_chunks), "error": None}
            
        except Exception as e:
            return {"status": "failed", "count": 0, "error": str(e)}
    
    def _store_symptoms(self, symptoms_chunks: List[Dict], case_id: str,
                        id_prefix: Optional[str] = None) -> Dict:
        """Store symptom chunks"""
        try:
            if not symptoms_chunks:
                return {"status": "success", "count": 0, "error": "No symptom chunk data"}

            document_id_prefix = id_prefix or case_id
            
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(symptoms_chunks):
                documents.append(chunk['page_content'])
                metadata = {
                    'case_id': chunk['metadata']['case_id'],
                    'layer': chunk['metadata']['layer'],
                    'layer_display_name': chunk['metadata']['layer_display_name'],
                    'metrics_count': chunk['metadata']['metrics_count'],
                    'time_window': chunk['metadata']['time_window']
                }
                metadatas.append(metadata)
                ids.append(f"{document_id_prefix}_symptom_{i}")
            
            embeddings = self.embedding_model.embed_documents(documents)
            
            with _CHROMA_WRITE_LOCK:
                symptoms_collection = self.client.get_or_create_collection(
                    name="fault_symptoms",
                    metadata={"description": "Fault symptom chunk storage"}
                )
                symptoms_collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                    ids=ids
                )
            
            return {"status": "success", "count": len(symptoms_chunks), "error": None}
            
        except Exception as e:
            return {"status": "failed", "count": 0, "error": str(e)}
    
    def _store_chains(self, chains_chunks: List[Dict], case_id: str,
                      id_prefix: Optional[str] = None) -> Dict:
        """Store propagation chain chunks"""
        try:
            if not chains_chunks:
                return {"status": "success", "count": 0, "error": "No propagation chain data"}

            document_id_prefix = id_prefix or case_id
            
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chains_chunks):
                documents.append(chunk['page_content'])
                metadata = {
                    'case_id': chunk['metadata']['case_id'],
                    'chain_id': str(chunk['metadata']['chain_id']),
                    'chain_type': chunk['metadata']['chain_type'],
                    'signature': chunk['metadata']['signature'],
                    'nodes_count': chunk['metadata']['nodes_count'],
                    'confidence': chunk['metadata']['confidence']
                }
                metadatas.append(metadata)
                ids.append(f"{document_id_prefix}_chain_{i}")
            
            embeddings = self.embedding_model.embed_documents(documents)
            
            with _CHROMA_WRITE_LOCK:
                chains_collection = self.client.get_or_create_collection(
                    name="fault_chains",
                    metadata={"description": "Fault propagation chain chunk storage"}
                )
                chains_collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                    ids=ids
                )
            
            return {"status": "success", "count": len(chains_chunks), "error": None}
            
        except Exception as e:
            return {"status": "failed", "count": 0, "error": str(e)}
    
    def _get_chunks_from_collection(self, collection_name: str, case_id: str, 
                                   metadata_fields: List[str]) -> List[Dict]:
        """Get text chunks from specified collection"""
        try:
            collection = self.client.get_collection(name=collection_name)
            result = collection.get(where={"case_id": case_id})
            
            chunks = []
            if result['documents']:
                for doc, metadata in zip(result['documents'], result['metadatas']):
                    chunk_data = {"content": doc}
                    for field in metadata_fields:
                        if field in metadata:
                            chunk_data[field] = metadata[field]
                    chunks.append(chunk_data)
            
            return chunks
            
        except Exception as e:
            print(f"Failed to get {collection_name} for case {case_id}: {e}")
            return []
