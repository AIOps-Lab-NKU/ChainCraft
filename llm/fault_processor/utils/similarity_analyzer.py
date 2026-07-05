import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from .chromadb_manager import ChromaDBManager
from .chain_graph import parse_candidate_chain, parse_current_chain
from .graph_similarity import DEFAULT_WEIGHTS, hybrid_similarity


class SimilarityAnalyzer:
    """Handle similarity computation and case matching"""

    def __init__(
        self,
        embedding_model,
        chromadb_manager: ChromaDBManager,
        use_structure_rag: bool = True,
        graph_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize similarity analyzer

        Args:
            embedding_model: Embedding model instance
            chromadb_manager: ChromaDB manager instance
            use_structure_rag: Whether to enable structure RAG based chain matching (default True).
                Falls back to old plain text cosine when disabled.
            graph_weights: Graph similarity weight dict, keys: text/node/edge/path/topo; defaults to DEFAULT_WEIGHTS.
        """
        self.embedding_model = embedding_model
        self.db_manager = chromadb_manager
        self.use_structure_rag = use_structure_rag
        self.graph_weights = {**DEFAULT_WEIGHTS, **(graph_weights or {})}
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (between 0 and 1)
        """
        try:
            # Generate embeddings
            embeddings = self.embedding_model.embed_documents([text1, text2])
            emb1, emb2 = np.array(embeddings[0]), np.array(embeddings[1])
            
            # Calculate cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            print(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def search_similar_cases(self, chunks_result: Dict) -> Dict:
        """
        Search for similar cases in processed text chunks
        
        Args:
            chunks_result: Result returned by process_fault_report or process_inference_report
            
        Returns:
            Dict containing search results and candidate cases
        """
        search_results = {
            'symptoms': [],
            'chains': []
        }
        
        # Collect all found case_ids
        collected_case_ids = set()
        
        # Search symptom chunks
        for symptom in chunks_result.get('symptoms', []):
            content = symptom['page_content']
            res = self.db_manager.search_chunks(
                content, 
                collection_name="fault_symptoms", 
                n_results=3
            )
            
            search_results['symptoms'].append({
                'original': symptom,
                'search_result': res
            })
            
            # Extract case_ids
            self._extract_case_ids(res, collected_case_ids)
        
        # Search chain chunks
        for chain in chunks_result.get('chains', []):
            content = chain['page_content']
            res = self.db_manager.search_chunks(
                content, 
                collection_name="fault_chains", 
                n_results=3
            )
            
            search_results['chains'].append({
                'original': chain,
                'search_result': res
            })
            
            # Extract case_ids
            self._extract_case_ids(res, collected_case_ids)
        
        # Get complete candidate cases
        candidate_cases = self._get_candidate_cases(collected_case_ids)
        
        return {
            'search_results': search_results,
            'candidate_cases': candidate_cases
        }
    
    def match_similar_cases(self, current_chunks: Dict, candidate_cases: Dict, 
                           symptom_weight: float = 0.6, chain_weight: float = 0.4,
                           use_chain_rerank: bool = True) -> List[str]:
        """
        Match current inference result with candidate cases by similarity
        
        Args:
            current_chunks: Current case text chunks
            candidate_cases: Candidate case dict
            symptom_weight: Symptom chunk weight
            chain_weight: Propagation chain weight
            use_chain_rerank: Whether to enable propagation chain reranking (default True).
                When False, only uses symptom chunk similarity for ranking.
            
        Returns:
            Sorted list of top 3 most similar case IDs
        """
        case_similarities = {}
        
        for case_id, case_data in candidate_cases.items():
            # Calculate symptom similarity
            symptom_similarity = self._calculate_symptom_similarity(
                current_chunks.get('symptoms', []),
                case_data.get('symptoms', [])
            )
            
            if use_chain_rerank:
                # Calculate chain similarity
                chain_similarity = self._calculate_chain_similarity(
                    current_chunks.get('chains', []),
                    case_data.get('chains', [])
                )
                
                # Weighted sum
                total_similarity = symptom_similarity * symptom_weight + chain_similarity * chain_weight
            else:
                # Only use symptom similarity
                total_similarity = symptom_similarity
            
            case_similarities[case_id] = {
                'total_similarity': total_similarity,
                'symptom_similarity': symptom_similarity,
                'chain_similarity': chain_similarity if use_chain_rerank else 0.0,
                'symptom_matches': len(current_chunks.get('symptoms', []))
            }
        
        # Sort and take top 3
        sorted_cases = sorted(
            case_similarities.items(), 
            key=lambda x: x[1]['total_similarity'], 
            reverse=True
        )
        
        return [case_id for case_id, _ in sorted_cases[:3]]
    
    def get_similarity_details(self, current_chunks: Dict, case_id: str) -> Dict:
        """
        Get detailed similarity information for a specific case
        
        Args:
            current_chunks: Current case text chunks
            case_id: Target case ID
            
        Returns:
            Detailed similarity information
        """
        case_data = self.db_manager.get_chunks_by_case_id(case_id)
        
        symptom_similarity = self._calculate_symptom_similarity(
            current_chunks.get('symptoms', []),
            case_data.get('symptoms', [])
        )
        
        chain_similarity = self._calculate_chain_similarity(
            current_chunks.get('chains', []),
            case_data.get('chains', [])
        )
        
        return {
            'case_id': case_id,
            'symptom_similarity': symptom_similarity,
            'chain_similarity': chain_similarity,
            'overall_similarity': symptom_similarity * 0.6 + chain_similarity * 0.4
        }
    
    # Private helper methods
    def _extract_case_ids(self, search_result: Dict, collected_ids: set):
        """Extract case_ids from search results"""
        if 'metadatas' in search_result and search_result['metadatas']:
            for metadata_list in search_result['metadatas']:
                for metadata in metadata_list:
                    if 'case_id' in metadata:
                        collected_ids.add(metadata['case_id'])
    
    def _get_candidate_cases(self, case_ids: set) -> Dict:
        """Get complete candidate cases by case_id set"""
        candidate_cases = {}
        
        for case_id in case_ids:
            case_chunks = self.db_manager.get_chunks_by_case_id(case_id)
            if 'error' not in case_chunks:
                candidate_cases[case_id] = case_chunks
            else:
                print(f"Failed to get case {case_id}: {case_chunks.get('error', 'Unknown error')}")
        
        return candidate_cases
    
    def _calculate_symptom_similarity(self, current_symptoms: List[Dict], 
                                     candidate_symptoms: List[Dict]) -> float:
        """Calculate symptom chunk similarity"""
        if not current_symptoms or not candidate_symptoms:
            return 0.0
        
        total_similarity = 0.0
        matches = 0
        
        for current_symptom in current_symptoms:
            current_layer = current_symptom['metadata']['layer']
            current_content = current_symptom['page_content']
            
            # Find candidate symptom chunks with same layer
            layer_similarities = []
            for candidate_symptom in candidate_symptoms:
                if candidate_symptom.get('layer') == current_layer:
                    similarity = self.calculate_text_similarity(
                        current_content, 
                        candidate_symptom.get('content', '')
                    )
                    layer_similarities.append(similarity)
            
            # Take max similarity for this layer
            if layer_similarities:
                max_layer_similarity = max(layer_similarities)
                total_similarity += max_layer_similarity
                matches += 1
        
        return total_similarity / matches if matches > 0 else 0.0
    
    def _calculate_chain_similarity(self, current_chains: List[Dict],
                                   candidate_chains: List[Dict]) -> float:
        """
        Calculate propagation chain similarity.

        Default uses "vector recall + graph similarity refinement" hybrid scoring:
        - Text cosine kept as α term (fallback to avoid zero score when graph parsing fails)
        - Node/edge/path/topology four terms computed on structured graph
        - Weighted aggregate, take max current->candidate, then average over all current chains

        Falls back to old plain text cosine when self.use_structure_rag is disabled.
        """
        if not current_chains or not candidate_chains:
            return 0.0

        if not self.use_structure_rag:
            return self._calculate_chain_similarity_text_only(current_chains, candidate_chains)

        # Pre-parse candidate chains (avoid repeated reverse-parsing when many candidates)
        candidate_graphs: List[Tuple[Dict, Optional[Any]]] = []
        for cc in candidate_chains:
            candidate_graphs.append((cc, parse_candidate_chain(cc)))

        all_similarities: List[float] = []

        for current_chain in current_chains:
            current_content = current_chain.get('page_content', '')
            current_graph = parse_current_chain(current_chain)

            chain_sims: List[float] = []
            for candidate_chain, candidate_graph in candidate_graphs:
                candidate_content = candidate_chain.get('content', '')
                text_sim = self.calculate_text_similarity(current_content, candidate_content)

                # If either side graph parsing fails -> degrade to plain text cosine
                if current_graph is None or candidate_graph is None:
                    chain_sims.append(text_sim)
                    continue

                breakdown = hybrid_similarity(
                    current_graph,
                    candidate_graph,
                    text_sim=text_sim,
                    weights=self.graph_weights,
                )
                chain_sims.append(breakdown.score)

            if chain_sims:
                all_similarities.append(max(chain_sims))

        return sum(all_similarities) / len(all_similarities) if all_similarities else 0.0

    def _calculate_chain_similarity_text_only(self, current_chains: List[Dict],
                                              candidate_chains: List[Dict]) -> float:
        """Legacy plain text cosine implementation, used as fallback when graph similarity is disabled"""
        all_similarities: List[float] = []

        for current_chain in current_chains:
            current_content = current_chain.get('page_content', '')

            chain_sims = []
            for candidate_chain in candidate_chains:
                similarity = self.calculate_text_similarity(
                    current_content,
                    candidate_chain.get('content', '')
                )
                chain_sims.append(similarity)

            if chain_sims:
                all_similarities.append(max(chain_sims))

        return sum(all_similarities) / len(all_similarities) if all_similarities else 0.0
