import os
import json
import re
from typing import Dict, List, Optional, Any
from config import Config
from .chunk_processor import ChunkProcessor
from ..utils.chromadb_manager import ChromaDBManager
from ..utils.similarity_analyzer import SimilarityAnalyzer


class FaultReportProcessor:
    """Main processor class that coordinates fault report processing and analysis"""
    
    def __init__(self, 
                 embedding_model,
                 judgment_agent=None,
                 chroma_base_path: str = "./chroma_db",
                 use_structure_rag: bool = True,
                 use_chain_rerank: bool = True):
        """
        Initialize fault report processor
        
        Args:
            embedding_model: Embedding model instance
            judgment_agent: Judgment agent instance (optional)
            chroma_base_path: ChromaDB storage path
            use_structure_rag: Whether to use structure RAG for chain matching (default True)
            use_chain_rerank: Whether to enable propagation chain reranking (default True)
        """
        self.embedding_model = embedding_model
        self.judgment_agent = judgment_agent
        self.use_chain_rerank = use_chain_rerank
        
        # Initialize components
        self.chunk_processor = ChunkProcessor()
        self.db_manager = ChromaDBManager(chroma_base_path, embedding_model)
        self.similarity_analyzer = SimilarityAnalyzer(embedding_model, self.db_manager, use_structure_rag=use_structure_rag)

    @staticmethod
    def _clean_json_string(content: str) -> str:
        """
        Clean markdown format markers from JSON strings

        Args:
            content: Original string content

        Returns:
            Cleaned JSON string
        """
        # Remove leading and trailing markdown code block markers
        content = content.strip()

        # Match and remove ```json at beginning
        content = re.sub(r'^```json\s*\n?', '', content, flags=re.MULTILINE)

        # Match and remove ``` at end
        content = re.sub(r'\n?```\s*$', '', content, flags=re.MULTILINE)

        return content.strip()

    def process_and_store_fault_report(self, report_json: Dict, case_id: str,
                                       document_id_prefix: Optional[str] = None) -> Dict:
        """
        Process fault report and store to vector database
        
        Args:
            report_json: Fault analysis report dict
            case_id: Case ID
            document_id_prefix: ChromaDB document ID prefix
            
        Returns:
            Dict containing processing and storage results
        """
        # Process report, generate text chunks
        chunks_result = self.chunk_processor.process_fault_report(report_json, case_id)
        
        # Print processing results
        self._print_chunks_result(chunks_result, "Fault report")
        
        # Store to ChromaDB
        storage_result = self.db_manager.store_chunks_to_collections(
            chunks_result, case_id, id_prefix=document_id_prefix
        )
        
        # Print storage results
        print("\n=== Storage Results ===")
        for chunk_type, status in storage_result.items():
            print(f"  {chunk_type}: {status}")
        
        return {
            'chunks': chunks_result,
            'storage': storage_result
        }
    
    def process_and_analyze_inference_report(self, report_json: Dict, case_id: str, 
                                           case_table: Optional[Dict] = None,
                                           max_items_per_section: int = 5) -> Dict:
        """
        Process inference report and perform similar case analysis
        
        Args:
            report_json: Inference report dict
            case_id: Case ID
            case_table: Case info table (for judgment analysis)
            max_items_per_section: Max items per section
            
        Returns:
            Dict containing processing and analysis results
        """
        # Process inference report, generate text chunks
        chunks_result = self.chunk_processor.process_inference_report(report_json, case_id)
        
        # Print processing results
        self._print_chunks_result(chunks_result, "Inference report")
        
        # Search for similar cases
        print("\n=== Searching for similar cases ===")
        search_result = self.similarity_analyzer.search_similar_cases(chunks_result)
        
        # Print search results
        self._print_search_results(search_result['search_results'])
        
        # Match most similar cases
        print("\n=== Matching similarity ===")
        candidate_cases = search_result['candidate_cases']
        top_similar_cases = self.similarity_analyzer.match_similar_cases(
            chunks_result, 
            candidate_cases,
            use_chain_rerank=self.use_chain_rerank
        )
        
        # Print most similar cases
        print(f"\n=== Top 3 recommended similar cases ===")
        for i, similar_case_id in enumerate(top_similar_cases, 1):
            print(f"{i}. {similar_case_id}")
        
        result = {
            'chunks': chunks_result,
            'search_results': search_result['search_results'],
            'candidate_cases': candidate_cases,
            'top_similar_cases': top_similar_cases
        }
        
        # Run judgment analysis if judgment agent is available
        if self.judgment_agent and case_table:
            try:
                judgment_result = self._run_judgment_analysis(
                    case_id,
                    chunks_result,
                    top_similar_cases,
                    candidate_cases,
                    case_table,
                    max_items_per_section
                )

                # Validate if returned content is empty
                if not judgment_result or not judgment_result.strip():
                    print("\nWarning: JudgmentAgent returned empty content")
                    result['judgment_analysis'] = None
                else:
                    # Clean markdown format markers
                    cleaned_result = self._clean_json_string(judgment_result)

                    # Try to parse JSON
                    try:
                        result['judgment_analysis'] = json.loads(cleaned_result)
                        print("\n=== JudgmentAgent diagnosis result ===")
                        print(json.dumps(result['judgment_analysis'], ensure_ascii=False, indent=2))
                    except json.JSONDecodeError as json_error:
                        print(f"\nWarning: JudgmentAgent returned content is not valid JSON format")
                        print(f"JSON parse error: {json_error}")
                        print(f"Cleaned content: {cleaned_result[:500]}...")  # Print first 500 chars
                        result['judgment_analysis'] = {
                            'error': 'JSON parse failed',
                            'raw_content': judgment_result
                        }

            except Exception as e:
                print(f"JudgmentAgent call failed: {e}")
                import traceback
                traceback.print_exc()
                result['judgment_analysis'] = None
        
        return result
    
    def search_by_text(self, query_text: str, collection: str = "fault_events", 
                      n_results: int = 3) -> Dict:
        """
        Search for related cases by text
        
        Args:
            query_text: Query text
            collection: Collection name
            n_results: Number of results to return
            
        Returns:
            Search results
        """
        return self.db_manager.search_chunks(query_text, collection, n_results)
    
    def get_case_details(self, case_id: str) -> Dict:
        """
        Get all detailed information for specified case
        
        Args:
            case_id: Case ID
            
        Returns:
            All text chunk information for the case
        """
        return self.db_manager.get_chunks_by_case_id(case_id)
    
    # Private helper methods
    def _print_chunks_result(self, chunks_result: Dict, report_type: str):
        """Print text chunk processing results"""
        print(f"\n=== {report_type} Processing Results ===")
        
        if 'events' in chunks_result:
            print(f"\nEvent chunks: {len(chunks_result['events'])}")
            for i, event in enumerate(chunks_result['events']):
                print(f"  Event {i+1}:")
                print(f"    Content: {event['page_content'][:100]}...")
                print(f"    Root cause: {event['metadata'].get('root_cause_type', 'N/A')}")
        
        print(f"\nSymptom chunks: {len(chunks_result.get('symptoms', []))}")
        for i, symptom in enumerate(chunks_result.get('symptoms', [])):
            print(f"  Symptom {i+1}:")
            print(f"    Layer: {symptom['metadata'].get('layer', 'N/A')}")
            print(f"    Metrics count: {symptom['metadata'].get('metrics_count', 0)}")
        
        print(f"\nChain chunks: {len(chunks_result.get('chains', []))}")
        for i, chain in enumerate(chunks_result.get('chains', [])):
            print(f"  Chain {i+1}:")
            print(f"    Signature: {chain['metadata'].get('signature', 'N/A')}")
            print(f"    Confidence: {chain['metadata'].get('confidence', 0)}")
    
    def _print_search_results(self, search_results: Dict):
        """Print search results"""
        print("\n=== Symptom chunk search results ===")
        for i, result in enumerate(search_results.get('symptoms', [])):
            print(f"Symptom {i+1} found {len(result['search_result'].get('documents', [[]])[0])} similar results")
        
        print("\n=== Chain chunk search results ===")
        for i, result in enumerate(search_results.get('chains', [])):
            print(f"Chain {i+1} found {len(result['search_result'].get('documents', [[]])[0])} similar results")
    
    def _run_judgment_analysis(self, case_id: str, current_chunks: Dict, 
                              top_case_ids: List[str], candidate_cases: Dict,
                              case_table: Dict, max_items: int = 5) -> str:
        """Run judgment analysis"""
        # Build current case snapshot
        current_snapshot = self._build_current_snapshot(case_id, current_chunks, max_items)
        
        # Build historical reference cases
        historical_references = []
        for rank, ref_case_id in enumerate(top_case_ids, start=1):
            if ref_case_id in candidate_cases:
                ref = self._build_historical_reference(
                    ref_case_id, 
                    rank, 
                    candidate_cases[ref_case_id], 
                    case_table,
                    max_items
                )
                historical_references.append(ref)
        
        # Read prompt template
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "prompt", 
            "judgment_analysis_user.txt"
        )
        
        try:
            with open(prompt_path, "r") as file:
                user_prompt_template = file.read()
        except FileNotFoundError:
            # Use default prompt
            user_prompt_template = """
Analyze the current fault event:
{current_event_json}

Reference historical similar cases:
{retrieved_historical_cases_json}

Please analyze the possible causes of the current fault based on historical cases.
"""
        
        # Format prompt
        current_snapshot_json = json.dumps(current_snapshot, ensure_ascii=False, indent=2)
        historical_references_json = json.dumps(historical_references, ensure_ascii=False, indent=2)
        
        with open(f"{Config.BASE_PATH}/llm/prompts/judgment_analysis_system.txt", "r") as file:
            system_prompt = file.read()
        formatted_prompt = user_prompt_template.replace("{current_event_json}", current_snapshot_json)
        formatted_prompt = formatted_prompt.replace("{retrieved_historical_cases_json}", historical_references_json)
        
        # Call judgment agent
        return self.judgment_agent.analyze_with_prompt(system_prompt, formatted_prompt)
    
    def _build_current_snapshot(self, case_id: str, chunks: Dict, max_items: int) -> Dict:
        """Build current case snapshot"""
        snapshot = {
            "case_id": case_id,
            "symptom_groups": [],
            "propagation_chains": []
        }
        
        for symptom in chunks.get('symptoms', [])[:max_items]:
            metadata = symptom.get('metadata', {})
            snapshot["symptom_groups"].append({
                "layer": metadata.get('layer'),
                "layer_display_name": metadata.get('layer_display_name'),
                "time_window": metadata.get('time_window'),
                "metrics_count": metadata.get('metrics_count'),
                "group_name": metadata.get('group_name'),
                "details": symptom.get('page_content', '').strip()
            })
        
        for chain in chunks.get('chains', [])[:max_items]:
            metadata = chain.get('metadata', {})
            snapshot["propagation_chains"].append({
                "chain_id": metadata.get('chain_id'),
                "signature": metadata.get('signature'),
                "confidence": metadata.get('confidence'),
                "nodes_count": metadata.get('nodes_count'),
                "details": chain.get('page_content', '').strip()
            })
        
        return snapshot
    
    def _build_historical_reference(self, case_id: str, rank: int, case_data: Dict,
                                   case_table: Dict, max_items: int) -> Dict:
        """Build historical reference case"""
        case_info = case_table.get(case_id, {})
        
        events_summary = []
        for event in case_data.get('events', [])[:max_items]:
            events_summary.append({
                "summary": event.get('content', '').strip(),
                "root_cause_type": event.get('root_cause_type'),
                "suspected_component": event.get('suspected_component'),
                "start_time": event.get('start_time'),
                "severity": event.get('severity')
            })
        
        symptom_summary = []
        for symptom in case_data.get('symptoms', [])[:max_items]:
            symptom_summary.append({
                "layer": symptom.get('layer'),
                "details": symptom.get('content', '').strip()
            })
        
        chain_summary = []
        for chain in case_data.get('chains', [])[:max_items]:
            chain_summary.append({
                "details": chain.get('content', '').strip()
            })
        
        return {
            "rank": rank,
            "case_id": case_id,
            "given_root_cause": case_info.get('given_root_cause', 'Unknown'),
            "hypothesis": case_info.get('hypothesis', 'Unknown'),
            "suspected_component": case_info.get('suspected_component', 'Unknown'),
            "events": events_summary,
            "symptom_groups": symptom_summary,
            "propagation_chains": chain_summary
        }
