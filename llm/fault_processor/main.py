import os
import sys
import json
import re
from config import Config

# Add required paths
# sys.path.append(f'{Config.BASE_PATH}/data_handle')
# sys.path.append(f'{Config.BASE_PATH}/llm')

from data_handle.data_config import case_table
from llm.agent.EmbeddingAgent import StringInputEmbeddings
from llm.agent.JudgmentAgent import JudgmentAgent

from llm.fault_processor import FaultReportProcessor, Config


def clean_json_string(content):
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


def initialize_components(use_structure_rag=True, use_chain_rerank=True):
    """Initialize all necessary components
    
    Args:
        use_structure_rag: Whether to use structure RAG for chain matching (default True)
        use_chain_rerank: Whether to enable propagation chain reranking (default True)
    """
    # Initialize embedding model
    embedding_model = StringInputEmbeddings(
        model=Config.EMBEDDING_MODEL_NAME,
        base_url=Config.OPENAI_API_BASE,
        api_key=Config.OPENAI_API_KEY,
        encoding_format=Config.EMBEDDING_ENCODING_FORMAT,
        dimensions=Config.EMBEDDING_DIMENSIONS,
    )
    
    # Initialize JudgmentAgent
    judgment_agent = JudgmentAgent()
    
    # Initialize fault report processor
    processor = FaultReportProcessor(
        embedding_model=embedding_model,
        judgment_agent=judgment_agent,
        # chroma_base_path=Config.TEST_CHROMA_DB_PATH
        chroma_base_path=Config.CHROMA_DB_PATH,
        use_structure_rag=use_structure_rag,
        use_chain_rerank=use_chain_rerank
    )
    
    return processor


def deal_fault_report(case_id, item_index=0):
    """
    Process fault report and store to vector database
    
    Args:
        case_id: Case ID
        item_index: Application index
    """
    print(f"\n{'='*50}")
    print(f"Processing fault report - Case ID: {case_id}")
    print(f"{'='*50}\n")
    
    # Initialize processor
    processor = initialize_components()
    
    # Get case info
    case_info = case_table[case_id]
    app = case_info['app_name'][item_index]
    app_group = case_info['app_groups'][item_index][0]
    
    # Read analysis result file
    json_path = Config.get_case_analysis_path(case_id, app, app_group)
    
    try:
        with open(json_path, "r", encoding='utf-8') as file:
            content = file.read()
            # Clean markdown format markers
            clean_content = clean_json_string(content)
            report_json = json.loads(clean_content)
    except FileNotFoundError:
        print(f"Error: file not found {json_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: file {json_path} is not valid JSON format: {e}")
        return
    
    # item_index=0 keeps original document IDs; multi-app cases use independent prefix to avoid conflicts
    document_id_prefix = case_id if item_index == 0 else f"{case_id}_{item_index}"

    # Process and store fault report
    result = processor.process_and_store_fault_report(
        report_json, case_id, document_id_prefix=document_id_prefix
    )
    
    print(f"\n{'='*50}")
    print("Fault report processing complete!")
    print(f"{'='*50}")
    
    return result


def deal_inference_report(case_id, item_index=0, use_structure_rag=True, use_chain_rerank=True):
    """
    Process inference report and perform similar case analysis
    
    Args:
        case_id: Case ID
        item_index: Application index
        use_structure_rag: Whether to use structure RAG for chain matching (default True)
        use_chain_rerank: Whether to enable propagation chain reranking (default True)
    """
    print(f"\n{'='*50}")
    print(f"Processing inference report - Case ID: {case_id}")
    print(f"{'='*50}\n")
    
    # Initialize processor
    processor = initialize_components(use_structure_rag=use_structure_rag, use_chain_rerank=use_chain_rerank)
    
    # Get case info
    case_info = case_table[case_id]
    app = case_info['app_name'][item_index]
    app_group = case_info['app_groups'][item_index][0]
    
    # Read inference result file
    json_path = Config.get_inference_result_path(case_id, app, app_group)
    
    try:
        with open(json_path, "r", encoding='utf-8') as file:
            content = file.read()
            # Clean markdown format markers
            clean_content = clean_json_string(content)
            report_json = json.loads(clean_content)
    except FileNotFoundError:
        print(f"Error: file not found {json_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: file {json_path} is not valid JSON format: {e}")
        return
    
    # Process and analyze inference report
    result = processor.process_and_analyze_inference_report(
        report_json, 
        case_id,
        case_table=case_table,
        max_items_per_section=Config.MAX_ITEMS_PER_SECTION
    )
    
    # Write result to file
    save_path = Config.get_judgment_result_path(case_id, app, app_group)
    ans = dict()
    ans['top_similar_cases'] = result.get('top_similar_cases', [])
    
    # Process judgment_analysis: parse if it's already a string
    judgment_analysis = result.get('judgment_analysis', {})
    ans['judgment_analysis'] = judgment_analysis
    
    # ans_str = json.dumps(ans, ensure_ascii=False, indent=4)
    with open(save_path, "w") as file:
        # file.write(ans_str)
        json.dump(ans, file, ensure_ascii=False, indent=4)
    
    print(f"\n{'='*50}")
    print("Inference report analysis complete!")
    print(f"{'='*50}")
    
    return result


def search_by_text(query_text: str, collection: str = "fault_events"):
    """
    Search for related cases by text
    
    Args:
        query_text: Query text
        collection: Collection name
    """
    print(f"\n{'='*50}")
    print(f"Search query: {query_text}")
    print(f"Search collection: {collection}")
    print(f"{'='*50}\n")
    
    processor = initialize_components()
    results = processor.search_by_text(query_text, collection)
    
    if 'error' in results:
        print(f"Search failed: {results['error']}")
    else:
        print(f"Found {len(results.get('documents', [[]])[0])} related results")
        
        # Print search results
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        
        for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
            print(f"\nResult {i+1}:")
            print(f"Case ID: {metadata.get('case_id', 'N/A')}")
            print(f"Content preview: {doc[:200]}...")
    
    return results


def get_case_details(case_id: str):
    """
    Get detailed information for specified case
    
    Args:
        case_id: Case ID
    """
    print(f"\n{'='*50}")
    print(f"Getting case details - Case ID: {case_id}")
    print(f"{'='*50}\n")
    
    processor = initialize_components()
    details = processor.get_case_details(case_id)
    
    # Print case details
    print(f"Event chunks: {len(details.get('events', []))}")
    print(f"Symptom chunks: {len(details.get('symptoms', []))}")
    print(f"Chain chunks: {len(details.get('chains', []))}")
    
    # Print event summary
    if details.get('events'):
        print("\nEvent summary:")
        for event in details['events'][:1]:  # Only show first event
            print(f"  Root cause type: {event.get('root_cause_type', 'N/A')}")
            print(f"  Suspected component: {event.get('suspected_component', 'N/A')}")
            print(f"  Start time: {event.get('start_time', 'N/A')}")
    
    return details


if __name__ == "__main__":
    # Test deal_fault_report
    # cases = ['case1','case2','case14','case17','case21','case26','case27','case32','case33','case41','case48']
    # cases = ['risk1', 'risk2', 'risk3','risk4']

    deal_inference_report('risk21',0)
    # for case in cases:
    #     deal_fault_report(case,0)
    # deal_fault_report('case39',6)
    
    # Test deal_inference_report  
    #deal_inference_report('case1')
    
    # Test search function
    # search_by_text("database connection failure", "fault_events")
    
    # Test get case details
    # get_case_details('case2')
