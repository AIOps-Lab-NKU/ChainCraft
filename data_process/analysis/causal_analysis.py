import pandas as pd
import numpy as np
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import sys
import matplotlib.pyplot as plt
import networkx as nx
import json
import os
from config import Config

# Import configuration
# sys.path.append(f'{Config.BASE_PATH}/data_handle')
from data_handle.data_config import case_table

def build_causal_graph_pcmci(data_df, selected_metrics, max_lag=3, alpha=0.05):
    """
    Build causal graph for selected metrics using the PCMCI algorithm
    
    Args:
        data_df: DataFrame containing all data
        selected_metrics: list of metric names to analyze (list of strings)
        max_lag: maximum lag order (recommended not too large, 3-5 is sufficient)
        alpha: significance level
        
    Returns:
        dict: contains 'graph' (NetworkX object), 'edges' (edge list), 'matrix' (p-value matrix)
    """
    # 1. Data preparation
    # Extract selected columns and remove null values
    temp_df = data_df[selected_metrics].dropna()
    
    # Convert to numpy array (T, N)
    data_array = temp_df.values
    var_names = selected_metrics
    N = len(var_names)
    
    if len(temp_df) < 20:
        print("Too few data points to build causal graph")
        return None

    # 2. Create Tigramite DataFrame
    dataframe = pp.DataFrame(data_array, 
                             datatime=np.arange(len(data_array)), 
                             var_names=var_names)
    
    # 3. Initialize PCMCI
    # ParCorr is suitable for linear relationships, computationally fast. Use GPDC for strong nonlinearity (but very slow)
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
    
    # 4. Run full causal discovery
    print(f"Building causal graph for {N} metrics, max lag: {max_lag}...")
    results = pcmci.run_pcmci(tau_max=max_lag, pc_alpha=None, alpha_level=alpha)
    
    # 5. Parse results and build NetworkX graph
    # p_matrix and val_matrix dimensions are (N, N, tau_max+1)
    # p_matrix[i, j, tau] means: variable i (lag tau) -> variable j
    p_matrix = results['p_matrix']
    val_matrix = results['val_matrix']
    # graph = results['graph'] # Tigramite internal graph representation (unused)
    
    G = nx.DiGraph()
    edges_list = []
    
    # Add nodes
    for name in var_names:
        G.add_node(name)
        
    # Add edges
    # Iterate over all possible connections
    for j in range(N): # Target node (Effect)
        for i in range(N): # Source node (Cause)
            if i == j: continue
            
            # Aggregate edges from multiple lags
            max_strength = 0.0
            best_lag = -1
            min_p_val = 1.0
            has_causality = False
            
            for tau in range(1, max_lag + 1): # Lag
                p_val = p_matrix[i, j, tau]
                if p_val < alpha:
                    has_causality = True
                    strength = val_matrix[i, j, tau]
                    # Take the strength with the largest absolute value as representative
                    if abs(strength) > abs(max_strength):
                        max_strength = strength
                        best_lag = tau
                        min_p_val = p_val
            
            if has_causality:
                # Store edge info, resolve bidirectional conflicts later
                edges_list.append({
                    "source": var_names[i],
                    "target": var_names[j],
                    "lag": best_lag,
                    "strength": max_strength,
                    "weight": abs(max_strength),
                    "p_value": min_p_val
                })
    
    # --- Enforce unidirectional logic (Resolve Bidirectional Conflicts) ---
    # If A->B and B->A both exist, only keep the one with larger absolute Strength
    final_edges = []
    # Use dict for easy reverse edge lookup: key="A->B", value=edge_dict
    edge_map = {}
    
    for edge in edges_list:
        key = f"{edge['source']}->{edge['target']}"
        edge_map[key] = edge

    processed_pairs = set()
    
    for edge in edges_list:
        src = edge['source']
        tgt = edge['target']
        
        # If this node pair has been processed (e.g., when handling reverse edge), skip
        pair_key = tuple(sorted([src, tgt]))
        if pair_key in processed_pairs:
            continue
            
        reverse_key = f"{tgt}->{src}"
        
        if reverse_key in edge_map:
            # Found bidirectional edge, perform PK
            reverse_edge = edge_map[reverse_key]
            
            strength_fwd = abs(edge['strength'])
            strength_rev = abs(reverse_edge['strength'])
            
            if strength_fwd >= strength_rev:
                final_edges.append(edge) # Keep forward direction
            else:
                final_edges.append(reverse_edge) # Keep reverse direction
        else:
            # No reverse edge, keep directly
            final_edges.append(edge)
            
        processed_pairs.add(pair_key)
    
    # Add final filtered edges to NetworkX graph
    for edge in final_edges:
        G.add_edge(edge['source'], edge['target'], 
                   lag=edge['lag'], 
                   weight=edge['weight'], 
                   strength=edge['strength'],
                   p_value=edge['p_value'])
                    
    return {
        "nx_graph": G,
        "edges": final_edges,
        "raw_results": results
    }

def visualize_graph(G, title="Causal Graph", output_path="causal_graph.png"):
    """Simple visualization function"""
    plt.figure(figsize=(14, 10))
    # Increase k for more spread out nodes; increase iterations for more stable layout
    pos = nx.spring_layout(G, k=2.0, iterations=100)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color='lightblue', alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    # Draw edges (adjust thickness based on weight)
    weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    # connectionstyle='arc3,rad=0.1' makes bidirectional edges curved to avoid overlap
    # Add node_size parameter to prevent arrows from being hidden by nodes
    nx.draw_networkx_edges(G, pos, width=weights, arrowsize=25, arrowstyle='-|>', 
                           edge_color='gray', connectionstyle='arc3,rad=0.1', node_size=2500)
    
    # Draw edge labels (only show Strength)
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        # Prefer strength (signed), fall back to weight if not available
        val = d.get('strength', d.get('weight'))
        edge_labels[(u, v)] = f"{val:.2f}"
        
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_color='red', label_pos=0.3)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()  # Close figure to free memory
    print(f"Figure saved to: {output_path}")

def calculate_net_influence_from_edges(causal_edges, save_path=None):
    """
    Calculate net influence ranking from causal edge list, return CSV format results
    
    Args:
        causal_edges: causal edge list, each element is a dict containing 'source' and 'target'
        save_path: optional save path for statistical analysis results
        
    Returns:
        str: CSV format statistics results, including rank, metric name, net influence score, (out-degree/in-degree)
    """
    # 1. Build graph
    G = nx.DiGraph()
    for edge in causal_edges:
        G.add_edge(edge['source'], edge['target'])
    
    # Collect all output content for file saving
    output_lines = []
    
    output_lines.append("=== Graph Statistics ===")
    output_lines.append(f"Nodes: {G.number_of_nodes()}")
    output_lines.append(f"Edges: {G.number_of_edges()}")
    output_lines.append("-" * 30)

    # 2. Calculate Net Influence Score
    # Score = Out_Degree (times as cause) - In_Degree (times as effect)
    scores = {}
    details = {}
    
    for node in G.nodes():
        out_d = G.out_degree(node)
        in_d = G.in_degree(node)
        score = out_d - in_d
        
        scores[node] = score
        details[node] = {'out': out_d, 'in': in_d}
    
    # 3. Sort (score from high to low)
    # Higher score -> more likely root cause
    # Lower score -> more likely final symptom
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # 4. Build output results
    header = f"{'Rank':<5} {'Metric Name':<50} {'Net Influence':<10} {'(Out/In)'}"
    output_lines.append(header)
    
    separator = "=" * 80
    output_lines.append(separator)
    
    for rank, (node, score) in enumerate(sorted_nodes, 1):
        d = details[node]
        line = f"{rank:<5} {node:<50} {score:<10} ({d['out']}/{d['in']})"
        output_lines.append(line)

    output_lines.append(separator)
    
    guide_lines = [
        "Interpretation Guide:",
        "1. Top ranked (high positive score): primarily 'cause', influences other metrics, likely the root cause.",
        "2. Bottom ranked (low negative score): primarily 'effect', influenced by other metrics, typically a manifestation of the fault.",
        "3. Score near 0: may be an intermediate propagation node, or both cause and effect."
    ]
    
    for line in guide_lines:
        output_lines.append(line)
    
    # 5. Save results to file (if save_path specified)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        print(f"Statistics results saved to: {save_path}")
    
    # 6. Return CSV format results, matching terminal output table format exactly
    csv_lines = ["Rank,Metric Name,Net Influence,(Out/In)"]
    for rank, (node, score) in enumerate(sorted_nodes, 1):
        d = details[node]
        csv_lines.append(f"{rank},{node},{score},({d['out']}/{d['in']})")
    
    return "\n".join(csv_lines)

import json
import os

def case_causal_relations_process(base_collected_data_path, case_id, item_index=0):
    """
    Process causal graph construction for a single case

    Args:
        base_collected_data_path: raw data read path (for reading all_metrics.csv)
        case_id: case ID
        item_index: application index
    """
    try:
        print(f"\n=== Processing case: {case_id} ===")

        if case_id not in case_table:
            print(f"Case {case_id} does not exist in case_table")
            return False

        case_info = case_table[case_id]

        # Check if item_index is valid
        if item_index >= len(case_info['app_name']):
            print(f"item_index {item_index} out of range")
            return False

        app = case_info['app_name'][item_index]
        app_group = case_info['app_groups'][item_index][0]

        # Raw data path (read from DATA_READ_PATH)
        data_base_path = f"{base_collected_data_path}/{case_id}/{app}_{app_group}"
        data_path = f"{data_base_path}/metric/all_metrics.csv"

        # Analysis results path (read from RESULT_WRITE_PATH)
        result_summary_path = Config.get_result_summary(case_id, app, app_group)
        metric_json_path = f"{result_summary_path}/metric_analysis_result.txt"
        summary_path = result_summary_path
        
        # Check if files exist
        if not os.path.exists(data_path):
            print(f"Data file does not exist: {data_path}")
            return False
            
        if not os.path.exists(metric_json_path):
            print(f"Metric analysis file does not exist: {metric_json_path}")
            return False
        
        # Time window settings
        fault_start = case_info['fault_start']
        fault_start_ts = int((pd.to_datetime(fault_start) - pd.Timedelta(hours=8)).timestamp())*1000
        dectection_ts = fault_start_ts - 3600*1000 
        
        print(f"Reading data: {data_path}")
        df = pd.read_csv(data_path)
        df = df[(df['timestamp'] >= dectection_ts) & (df['timestamp'] <= fault_start_ts)]
        
        if len(df) < 20:
            print(f"Too few data points ({len(df)} points)")
            return False
        
        # Load metric analysis results
        with open(metric_json_path, "r") as f:
            metrics_data = json.load(f)
        
        target_metrics = []
        for metric in metrics_data:
            if metric['operational_assessment']['operational_severity'] == 'LOW':
                continue
            target_metrics.append(metric['metric_name'])
        
        print(f"Selected metrics ({len(target_metrics)} total): {target_metrics[:3]}..." if len(target_metrics) > 3 else f"Selected metrics: {target_metrics}")
        
        # Filter valid metrics
        valid_metrics = [m for m in target_metrics if m in df.columns]
        
        if len(valid_metrics) < 2:
            print(f"Insufficient valid metrics ({len(valid_metrics)} available)")
            return False
        
        print(f"Metrics used for construction: {len(valid_metrics)}")
        
        # Build causal graph
        result = build_causal_graph_pcmci(df, valid_metrics, max_lag=3)
        
        if not result:
            print("Causal graph construction failed")
            return False
        
        print(f"Construction complete! Found {len(result['edges'])} causal edges")
        
        causal_relations = []
        for edge in result['edges']:
            #print(f"  {edge['source']} -> {edge['target']} (Lag: {edge['lag']}, Strength: {edge['strength']:.3f})")
            causal_relations.append(f"{edge['source']} -> {edge['target']}")
        
        # Save to summary folder
        causal_edges_file = os.path.join(summary_path, "causal_edges.txt")
        with open(causal_edges_file, "w") as f:
            f.write("\n".join(causal_relations))
        print(f"Causal relations saved to: {causal_edges_file}")
        
        # Calculate and save net influence score statistics
        influence_stats_file = os.path.join(summary_path, "influence_statistics.txt")
        csv_result = calculate_net_influence_from_edges(result['edges'], save_path=influence_stats_file)
        
        # Save visualization image
        graph_file = os.path.join(summary_path, "causal_graph.png")
        visualize_graph(result['nx_graph'], title=f"Causal Graph - {case_id} ({app})", output_path=graph_file)
        
        return csv_result
        
    except Exception as e:
        print(f"Error processing case {case_id}: {str(e)}")
        return False

if __name__ == "__main__":
    # ===== Configuration area - modify settings here =====
    
    # Select processing mode
    PROCESS_MODE = "batch"  # "single" or "batch"
    
    if PROCESS_MODE == "single":
        # Single case processing configuration
        CASE_ID = "case20"  # Modify to the case ID you want to process
        ITEM_INDEX = 0      # Application index (for cases with multiple applications)
        
        print(f"Single case processing mode: {CASE_ID}")
            
    elif PROCESS_MODE == "batch":
        # Batch processing mode - specify case list to process here
        CASE_LIST = ["case23", "case24", "case25", "case29", "case30", "case36", "case37", "case43", "case44", "case46"]  # Modify to the case list you want to process
        
        print(f"Batch processing mode")
        # batch_process_cases(CASE_LIST)
        
    else:
        print("Invalid processing mode, please set PROCESS_MODE to 'single' or 'batch'")
