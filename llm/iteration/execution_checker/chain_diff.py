#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chain Diff Engine

Compute structural differences between two propagation chains
"""

from typing import Dict, List, Tuple, Any, Optional, Set

try:
    from .checker_utils import compute_lcs, edges_equal
except ImportError:
    from checker_utils import compute_lcs, edges_equal


class ChainDiff:
    """
    Propagation chain diff computation engine

    By comparing original and refined chain structures, identify all changes:
    - Added/removed nodes
    - Added/removed edges
    - Modified relation descriptions
    - Position mapping
    """

    def __init__(self, original_chain: Dict[str, Any], refined_chain: Dict[str, Any]):
        """
        Initialize ChainDiff

        Args:
            original_chain: Original chain dict
            refined_chain: Refined chain dict
        """
        self.original = original_chain
        self.refined = refined_chain
        self.chain_id = original_chain.get('chain_id')

    def compute_diff(self) -> Dict[str, Any]:
        """
        Compute complete chain diff

        Returns:
            Diff result dict
            {
                'chain_id': int,
                'length_change': {'old': int, 'new': int},
                'confidence_change': {'old': float, 'new': float},
                'summary_changed': bool,
                'nodes_added': [str],
                'nodes_removed': [str],
                'nodes_retained': [str],
                'edges_added': [dict],
                'edges_removed': [dict],
                'edges_modified': [dict],
                'step_sequence_original': [(from, to), ...],
                'step_sequence_refined': [(from, to), ...],
                'position_map': {original_idx: refined_idx}
            }
        """
        result = {
            'chain_id': self.chain_id
        }

        # 1. Compute length change
        result['length_change'] = self._compute_length_change()

        # 2. Compute confidence change
        result['confidence_change'] = self._compute_confidence_change()

        # 3. Check if summary changed
        result['summary_changed'] = self._check_summary_changed()

        # 4. Compute node changes
        node_changes = self._compute_node_changes()
        result['nodes_added'] = node_changes['added']
        result['nodes_removed'] = node_changes['removed']
        result['nodes_retained'] = node_changes['retained']

        # 5. Extract edge sequence
        orig_edges = self._extract_edges(self.original)
        refn_edges = self._extract_edges(self.refined)
        result['step_sequence_original'] = [(e['from'], e['to']) for e in orig_edges]
        result['step_sequence_refined'] = [(e['from'], e['to']) for e in refn_edges]

        # 6. Compute edge changes (using LCS)
        edge_changes = self._compute_edge_changes(orig_edges, refn_edges)
        result['edges_added'] = edge_changes['added']
        result['edges_removed'] = edge_changes['removed']
        result['edges_modified'] = edge_changes['modified']

        # 7. Build position map
        result['position_map'] = self._build_position_map(orig_edges, refn_edges)

        return result

    def _compute_length_change(self) -> Dict[str, int]:
        """Compute chain length change"""
        orig_len = len(self.original.get('chain', []))
        refn_len = len(self.refined.get('chain', []))
        return {'old': orig_len, 'new': refn_len}

    def _compute_confidence_change(self) -> Dict[str, float]:
        """Compute confidence change"""
        orig_conf = self.original.get('confidence', 0.0)
        refn_conf = self.refined.get('confidence', 0.0)
        return {'old': orig_conf, 'new': refn_conf}

    def _check_summary_changed(self) -> bool:
        """Check if summary changed"""
        orig_summary = self.original.get('summary', '')
        refn_summary = self.refined.get('summary', '')
        return orig_summary != refn_summary

    def _extract_nodes(self, chain: Dict[str, Any]) -> Set[str]:
        """
        Extract all unique nodes from chain

        Args:
            chain: Chain dict

        Returns:
            Node name set
        """
        nodes = set()
        for step in chain.get('chain', []):
            from_node = step.get('from')
            to_node = step.get('to')
            if from_node:
                nodes.add(from_node)
            if to_node:
                nodes.add(to_node)
        return nodes

    def _compute_node_changes(self) -> Dict[str, List[str]]:
        """
        Compute node changes

        Returns:
            {'added': [...], 'removed': [...], 'retained': [...]}
        """
        orig_nodes = self._extract_nodes(self.original)
        refn_nodes = self._extract_nodes(self.refined)

        return {
            'added': sorted(list(refn_nodes - orig_nodes)),
            'removed': sorted(list(orig_nodes - refn_nodes)),
            'retained': sorted(list(orig_nodes & refn_nodes))
        }

    def _extract_edges(self, chain: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract all edges from chain

        Args:
            chain: Chain dict

        Returns:
            Edge list [{'from': str, 'to': str, 'relation': str, ...}, ...]
        """
        return chain.get('chain', [])

    def _compute_edge_changes(self, orig_edges: List[Dict], refn_edges: List[Dict]) -> Dict[str, List]:
        """
        Compute edge changes (using LCS algorithm)

        Args:
            orig_edges: Original edge list
            refn_edges: Refined edge list

        Returns:
            {'added': [...], 'removed': [...], 'modified': [...]}
        """
        # Extract edge (from, to) tuple sequence
        orig_seq = [(e['from'], e['to']) for e in orig_edges]
        refn_seq = [(e['from'], e['to']) for e in refn_edges]

        # Compute LCS
        common_seq = compute_lcs(orig_seq, refn_seq)
        common_set = set(common_seq)

        # Find added edges
        added_edges = []
        for edge in refn_edges:
            edge_tuple = (edge['from'], edge['to'])
            if edge_tuple not in common_set:
                added_edges.append(edge)

        # Find removed edges
        removed_edges = []
        for edge in orig_edges:
            edge_tuple = (edge['from'], edge['to'])
            if edge_tuple not in common_set:
                removed_edges.append(edge)

        # Find edges with modified relations (same nodes, different relation)
        modified_edges = []
        for orig_edge in orig_edges:
            edge_tuple = (orig_edge['from'], orig_edge['to'])
            if edge_tuple in common_set:
                # Find corresponding edge in refined
                refn_edge = None
                for re in refn_edges:
                    if (re['from'], re['to']) == edge_tuple:
                        refn_edge = re
                        break

                if refn_edge and orig_edge.get('relation') != refn_edge.get('relation'):
                    modified_edges.append({
                        'from': orig_edge['from'],
                        'to': orig_edge['to'],
                        'original_relation': orig_edge.get('relation', ''),
                        'refined_relation': refn_edge.get('relation', '')
                    })

        return {
            'added': added_edges,
            'removed': removed_edges,
            'modified': modified_edges
        }

    def _build_position_map(self, orig_edges: List[Dict], refn_edges: List[Dict]) -> Dict[int, Optional[int]]:
        """
        Build mapping from original step index to refined step index

        Based on LCS results, map positions of retained edges

        Args:
            orig_edges: Original edge list
            refn_edges: Refined edge list

        Returns:
            {original_step_idx: refined_step_idx | None}
            Indices are 1-based
        """
        # Extract edge (from, to) tuple sequence
        orig_seq = [(e['from'], e['to']) for e in orig_edges]
        refn_seq = [(e['from'], e['to']) for e in refn_edges]

        # Compute LCS
        common_seq = compute_lcs(orig_seq, refn_seq)

        position_map = {}

        # Find corresponding refined step for each original step (if exists)
        for orig_idx, orig_tuple in enumerate(orig_seq):
            if orig_tuple in common_seq:
                # Find position in refined
                try:
                    refn_idx = refn_seq.index(orig_tuple)
                    position_map[orig_idx + 1] = refn_idx + 1  # Convert to 1-based
                except ValueError:
                    position_map[orig_idx + 1] = None
            else:
                position_map[orig_idx + 1] = None

        return position_map

    def get_step_at_position(self, chain: Dict[str, Any], position: int) -> Optional[Dict[str, Any]]:
        """
        Get step at specified position

        Args:
            chain: Chain dict
            position: Step position (1-based)

        Returns:
            Step dict, returns None if position is invalid
        """
        steps = chain.get('chain', [])
        if position < 1 or position > len(steps):
            return None
        return steps[position - 1]

    def find_edge_position(self, chain: Dict[str, Any], from_node: str, to_node: str) -> Optional[int]:
        """
        Find position of specified edge in chain

        Args:
            chain: Chain dict
            from_node: Start node
            to_node: Target node

        Returns:
            Step position (1-based), returns None if not found
        """
        steps = chain.get('chain', [])
        for idx, step in enumerate(steps):
            if step.get('from') == from_node and step.get('to') == to_node:
                return idx + 1
        return None


if __name__ == "__main__":
    # Simple test
    print("Testing ChainDiff...")

    # Test case 1: Node insertion
    original_chain = {
        'chain_id': 1,
        'summary': 'Original summary',
        'confidence': 0.8,
        'chain': [
            {'from': 'A', 'to': 'B', 'relation': 'causes', 'observed': True},
            {'from': 'B', 'to': 'C', 'relation': 'leads to', 'observed': True}
        ]
    }

    refined_chain = {
        'chain_id': 1,
        'summary': 'Original summary',
        'confidence': 0.85,
        'chain': [
            {'from': 'A', 'to': 'B', 'relation': 'causes', 'observed': True},
            {'from': 'B', 'to': 'X', 'relation': 'triggers', 'observed': True},  # Newly inserted
            {'from': 'X', 'to': 'C', 'relation': 'leads to', 'observed': True}
        ]
    }

    diff = ChainDiff(original_chain, refined_chain)
    result = diff.compute_diff()

    print("\n=== Test Case 1: Node Insertion ===")
    print(f"Length change: {result['length_change']}")
    print(f"Confidence change: {result['confidence_change']}")
    print(f"Nodes added: {result['nodes_added']}")
    print(f"Edges added: {len(result['edges_added'])} edges")
    print(f"Edges removed: {len(result['edges_removed'])} edges")
    print(f"Position map: {result['position_map']}")

    assert result['length_change'] == {'old': 2, 'new': 3}, "Length change test failed"
    assert 'X' in result['nodes_added'], "Node addition test failed"
    assert len(result['edges_added']) == 2, "Edge addition test failed"

    # Test case 2: Node deletion
    original_chain2 = {
        'chain_id': 2,
        'chain': [
            {'from': 'A', 'to': 'B', 'relation': 'causes'},
            {'from': 'B', 'to': 'C', 'relation': 'leads to'},
            {'from': 'C', 'to': 'D', 'relation': 'triggers'}
        ]
    }

    refined_chain2 = {
        'chain_id': 2,
        'chain': [
            {'from': 'A', 'to': 'B', 'relation': 'causes'},
            {'from': 'B', 'to': 'D', 'relation': 'directly triggers'}  # Skip C
        ]
    }

    diff2 = ChainDiff(original_chain2, refined_chain2)
    result2 = diff2.compute_diff()

    print("\n=== Test Case 2: Node Deletion ===")
    print(f"Length change: {result2['length_change']}")
    print(f"Nodes removed: {result2['nodes_removed']}")
    print(f"Edges removed: {len(result2['edges_removed'])} edges")

    assert result2['length_change'] == {'old': 3, 'new': 2}, "Length change test failed"
    assert 'C' in result2['nodes_removed'], "Node removal test failed"

    print("\n✓ All ChainDiff tests passed!")
