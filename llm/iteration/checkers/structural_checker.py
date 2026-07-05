#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StructuralChecker - M2 Structural Assertion Layer

Independently and verifiably run 5 structural assertions on a batch of propagation_chains:

    A1. every_edge_in_causal_graph   Every edge must exist in the PCMCI causal graph
    A2. no_cycles                    No cycles allowed on a single chain
    A3. root_no_incoming_in_chain    Nodes marked as root_cause should not be targets within the chain
    A4. endpoints_in_observed_symptoms  Chain endpoints should be in the anomaly metric set
    A5. no_node_repetition           No repeated nodes on the chain

All assertions only depend on: CausalGraph + optional context (observed anomaly metric set).
No LLM calls. Each failed assertion generates a Violation.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

from .base_checker import (
    BaseChecker,
    CheckerKind,
    CritiqueResult,
    Severity,
    Violation,
)


class StructuralChecker(BaseChecker):

    kind = CheckerKind.STRUCTURAL
    name = "structural"

    def __init__(self, graph):
        self.graph = graph

    # -------- Main entry --------

    def check(
        self,
        chains: List[Dict[str, Any]],
        observed_symptoms: Optional[Iterable[str]] = None,
        **kwargs,
    ) -> CritiqueResult:
        observed_set: Set[str] = set(observed_symptoms or [])
        violations: List[Violation] = []

        for chain_obj in chains or []:
            chain_id = str(chain_obj.get("chain_id", ""))
            steps = chain_obj.get("chain") or []
            nodes = self.steps_to_nodes(steps)

            if len(nodes) < 2:
                violations.append(Violation(
                    checker=self.name,
                    assertion="non_empty_chain",
                    chain_id=chain_id,
                    severity=Severity.HIGH,
                    detail="Chain requires at least 2 nodes",
                    suggested_action="Complete chain nodes or discard the chain",
                ))
                continue

            violations.extend(self._a1_every_edge_in_graph(chain_id, nodes))
            violations.extend(self._a2_no_cycles(chain_id, nodes))
            violations.extend(self._a3_root_no_incoming(chain_id, steps, nodes))
            violations.extend(self._a4_endpoints_in_observed(
                chain_id, nodes, observed_set))
            violations.extend(self._a5_no_node_repetition(chain_id, nodes))

        return CritiqueResult(
            checker=self.name,
            passed=not violations,
            violations=violations,
            meta={
                "n_chains": len(chains or []),
                "n_assertions": 5,
            },
        )

    # -------- Individual assertions --------

    def _a1_every_edge_in_graph(self, chain_id: str, nodes: List[str]) -> List[Violation]:
        out: List[Violation] = []
        for i in range(len(nodes) - 1):
            src, dst = nodes[i], nodes[i + 1]
            if not self.graph.has_node(src):
                out.append(Violation(
                    checker=self.name,
                    assertion="every_edge_in_causal_graph",
                    chain_id=chain_id,
                    edge=[src, dst],
                    position=i,
                    severity=Severity.HIGH,
                    detail=f"Node '{src}' is not in the causal graph",
                    suggested_action=f"Remove node '{src}' or replace with an adjacent valid metric in the graph",
                ))
                continue
            if not self.graph.has_node(dst):
                out.append(Violation(
                    checker=self.name,
                    assertion="every_edge_in_causal_graph",
                    chain_id=chain_id,
                    edge=[src, dst],
                    position=i,
                    severity=Severity.HIGH,
                    detail=f"Node '{dst}' is not in the causal graph",
                    suggested_action=f"Remove node '{dst}' or replace with an adjacent valid metric in the graph",
                ))
                continue
            if not self.graph.has_edge(src, dst):
                out.append(Violation(
                    checker=self.name,
                    assertion="every_edge_in_causal_graph",
                    chain_id=chain_id,
                    edge=[src, dst],
                    position=i,
                    severity=Severity.HIGH,
                    detail=f"Edge '{src} -> {dst}' is not in the causal graph",
                    suggested_action=f"Insert the shortest valid path between '{src}' and '{dst}' in the graph, or remove one endpoint",
                ))
        return out

    def _a2_no_cycles(self, chain_id: str, nodes: List[str]) -> List[Violation]:
        seen: Dict[str, int] = {}
        out: List[Violation] = []
        for i, n in enumerate(nodes):
            if n in seen and seen[n] < i - 1:
                # Non-adjacent repetition detected -> forms cycle or chain backtrack
                out.append(Violation(
                    checker=self.name,
                    assertion="no_cycles",
                    chain_id=chain_id,
                    position=i,
                    severity=Severity.HIGH,
                    detail=f"Node '{n}' repeats on chain (positions {seen[n]} and {i}), forming a cycle",
                    suggested_action="Truncate the backtrack segment to keep the chain as a DAG",
                ))
            seen.setdefault(n, i)
        return out

    def _a3_root_no_incoming(
        self,
        chain_id: str,
        steps: List[Dict[str, Any]],
        nodes: List[str],
    ) -> List[Violation]:
        """If step.type is marked as 'root_cause', this node should not be a target of any edge"""
        root_nodes: Set[str] = set()
        for s in steps:
            if not isinstance(s, dict):
                continue
            kind = (s.get("type") or s.get("node_type") or "").lower()
            from_kind = (s.get("from_type") or "").lower()
            to_kind = (s.get("to_type") or "").lower()
            if kind == "root_cause" and s.get("from"):
                root_nodes.add(s["from"])
            if from_kind == "root_cause" and s.get("from"):
                root_nodes.add(s["from"])
            if to_kind == "root_cause" and s.get("to"):
                root_nodes.add(s["to"])

        out: List[Violation] = []
        for i in range(1, len(nodes)):  # Skip chain head
            if nodes[i] in root_nodes:
                out.append(Violation(
                    checker=self.name,
                    assertion="root_no_incoming_in_chain",
                    chain_id=chain_id,
                    edge=[nodes[i - 1], nodes[i]],
                    position=i,
                    severity=Severity.MEDIUM,
                    detail=f"Node '{nodes[i]}' marked as root_cause appears mid-chain with incoming edges",
                    suggested_action="Move root cause node to chain head, or correct the node type label",
                ))
        return out

    def _a4_endpoints_in_observed(
        self,
        chain_id: str,
        nodes: List[str],
        observed: Set[str],
    ) -> List[Violation]:
        if not observed:
            return []
        tail = nodes[-1]
        if tail in observed:
            return []
        return [Violation(
            checker=self.name,
            assertion="endpoints_in_observed_symptoms",
            chain_id=chain_id,
            position=len(nodes) - 1,
            severity=Severity.MEDIUM,
            detail=f"Chain endpoint '{tail}' is not in the observed anomaly metric set",
            suggested_action="Extend chain to a real anomaly metric, or truncate to the last anomaly metric",
        )]

    def _a5_no_node_repetition(self, chain_id: str, nodes: List[str]) -> List[Violation]:
        seen: Dict[str, int] = {}
        out: List[Violation] = []
        for i, n in enumerate(nodes):
            if n in seen:
                out.append(Violation(
                    checker=self.name,
                    assertion="no_node_repetition",
                    chain_id=chain_id,
                    position=i,
                    severity=Severity.LOW,
                    detail=f"Node '{n}' is duplicated (first appeared at position {seen[n]})",
                    suggested_action="Deduplicate, keeping the first occurrence",
                ))
            seen.setdefault(n, i)
        return out
