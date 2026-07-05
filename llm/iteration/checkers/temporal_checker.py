#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TemporalChecker - M2 Temporal Assertion Layer

3 temporal assertions:

    T1. monotonic_anomaly_timestamps    Anomaly timestamps of chain nodes should be monotonically non-decreasing
    T2. delay_within_tolerance          The gap between observed delta_t and PCMCI lag for adjacent nodes <= tolerance
    T3. root_anomaly_precedes_symptoms  Root cause node anomaly time should precede chain endpoint (symptom)

Dependencies: lag from CausalGraph + upstream `observed_anomaly_ts` ({metric: timestamp}).
Degrades to 'undecidable' when data is missing (no violation generated, marked as skipped in meta).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base_checker import (
    BaseChecker,
    CheckerKind,
    CritiqueResult,
    Severity,
    Violation,
)


class TemporalChecker(BaseChecker):

    kind = CheckerKind.TEMPORAL
    name = "temporal"

    def __init__(self, graph, delay_tolerance: float = 2.0):
        """
        Args:
            graph: CausalGraph, providing edge PCMCI lag
            delay_tolerance: Tolerance upper bound for |delta_t_obs - lag| (same unit as lag)
        """
        self.graph = graph
        self.delay_tolerance = delay_tolerance

    def check(
        self,
        chains: List[Dict[str, Any]],
        observed_anomaly_ts: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> CritiqueResult:
        violations: List[Violation] = []
        ts_map: Dict[str, float] = observed_anomaly_ts or {}

        skipped = {"t1_missing_ts": 0, "t2_missing_lag_or_ts": 0, "t3_missing_ts": 0}

        for chain_obj in chains or []:
            chain_id = str(chain_obj.get("chain_id", ""))
            steps = chain_obj.get("chain") or []
            nodes = self.steps_to_nodes(steps)
            if len(nodes) < 2:
                continue

            violations.extend(self._t1_monotonic_ts(chain_id, nodes, ts_map, skipped))
            violations.extend(self._t2_delay_tolerance(chain_id, nodes, ts_map, skipped))
            violations.extend(self._t3_root_before_symptom(
                chain_id, steps, nodes, ts_map, skipped))

        return CritiqueResult(
            checker=self.name,
            passed=not violations,
            violations=violations,
            meta={
                "n_chains": len(chains or []),
                "delay_tolerance": self.delay_tolerance,
                "skipped": skipped,
            },
        )

    # -------- Assertion implementations --------

    def _t1_monotonic_ts(
        self,
        chain_id: str,
        nodes: List[str],
        ts_map: Dict[str, float],
        skipped: Dict[str, int],
    ) -> List[Violation]:
        out: List[Violation] = []
        last_ts: Optional[float] = None
        last_node: Optional[str] = None
        for i, n in enumerate(nodes):
            ts = ts_map.get(n)
            if ts is None:
                skipped["t1_missing_ts"] += 1
                continue
            if last_ts is not None and ts + 1e-9 < last_ts:
                out.append(Violation(
                    checker=self.name,
                    assertion="monotonic_anomaly_timestamps",
                    chain_id=chain_id,
                    edge=[last_node or "", n],
                    position=i,
                    severity=Severity.HIGH,
                    detail=(
                        f"Time inversion: '{last_node}' anomaly @ {last_ts:.2f},"
                        f"but downstream '{n}' anomaly @ {ts:.2f}"
                    ),
                    suggested_action="Check if chain direction is reversed; or the two nodes may belong to different faults",
                ))
            last_ts = ts
            last_node = n
        return out

    def _t2_delay_tolerance(
        self,
        chain_id: str,
        nodes: List[str],
        ts_map: Dict[str, float],
        skipped: Dict[str, int],
    ) -> List[Violation]:
        out: List[Violation] = []
        for i in range(len(nodes) - 1):
            src, dst = nodes[i], nodes[i + 1]
            graph_lag = self.graph.get_edge_lag(src, dst) if self.graph.has_edge(src, dst) else None
            ts_src = ts_map.get(src)
            ts_dst = ts_map.get(dst)
            if graph_lag is None or ts_src is None or ts_dst is None:
                skipped["t2_missing_lag_or_ts"] += 1
                continue
            observed_dt = float(ts_dst) - float(ts_src)
            diff = abs(observed_dt - float(graph_lag))
            if diff > self.delay_tolerance:
                out.append(Violation(
                    checker=self.name,
                    assertion="delay_within_tolerance",
                    chain_id=chain_id,
                    edge=[src, dst],
                    position=i,
                    severity=Severity.MEDIUM,
                    detail=(
                        f"Edge '{src} -> {dst}' delay mismatch:"
                        f"Observed delta_t={observed_dt:.2f}, PCMCI lag={float(graph_lag):.2f},"
                        f"exceeds tolerance {self.delay_tolerance}"
                    ),
                    suggested_action="Verify sampling window or consider whether the edge passes through unobserved intermediate nodes",
                    meta={"observed_dt": observed_dt, "graph_lag": float(graph_lag)},
                ))
        return out

    def _t3_root_before_symptom(
        self,
        chain_id: str,
        steps: List[Dict[str, Any]],
        nodes: List[str],
        ts_map: Dict[str, float],
        skipped: Dict[str, int],
    ) -> List[Violation]:
        # Identify root cause node
        root_node: Optional[str] = None
        for s in steps:
            if not isinstance(s, dict):
                continue
            kind = (s.get("type") or s.get("node_type") or "").lower()
            from_kind = (s.get("from_type") or "").lower()
            if kind == "root_cause" and s.get("from"):
                root_node = s["from"]
                break
            if from_kind == "root_cause" and s.get("from"):
                root_node = s["from"]
                break
        if root_node is None and nodes:
            root_node = nodes[0]  # Fallback: use chain head as root cause

        symptom_node = nodes[-1] if nodes else None
        if not root_node or not symptom_node or root_node == symptom_node:
            return []

        ts_root = ts_map.get(root_node)
        ts_sym = ts_map.get(symptom_node)
        if ts_root is None or ts_sym is None:
            skipped["t3_missing_ts"] += 1
            return []

        if ts_root > ts_sym + 1e-9:
            return [Violation(
                checker=self.name,
                assertion="root_anomaly_precedes_symptoms",
                chain_id=chain_id,
                edge=[root_node, symptom_node],
                severity=Severity.HIGH,
                detail=(
                    f"Root cause '{root_node}' anomaly @ {ts_root:.2f} is later than symptom "
                    f"'{symptom_node}' @ {ts_sym:.2f}"
                ),
                suggested_action="Re-determine root cause node, or check chain direction",
            )]
        return []
