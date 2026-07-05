#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CausalConstrainedGenerator - M1 main class: causal constrained chain generation

Soft mode (this implementation):
    LLM generates freely -> validate with CausalGraph -> attempt repair / reject and resample if invalid

Future hard mode (reserved extension point):
    Use logit mask on open-source LLM to enforce each generated node in valid neighbor set
    (requires outlines / lm-format-enforcer)

Core constraints:
- C1: Every adjacent node pair (u, v) in chain must satisfy (u, v) in G.edges
- C2: No duplicate nodes (chain is a simple path)
- C3: Nodes must exist in G

Repair strategies (by priority):
- R1: After removing invalid edge, if endpoints still connected (<= max_repair_hops hops), insert shortest path
- R2: Otherwise truncate the invalid edge segment entirely (keep valid prefix as new chain)

Does not modify original LLM calling convention: generator only accepts a no-arg callable as "resample action",
specific prompt assembly and API calls are handled by upper layer (AnalysisAgent).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from llm.causal_graph import CausalGraph


class ViolationType(str, Enum):
    UNKNOWN_NODE = "unknown_node"             # Node not in G
    MISSING_EDGE = "missing_edge"             # Adjacent edge not in G
    NODE_REPETITION = "node_repetition"       # Node repeated in chain
    EMPTY_CHAIN = "empty_chain"               # Chain is empty or has only 1 node


@dataclass
class Violation:
    type: ViolationType
    position: int                   # Problematic node position / edge left endpoint index
    detail: str = ""


@dataclass
class ValidationResult:
    passed: bool
    violations: List[Violation] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "violations": [
                {"type": v.type.value, "position": v.position, "detail": v.detail}
                for v in self.violations
            ],
        }


@dataclass
class RepairResult:
    original_chain: List[str]
    repaired_chain: List[str]
    actions: List[str]               # Description of repair actions performed
    fully_valid: bool                # Whether fully valid after repair


@dataclass
class ScoredChain:
    chain: List[str]
    validity: ValidationResult
    repair: Optional[RepairResult] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.validity.passed


class CausalConstrainedGenerator:
    """
    Causal constrained chain generator (soft mode)

    Usage:
        gen = CausalConstrainedGenerator(graph)

        # 1) Single validation
        result = gen.validate(['a', 'b', 'c'])

        # 2) Validate + repair
        repaired = gen.validate_and_repair(['a', 'b', 'c'])

        # 3) Rejection sampling: pass resample callback
        def resample():
            return llm_client.call(prompt)  # assembled by upper layer
        chain = gen.generate_with_resampling(initial_chain, resample, max_retries=3)
    """

    def __init__(
        self,
        graph: CausalGraph,
        max_repair_hops: int = 3,
        allow_repair: bool = True,
    ):
        self.graph = graph
        self.max_repair_hops = max_repair_hops
        self.allow_repair = allow_repair

    # -------- 1. Validation --------

    def validate(self, chain: List[str]) -> ValidationResult:
        """Validate hard constraints on a node chain"""
        violations: List[Violation] = []

        if not chain or len(chain) < 2:
            violations.append(
                Violation(ViolationType.EMPTY_CHAIN, 0, "Chain requires at least 2 nodes")
            )
            return ValidationResult(passed=False, violations=violations)

        seen = {}
        for i, node in enumerate(chain):
            if not self.graph.has_node(node):
                violations.append(
                    Violation(ViolationType.UNKNOWN_NODE, i,
                              f"Node '{node}' not in causal graph")
                )
            if node in seen:
                violations.append(
                    Violation(ViolationType.NODE_REPETITION, i,
                              f"Node '{node}' already appeared at position {seen[node]}")
                )
            else:
                seen[node] = i

        for i in range(len(chain) - 1):
            src, dst = chain[i], chain[i + 1]
            if (self.graph.has_node(src) and self.graph.has_node(dst)
                    and not self.graph.has_edge(src, dst)):
                violations.append(
                    Violation(ViolationType.MISSING_EDGE, i,
                              f"Edge '{src} -> {dst}' not in causal graph")
                )

        return ValidationResult(passed=not violations, violations=violations)

    # -------- 2. Repair --------

    def repair(self, chain: List[str]) -> RepairResult:
        """
        Attempt to repair an invalid chain into a valid one (core action of soft mode)

        Strategy:
        - Unknown nodes: remove directly from chain
        - Missing edges: try to find shortest path src->dst in G (<= max_repair_hops hops) for insertion;
                         if not found, truncate (keep prefix)
        - Duplicate nodes: deduplicate keeping first occurrence
        """
        actions: List[str] = []
        if not chain:
            return RepairResult(original_chain=[], repaired_chain=[], actions=[], fully_valid=False)

        # Step 1: Remove unknown nodes
        cleaned = []
        for node in chain:
            if self.graph.has_node(node):
                cleaned.append(node)
            else:
                actions.append(f"Removed unknown node '{node}'")

        if len(cleaned) < 2:
            return RepairResult(
                original_chain=list(chain),
                repaired_chain=cleaned,
                actions=actions,
                fully_valid=False,
            )

        # Step 2: Deduplicate (keep first occurrence)
        deduped = []
        seen = set()
        for node in cleaned:
            if node in seen:
                actions.append(f"Removed duplicate node '{node}'")
                continue
            seen.add(node)
            deduped.append(node)

        if len(deduped) < 2:
            return RepairResult(
                original_chain=list(chain),
                repaired_chain=deduped,
                actions=actions,
                fully_valid=False,
            )

        # Step 3: Repair missing edges (insertion / truncation)
        repaired = [deduped[0]]
        truncated = False
        for i in range(len(deduped) - 1):
            src = repaired[-1]
            dst = deduped[i + 1]
            if self.graph.has_edge(src, dst):
                repaired.append(dst)
                continue

            shortest = self.graph.shortest_path(src, dst, max_hops=self.max_repair_hops)
            if shortest and len(shortest) >= 2:
                # Insert intermediate nodes (remove duplicate src)
                inserted = shortest[1:]
                # Skip already seen nodes (ensure simple path)
                inserted_clean = [n for n in inserted if n not in repaired]
                if inserted_clean and inserted_clean[-1] == dst:
                    repaired.extend(inserted_clean)
                    if len(inserted_clean) > 1:
                        actions.append(
                            f"Inserted path '{src} -> ... -> {dst}', added intermediate nodes "
                            f"{inserted_clean[:-1]}"
                        )
                    continue

            # No reasonable path found -> truncate
            actions.append(f"Truncated: edge '{src} -> {dst}' is unrepairable, stopped at '{src}'")
            truncated = True
            break

        validity = self.validate(repaired)
        if truncated and not validity.passed:
            # Should not happen theoretically, fallback
            pass

        return RepairResult(
            original_chain=list(chain),
            repaired_chain=repaired,
            actions=actions,
            fully_valid=validity.passed,
        )

    def validate_and_repair(self, chain: List[str]) -> ScoredChain:
        """Validate, and attempt repair if validation fails and allow_repair=True"""
        validity = self.validate(chain)
        if validity.passed or not self.allow_repair:
            return ScoredChain(chain=list(chain), validity=validity, repair=None)

        repair = self.repair(chain)
        post_validity = self.validate(repair.repaired_chain)
        return ScoredChain(
            chain=repair.repaired_chain,
            validity=post_validity,
            repair=repair,
        )

    # -------- 3. Rejection sampling --------

    def generate_with_resampling(
        self,
        initial_chain: List[str],
        resample_fn: Callable[[], List[str]],
        max_retries: int = 2,
    ) -> ScoredChain:
        """
        Core entry point for soft mode:
        1) Validate current chain; return if valid
        2) Otherwise call resample_fn for resampling, up to max_retries times
        3) If all fail, fallback to repair (if allow_repair is enabled)

        Args:
            initial_chain: First LLM output
            resample_fn: No-arg callable, each call should return new chain (List[str])
            max_retries: Max number of resampling attempts
        """
        current = list(initial_chain)
        best_so_far = ScoredChain(chain=current, validity=self.validate(current))
        if best_so_far.is_valid:
            return best_so_far

        for attempt in range(max_retries):
            try:
                resampled = resample_fn()
            except Exception as e:
                best_so_far.metadata.setdefault("resample_errors", []).append(str(e))
                break

            if not resampled:
                continue

            check = self.validate(resampled)
            if check.passed:
                return ScoredChain(
                    chain=list(resampled),
                    validity=check,
                    metadata={"attempts": attempt + 1},
                )
            # Record intermediate result with fewer violations
            if len(check.violations) < len(best_so_far.validity.violations):
                best_so_far = ScoredChain(chain=list(resampled), validity=check)

        # All resampling attempts failed -> final repair attempt
        if self.allow_repair:
            return self.validate_and_repair(best_so_far.chain)
        return best_so_far

    # -------- 4. Batch entry (process propagation_chains JSON structure) --------

    def constrain_propagation_chains(self, propagation_chains: List[Dict]) -> Dict:
        """
        Constrain-process the propagation_chains JSON directly produced by LLM (as expected by chunk_processor).

        Input example (aligned with AnalysisAgent output):
            [
              {
                "chain_id": "c1",
                "confidence": 0.8,
                "chain": [
                    {"from": "metric_a", "to": "metric_b", "relation": "causes", ...},
                    ...
                ]
              }
            ]

        Output:
            {
              "constrained_chains": [...],  # Same structure, but only valid/repaired chains kept
              "report": {
                  "total": int,
                  "fully_valid_before": int,
                  "fully_valid_after": int,
                  "repaired": int,
                  "dropped": int,
                  "per_chain": [...],  # Per-chain details
              }
            }
        """
        report = {
            "total": len(propagation_chains or []),
            "fully_valid_before": 0,
            "fully_valid_after": 0,
            "repaired": 0,
            "dropped": 0,
            "per_chain": [],
        }
        constrained: List[Dict] = []

        for chain_obj in propagation_chains or []:
            steps = chain_obj.get("chain") or []
            node_seq = self._steps_to_node_seq(steps)

            scored = self.validate_and_repair(node_seq)
            pre_validity = self.validate(node_seq)
            if pre_validity.passed:
                report["fully_valid_before"] += 1

            per = {
                "chain_id": chain_obj.get("chain_id"),
                "original_nodes": node_seq,
                "repaired_nodes": scored.chain,
                "pre_valid": pre_validity.passed,
                "post_valid": scored.is_valid,
                "violations_before": pre_validity.to_dict()["violations"],
                "actions": scored.repair.actions if scored.repair else [],
            }
            report["per_chain"].append(per)

            if scored.is_valid:
                report["fully_valid_after"] += 1
                if scored.repair and scored.repair.actions:
                    report["repaired"] += 1
                # Reassemble repaired node sequence into steps output
                constrained.append(self._rewrite_chain_obj(chain_obj, scored.chain))
            else:
                report["dropped"] += 1

        return {"constrained_chains": constrained, "report": report}

    # -------- Private --------

    @staticmethod
    def _steps_to_node_seq(steps: List[Dict]) -> List[str]:
        if not steps:
            return []
        seq = []
        for i, s in enumerate(steps):
            src = s.get("from")
            dst = s.get("to")
            if not src or not dst:
                continue
            if i == 0:
                seq.append(src)
            seq.append(dst)
        return seq

    @staticmethod
    def _rewrite_chain_obj(chain_obj: Dict, node_seq: List[str]) -> Dict:
        """Reassemble repaired node sequence into dict isomorphic to original chain object"""
        if len(node_seq) < 2:
            return chain_obj

        # Preserve original relation/when/observed fields, align by position; fill defaults for missing positions
        original_steps = chain_obj.get("chain") or []

        new_steps = []
        for i in range(len(node_seq) - 1):
            src, dst = node_seq[i], node_seq[i + 1]
            if i < len(original_steps):
                template = dict(original_steps[i])
            else:
                template = {}
            template["from"] = src
            template["to"] = dst
            template.setdefault("relation", "causes")
            template.setdefault("observed", False)
            template.setdefault("inferred", True)
            new_steps.append(template)

        rewritten = dict(chain_obj)
        rewritten["chain"] = new_steps
        return rewritten


def create_constrained_generator(causal_analysis, **kwargs) -> CausalConstrainedGenerator:
    """Convenience factory: construct generator from any form of causal_analysis input"""
    from llm.causal_graph.loader import autoload
    graph = autoload(causal_analysis)
    return CausalConstrainedGenerator(graph=graph, **kwargs)
