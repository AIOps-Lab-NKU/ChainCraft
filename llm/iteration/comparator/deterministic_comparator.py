#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeterministicComparator - M2 Deterministic Comparator

Use an interpretable scalar formula to determine if refined_chains are better than original_chains:

    quality_score(chains)
        = w_structural * (1 - normalized_structural_penalty)
        + w_temporal   * (1 - normalized_temporal_penalty)
        + w_semantic   * (1 - normalized_semantic_penalty)
        - w_uncertainty * vote_entropy_of_critic_ensemble

Final decision:
    better = refined.score > original.score + tie_margin
    worse  = refined.score < original.score - tie_margin
    tie    = otherwise

Output is isomorphic to the original Comparator (ComparatorResult), controller needs no branch handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .schemas import (
    ChainComparison,
    ComparatorResult,
    ComparisonResult,
    ComparisonSummary,
    Confidence,
    FixedIssue,
    NewIssue,
    RemainingIssue,
)


DEFAULT_WEIGHTS = {
    "structural": 0.4,
    "temporal": 0.25,
    "semantic": 0.25,
    "uncertainty": 0.1,
}


@dataclass
class QualityBreakdown:
    structural: float
    temporal: float
    semantic: float
    uncertainty: float
    score: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "structural": self.structural,
            "temporal": self.temporal,
            "semantic": self.semantic,
            "uncertainty": self.uncertainty,
            "score": self.score,
        }


class DeterministicComparator:

    def __init__(
        self,
        critic_pipeline,                   # Implements .critique(chains, **kwargs) -> AggregatedCritique
        weights: Optional[Dict[str, float]] = None,
        tie_margin: float = 0.02,
        confidence_high_gap: float = 0.10,
    ):
        self.pipeline = critic_pipeline
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.tie_margin = tie_margin
        self.confidence_high_gap = confidence_high_gap

    # -------- Main entry --------

    def compare(
        self,
        original_chains: List[Dict],
        refined_chains: List[Dict],
        evaluator_result: Optional[Dict] = None,
        refine_result: Optional[Dict] = None,
        execution_check_result: Optional[Dict] = None,
        causal_analysis: Optional[Any] = None,
        metric_analysis: Optional[Any] = None,
        observed_anomaly_ts: Optional[Dict[str, float]] = None,
        observed_symptoms: Optional[List[str]] = None,
        **kwargs,
    ) -> ComparatorResult:
        orig_critique = self.pipeline.critique(
            original_chains,
            observed_symptoms=observed_symptoms,
            observed_anomaly_ts=observed_anomaly_ts,
            causal_analysis=causal_analysis,
            metric_analysis=metric_analysis,
        )
        refined_critique = self.pipeline.critique(
            refined_chains,
            observed_symptoms=observed_symptoms,
            observed_anomaly_ts=observed_anomaly_ts,
            causal_analysis=causal_analysis,
            metric_analysis=metric_analysis,
        )

        orig_score = self._score(orig_critique)
        new_score = self._score(refined_critique)
        delta = new_score.score - orig_score.score

        if delta > self.tie_margin:
            result_type = ComparisonResult.BETTER
        elif delta < -self.tie_margin:
            result_type = ComparisonResult.WORSE
        else:
            result_type = ComparisonResult.TIE

        if abs(delta) >= self.confidence_high_gap:
            confidence = Confidence.HIGH
        elif abs(delta) >= self.tie_margin:
            confidence = Confidence.MEDIUM
        else:
            confidence = Confidence.LOW

        keep_new = result_type == ComparisonResult.BETTER or (
            result_type == ComparisonResult.TIE and new_score.score >= orig_score.score
        )

        fixed_issues, new_issues, remaining_issues, chain_comparisons = (
            self._diff_violations(
                orig_critique, refined_critique, original_chains, refined_chains
            )
        )

        reason = (
            f"[Deterministic Comparison] score_old={orig_score.score:.3f} -> "
            f"score_new={new_score.score:.3f} (delta={delta:+.3f}, margin={self.tie_margin}). "
            f"Breakdown old={orig_score.to_dict()}; new={new_score.to_dict()}. "
            f"Fixed {len(fixed_issues)}, introduced {len(new_issues)}, remaining {len(remaining_issues)}."
        )

        summary = ComparisonSummary(
            comparison_result=result_type,
            keep_new_version=keep_new,
            confidence=confidence,
        )

        return ComparatorResult(
            comparison_summary=summary,
            fixed_issues=fixed_issues,
            remaining_issues=remaining_issues,
            new_issues=new_issues,
            chain_comparisons=chain_comparisons,
            reason=reason,
        )

    # -------- Scoring --------

    def _score(self, aggregated) -> QualityBreakdown:
        """
        aggregated: AggregatedCritique (has .results list + .severity_counts() + .penalty_score())
        Aggregate penalties from different checkers by type, then apply simple linear scaling without softmax.
        """
        per_kind_penalty = {"structural": 0.0, "temporal": 0.0, "semantic": 0.0}
        uncertainty = 0.0

        for r in aggregated.results:
            kind = self._infer_kind(r)
            for v in r.violations:
                per_kind_penalty[kind] += self._sev_weight(v.severity)
            if isinstance(r.meta, dict):
                ent = r.meta.get("vote_entropy")
                if isinstance(ent, (int, float)):
                    uncertainty = max(uncertainty, float(ent))

        # Map penalties to [0, 1], clip above 1
        def clip(x): return max(0.0, min(1.0, 1.0 - x))
        s = clip(per_kind_penalty["structural"])
        t = clip(per_kind_penalty["temporal"])
        sem = clip(per_kind_penalty["semantic"])
        u = min(1.0, uncertainty)

        score = (
            self.weights["structural"] * s
            + self.weights["temporal"] * t
            + self.weights["semantic"] * sem
            - self.weights["uncertainty"] * u
        )
        return QualityBreakdown(s, t, sem, u, score)

    @staticmethod
    def _sev_weight(severity) -> float:
        s = severity.value if hasattr(severity, "value") else str(severity)
        return {"low": 0.05, "medium": 0.15, "high": 0.35, "critical": 0.6}.get(s, 0.15)

    @staticmethod
    def _infer_kind(r) -> str:
        name = (r.checker or "").lower()
        if "structural" in name:
            return "structural"
        if "temporal" in name:
            return "temporal"
        # Fallback: ensemble / semantic all map to semantic
        return "semantic"

    # -------- Diff: convert violation set diff to ComparatorResult fields --------

    @staticmethod
    def _violation_key(v) -> tuple:
        return (v.chain_id or "", v.assertion, "->".join(v.edge) if v.edge else "")

    def _diff_violations(self, orig_agg, new_agg, original_chains, refined_chains):
        orig_v = {self._violation_key(v): v for v in orig_agg.all_violations}
        new_v = {self._violation_key(v): v for v in new_agg.all_violations}

        fixed_issues: List[FixedIssue] = []
        remaining_issues: List[RemainingIssue] = []
        for key, v in orig_v.items():
            if key not in new_v:
                fixed_issues.append(FixedIssue(
                    chain_id=self._coerce_chain_id(v.chain_id),
                    issue_type=v.assertion,
                    original_severity=v.severity.value if hasattr(v.severity, "value") else str(v.severity),
                    status="fixed",
                    description=v.detail,
                ))
            else:
                remaining_issues.append(RemainingIssue(
                    chain_id=self._coerce_chain_id(v.chain_id),
                    issue_type=v.assertion,
                    status="still_exists",
                    description=v.detail,
                ))

        new_issues: List[NewIssue] = []
        for key, v in new_v.items():
            if key not in orig_v:
                new_issues.append(NewIssue(
                    chain_id=self._coerce_chain_id(v.chain_id),
                    issue_type=v.assertion,
                    severity=v.severity.value if hasattr(v.severity, "value") else str(v.severity),
                    description=v.detail,
                ))

        # Simple comparison per chain
        chain_comparisons: List[ChainComparison] = []
        chain_ids = {c.get("chain_id") for c in (refined_chains or [])}
        for cid in chain_ids:
            cid_str = str(cid) if cid is not None else ""
            fixed_in_chain = sum(1 for v in orig_v.values() if (v.chain_id or "") == cid_str
                                 and self._violation_key(v) not in new_v)
            new_in_chain = sum(1 for v in new_v.values() if (v.chain_id or "") == cid_str
                               and self._violation_key(v) not in orig_v)
            if fixed_in_chain > new_in_chain:
                res, reason = ComparisonResult.BETTER, f"Fixed {fixed_in_chain}, introduced {new_in_chain}"
            elif new_in_chain > fixed_in_chain:
                res, reason = ComparisonResult.WORSE, f"Fixed {fixed_in_chain}, introduced {new_in_chain}"
            else:
                res, reason = ComparisonResult.TIE, f"Fixed and introduced are comparable ({fixed_in_chain})"
            chain_comparisons.append(ChainComparison(
                chain_id=self._coerce_chain_id(cid),
                result=res,
                reason=reason,
            ))

        return fixed_issues, new_issues, remaining_issues, chain_comparisons

    @staticmethod
    def _coerce_chain_id(cid) -> int:
        if isinstance(cid, int):
            return cid
        try:
            return int(cid)
        except (TypeError, ValueError):
            return 0
