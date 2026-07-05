#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SemanticCritic - M2 Semantic Layer Critic

Responsible for evaluating "chains that appear valid but may not be semantically reasonable". For example:
    - Whether chain endpoints truly explain the business impact of observed anomalies
    - Whether mechanism descriptions are self-consistent
    - Whether unobserved but potentially existing implicit intermediate steps are included

Interface has two layers:
    1) BaseSemanticCritic  - Abstract interface
    2) NoopSemanticCritic  - Default implementation: returns passed=True, empty violations, for offline/unit testing
    3) LLMSemanticCritic   - Optional implementation: feeds chains + context to EvaluatorAgent,
                              parses its hard_violations / soft_issues into Violations

Design motivation:
    - Allow IterationController to run without LLM environment (Noop fallback)
    - Production environment just needs to inject one LLMSemanticCritic instance
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


class BaseSemanticCritic(BaseChecker):

    kind = CheckerKind.SEMANTIC
    name = "semantic"

    def check(self, chains: List[Dict[str, Any]], **kwargs) -> CritiqueResult:
        raise NotImplementedError


class NoopSemanticCritic(BaseSemanticCritic):
    """No-op implementation, always passed=True; for local self-check / no-LLM environments"""

    name = "semantic_noop"

    def check(self, chains: List[Dict[str, Any]], **kwargs) -> CritiqueResult:
        return CritiqueResult(
            checker=self.name,
            passed=True,
            violations=[],
            meta={"reason": "noop critic; semantic check skipped"},
        )


class LLMSemanticCritic(BaseSemanticCritic):
    """
    Use existing EvaluatorAgent for semantic evaluation. Map its output hard_violations / soft_issues
    uniformly to Violations.
    """

    name = "semantic_llm"

    SEVERITY_MAP = {
        "low": Severity.LOW,
        "medium": Severity.MEDIUM,
        "high": Severity.HIGH,
        "critical": Severity.CRITICAL,
    }

    def __init__(self, evaluator_agent, critic_id: int = 0):
        self.evaluator = evaluator_agent
        self.critic_id = critic_id

    def check(
        self,
        chains: List[Dict[str, Any]],
        causal_analysis: Optional[str] = None,
        metric_analysis: Optional[Any] = None,
        iteration_context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> CritiqueResult:
        try:
            evaluation = self.evaluator.evaluate_chains(
                propagation_chains=chains,
                causal_analysis=causal_analysis,
                metric_analysis=metric_analysis,
                iteration_context=iteration_context,
                save_path=None,
            )
        except Exception as e:
            return CritiqueResult(
                checker=self.name,
                passed=False,
                violations=[Violation(
                    checker=self.name,
                    assertion="evaluator_call_failed",
                    severity=Severity.LOW,
                    detail=f"LLM critic call failed: {e}",
                )],
                meta={"critic_id": self.critic_id, "error": str(e)},
            )

        violations: List[Violation] = []
        for chain_eval in evaluation.get("chain_evaluations", []) or []:
            chain_id = str(chain_eval.get("chain_id", ""))
            for issue in chain_eval.get("hard_violations", []) or []:
                violations.append(self._issue_to_violation(chain_id, issue, default=Severity.HIGH))
            for issue in chain_eval.get("soft_issues", []) or []:
                violations.append(self._issue_to_violation(chain_id, issue, default=Severity.MEDIUM))

        return CritiqueResult(
            checker=self.name,
            passed=not violations,
            violations=violations,
            meta={
                "critic_id": self.critic_id,
                "summary": evaluation.get("summary", {}),
            },
        )

    def _issue_to_violation(self, chain_id: str, issue: Dict[str, Any],
                            default: Severity) -> Violation:
        sev_raw = (issue.get("severity") or "").lower()
        severity = self.SEVERITY_MAP.get(sev_raw, default)
        return Violation(
            checker=self.name,
            assertion=str(issue.get("issue_type") or "semantic_issue"),
            chain_id=chain_id,
            edge=issue.get("location") if isinstance(issue.get("location"), list) else None,
            severity=severity,
            detail=str(issue.get("description") or ""),
            suggested_action=str(issue.get("suggested_fix") or "") or None,
            meta={"raw": issue},
        )
