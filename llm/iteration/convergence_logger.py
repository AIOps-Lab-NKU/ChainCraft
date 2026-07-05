#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConvergenceLogger - M2 Convergence Trajectory Logger

Record critique -> violations -> refiner_actions -> quality score for each iteration round.
Persist to JSON for post-hoc analysis of how a chain was fixed step by step (or why it was not).

Key events:
    record_round(): Called once per round, stores all signals for that round
    record_decision(): Called when the controller makes accept/reject/stop decision
    save(path): Serialize the entire trajectory to JSON

Independent of IterationLogger, but complementary to it:
    - IterationLogger focuses on 'engineering info' (agent calls, raw evaluation/refine results)
    - ConvergenceLogger focuses on 'scientific signals' (critic output, penalty scores, convergence signals)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional


class ConvergenceLogger:

    def __init__(self, case_id: str, app: Optional[str] = None):
        self.case_id = case_id
        self.app = app
        self.start_ts = time.time()
        self.rounds: List[Dict[str, Any]] = []
        self.decisions: List[Dict[str, Any]] = []
        self.final: Optional[Dict[str, Any]] = None

    # -------- Write --------

    def record_round(
        self,
        iteration: int,
        chains: List[Dict[str, Any]],
        aggregated_critique: Any,            # AggregatedCritique
        quality_score: Optional[float] = None,
        refiner_actions: Optional[List[Dict[str, Any]]] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        critique_dict = (
            aggregated_critique.to_dict()
            if hasattr(aggregated_critique, "to_dict")
            else aggregated_critique
        )
        self.rounds.append({
            "iteration": iteration,
            "ts": time.time(),
            "n_chains": len(chains or []),
            "critique": critique_dict,
            "quality_score": quality_score,
            "refiner_actions": refiner_actions or [],
            "extras": extras or {},
        })

    def record_decision(
        self,
        iteration: int,
        decision: str,                       # accept_new / reject_new / stop_converged / stop_max_iter / stop_degraded
        reason: str,
        delta_score: Optional[float] = None,
    ) -> None:
        self.decisions.append({
            "iteration": iteration,
            "ts": time.time(),
            "decision": decision,
            "reason": reason,
            "delta_score": delta_score,
        })

    def set_final(
        self,
        converged: bool,
        stop_reason: str,
        n_iterations: int,
        final_quality: Optional[float] = None,
    ) -> None:
        self.final = {
            "converged": converged,
            "stop_reason": stop_reason,
            "n_iterations": n_iterations,
            "final_quality": final_quality,
            "elapsed_sec": time.time() - self.start_ts,
        }

    # -------- Read / Summary --------

    def summary(self) -> Dict[str, Any]:
        if not self.rounds:
            return {"n_rounds": 0}
        quality_trace = [r.get("quality_score") for r in self.rounds]
        violation_trace = [
            r["critique"].get("severity_counts", {})
            for r in self.rounds
        ]
        return {
            "n_rounds": len(self.rounds),
            "quality_trace": quality_trace,
            "violation_trace": violation_trace,
            "decisions": [d["decision"] for d in self.decisions],
            "final": self.final,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "app": self.app,
            "start_ts": self.start_ts,
            "rounds": self.rounds,
            "decisions": self.decisions,
            "final": self.final,
            "summary": self.summary(),
        }

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path
