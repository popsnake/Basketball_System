from __future__ import annotations

from functools import lru_cache

from agents.baseline.pipeline import BaselineAgentService
from agents.upgraded.graph import UpgradedAgentService
from models_dl.inference import ScoringEngine


@lru_cache
def get_scoring_engine() -> ScoringEngine:
    return ScoringEngine()


@lru_cache
def get_baseline_agent_service() -> BaselineAgentService:
    return BaselineAgentService()


@lru_cache
def get_upgraded_agent_service() -> UpgradedAgentService:
    return UpgradedAgentService()
