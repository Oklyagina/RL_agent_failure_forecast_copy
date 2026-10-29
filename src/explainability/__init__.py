"""Optional LLM-based explainability for failure-forecast predictions."""

from src.explainability.llm_client import LLMClient, LLMConfig, LLMUnavailableError
from src.explainability.orchestrator import ExplainabilityOrchestrator
from src.explainability.rule_predictor import (
    RuleExplainer,
    RuleExplanation,
    compile_rule,
    translate_rule_code,
    translate_rule_to_sentence,
)

__all__ = [
    "ExplainabilityOrchestrator",
    "LLMClient",
    "LLMConfig",
    "LLMUnavailableError",
    "RuleExplainer",
    "RuleExplanation",
    "compile_rule",
    "translate_rule_code",
    "translate_rule_to_sentence",
]
