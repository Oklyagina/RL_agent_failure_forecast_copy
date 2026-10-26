"""Load distilled failure rules and turn them into plain-English text.

The HGB classifier in :mod:`src.failure_forecast` remains the source of the
actual failure probability and binary prediction. Rules loaded here are a
human-readable surrogate produced by the optional LLM distillation workflow.
"""
from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


FEATURE_TEXT = {
    "sum_load_p": "total active power load at the current time",
    "sum_load_q": "total reactive power load at the current time",
    "sum_gen_p": "total active power generation at the current time",
    "var_line_rho": "variance of line loading at the current time",
    "avg_line_rho": "average line loading at the current time",
    "max_line_rho": "maximum line loading at the current time",
    "nb_rho_ge_0.95": "number of highly loaded lines at the current time",
    "load_gen_ratio": "load-to-generation ratio at the current time",
    "aleatoric_load_p_mean": "forecast uncertainty for active load",
    "aleatoric_load_q_mean": "forecast uncertainty for reactive load",
    "aleatoric_gen_p_mean": "forecast uncertainty for generation",
    "epistemic_before": "epistemic uncertainty at the current time",
    "epistemic_after": "epistemic uncertainty in the forecast state",
    "fcast_sum_load_p": "forecast total active power load one hour ahead",
    "fcast_sum_load_q": "forecast total reactive power load one hour ahead",
    "fcast_sum_gen_p": "forecast total active power generation one hour ahead",
    "fcast_var_line_rho": "forecast variance of line loading one hour ahead",
    "fcast_avg_line_rho": "forecast average line loading one hour ahead",
    "fcast_max_line_rho": "forecast maximum line loading one hour ahead",
    "fcast_nb_rho_ge_0.95": "forecast number of highly loaded lines one hour ahead",
}

FEATURE_UNITS = {
    "sum_load_p": "MW",
    "sum_load_q": "MVAR",
    "sum_gen_p": "MW",
    "fcast_sum_load_p": "MW",
    "fcast_sum_load_q": "MVAR",
    "fcast_sum_gen_p": "MW",
}

_OP_TEXT = {
    ast.GtE: ">=",
    ast.Gt: ">",
    ast.LtE: "<=",
    ast.Lt: "<",
    ast.Eq: "=",
    ast.NotEq: "!=",
}

_INVERTED_OP_TEXT = {
    ast.GtE: "<",
    ast.Gt: "<=",
    ast.LtE: ">",
    ast.Lt: ">=",
    ast.Eq: "!=",
    ast.NotEq: "=",
}

_ALLOWED_NODES = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.If,
    ast.Return,
    ast.Compare,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Subscript,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.UnaryOp,
    ast.USub,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.Eq,
    ast.NotEq,
)


def _feature_name(node: ast.expr) -> str:
    if not isinstance(node, ast.Subscript):
        raise ValueError("A rule condition must read a feature from x[...].")
    key = node.slice
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return key.value
    raise ValueError("A rule feature name must be a string literal.")


def _number(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_number(node.operand)
    raise ValueError("A rule threshold must be numeric.")


def _format_threshold(value: float, unit: str | None) -> str:
    rendered = f"{value:.0f}" if abs(value) >= 10 else f"{value:.4g}"
    return f"{rendered} {unit}" if unit else rendered


def _condition_text(node: ast.expr, *, negated: bool = False) -> str:
    if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
        feature = _feature_name(node.left)
        value = _number(node.comparators[0])
        operators = _INVERTED_OP_TEXT if negated else _OP_TEXT
        operator = operators.get(type(node.ops[0]))
        if operator is None:
            raise ValueError("Unsupported comparison operator in rule.")
        description = FEATURE_TEXT.get(feature, feature.replace("_", " "))
        threshold = _format_threshold(value, FEATURE_UNITS.get(feature))
        return f"the {description} is {operator} {threshold}"

    if isinstance(node, ast.BoolOp):
        # De Morgan's law keeps descriptions of else branches logically correct.
        joiner = " or " if isinstance(node.op, ast.Or) else " and "
        if negated:
            joiner = " and " if joiner == " or " else " or "
        parts = [_condition_text(value, negated=negated) for value in node.values]
        return "(" + joiner.join(parts) + ")"
    raise ValueError("Unsupported rule condition.")


def validate_rule(rule_code: str) -> ast.FunctionDef:
    """Validate the deliberately small rule language and return its function AST."""
    tree = ast.parse(textwrap.dedent(rule_code).strip())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "rule":
        raise ValueError("Rule source must contain exactly one function named 'rule'.")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Unsupported syntax in rule: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in {"x", "rule"}:
            raise ValueError(f"Unsupported name in rule: {node.id}")
        if isinstance(node, ast.Subscript):
            feature = _feature_name(node)
            if feature not in FEATURE_TEXT:
                raise ValueError(f"Unsupported feature in rule: {feature}")
    return functions[0]


def compile_rule(rule_code: str) -> Callable[[Mapping[str, float]], int]:
    """Compile validated rule source without exposing Python built-ins."""
    validate_rule(rule_code)
    namespace: dict[str, Any] = {}
    exec(
        compile(textwrap.dedent(rule_code), "<explainability-rule>", "exec"),
        {"__builtins__": {}},
        namespace,
    )
    return namespace["rule"]


def _failure_paths(statements: list[ast.stmt], conditions: list[str]) -> list[list[str]]:
    """Collect the conditions of branches that return failure (1)."""
    if not statements:
        return []
    first, rest = statements[0], statements[1:]
    if isinstance(first, ast.Return):
        value = _number(first.value) if first.value is not None else 0.0
        return [conditions] if value == 1 else []
    if not isinstance(first, ast.If):
        return _failure_paths(rest, conditions)

    condition = _condition_text(first.test)
    inverse = _condition_text(first.test, negated=True)
    true_paths = _failure_paths(list(first.body) + rest, conditions + [condition])
    false_paths = _failure_paths(list(first.orelse) + rest, conditions + [inverse])
    return true_paths + false_paths


def translate_rule_to_sentence(rule_code: str, line_name: str) -> str:
    """Describe all failure-producing branches of a distilled rule."""
    function = validate_rule(rule_code)
    paths = _failure_paths(list(function.body), [])
    line_name = str(line_name).replace("line_", "")
    if not paths:
        return f"The explanatory rule for contingency {line_name} never predicts failure."

    clauses = [" and ".join(path) if path else "all conditions" for path in paths]
    conditions = "; or when ".join(clauses)
    return (
        f"For a contingency on line {line_name}, the simplified explanatory rule "
        f"predicts that the RL agent may fail when {conditions}."
    )


def translate_rule_code(rule_code: str, line_name: str) -> str:
    """Backward-compatible name for translating one rule into text."""
    return translate_rule_to_sentence(rule_code, line_name)


@dataclass(frozen=True)
class RuleExplanation:
    """Result produced by a human-readable surrogate rule."""

    line: str
    prediction: int
    text: str


class RuleExplainer:
    """Load ``line_*/best_rule.py`` files and explain their failure branches."""

    def __init__(self, rules_dir: str | Path):
        self.rules_dir = Path(rules_dir)
        self._codes: dict[str, str] = {}
        self._rules: dict[str, Callable[[Mapping[str, float]], int]] = {}
        for path in sorted(self.rules_dir.glob("line_*/best_rule.py")):
            line = path.parent.name.removeprefix("line_")
            code = path.read_text(encoding="utf-8")
            self._rules[line] = compile_rule(code)
            self._codes[line] = code

    @property
    def available_lines(self) -> list[str]:
        return sorted(self._rules)

    def describe(self, line: str) -> str:
        """Return the static model description for one contingency."""
        name = str(line).removeprefix("line_")
        if name not in self._codes:
            raise KeyError(f"No explanatory rule is available for line {name!r}.")
        return translate_rule_to_sentence(self._codes[name], name)

    def explain(self, line: str, features: Mapping[str, float]) -> RuleExplanation:
        """Evaluate and describe the surrogate rule for one feature row."""
        name = str(line).removeprefix("line_")
        if name not in self._rules:
            raise KeyError(f"No explanatory rule is available for line {name!r}.")
        prediction = int(self._rules[name](features))
        return RuleExplanation(name, prediction, self.describe(name))

    def all_descriptions(self) -> dict[str, str]:
        """Return model descriptions for every loaded contingency rule."""
        return {line: self.describe(line) for line in self.available_lines}
