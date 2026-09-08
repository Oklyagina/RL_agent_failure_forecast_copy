# Prompts Appendix

This document lists the exact prompts used by the dual-LLM framework that distils the
Histogram Gradient Boosting (HGB) failure predictor into compact, human-readable `if/else`
rules for power-grid failure prediction.

The framework uses three role-specific prompts, all queried on the **same base model**
(`gemma4-31b`): a **Generator** (proposes an improved rule), a **Critic** (returns targeted
feedback), and a **Syntax Repair** agent (fixes structurally invalid rules). Each rule is a
small decision tree written as nested `if/else`, ranked by its **F2-score** (recall-oriented).

The prompts are stored as templates in `prompts/` and filled at run time by `dual_llm.py`
through `{{token}}` substitution. The tables below describe each token.

---

## 1. Generator

The Generator is the optimization engine. Given the most relevant features for a line
(selected by SHAP on the HGB), class-separated statistics, and the cases the current rule
gets wrong, it proposes one improved rule that catches more failures (higher F2).

### Tokens

| Token | Meaning |
|-------|---------|
| `{{cur_f2}}`, `{{cur_ovr}}`, `{{cur_fa}}` | Metrics of the rule from the previous iteration (the current state). |
| `{{alerts}}` | Optional warning injected when the rule oscillates between missed failures and excessive false alarms. |
| `{{seed_section}}` | The best data-driven seed rules (computed before the first LLM call), shown as a baseline to beat. |
| `{{line_name}}` | Identifier of the disconnected transmission line under analysis. |
| `{{anchors_text}}` | The SHAP-selected features for this line, each with its failure/normal medians and a suggested threshold. |
| `{{feature_stats_text}}` | Min / mean / max of each selected feature in the current training window. |
| `{{worst_cases}}` | Misclassified rows (false negatives first), projected onto the selected features. |
| `{{feedback}}` | The Critic's feedback from the previous iteration. |
| `{{best_f2}}`, `{{best_rule}}` | The best F2 and the best rule found so far. |

### Template (`prompts/generator_prompt.md`)

```text
You are the GENERATOR in a dual-LLM framework that distils a teacher model (Histogram
Gradient Boosting, HGB) into a small, human-readable rule that predicts power-line failure
(1 = failure, 0 = no failure).

# How your rule works — it IS a small decision tree
A decision tree asks yes/no questions about feature values and follows branches to an
answer. Your rule is exactly that, written as nested `if/else`:
- Each `if x["feature"] >= threshold:` is one split (one question).
- To combine questions with AND (both must hold), nest the second `if` inside the first `if`.
- To combine questions with OR (either is enough), put the second `if` inside the `else`.
- Each path ends in `return 1` (failure) or `return 0` (no failure).
More conditions = a deeper tree = better separation, but keep it readable: 2 to 5 conditions.

OR example (fires if EITHER condition holds — good for catching more failures):
```python
def rule(x):
    if x["max_line_rho"] >= 0.95:
        return 1
    else:
        if x["avg_line_rho"] >= 0.80:
            return 1
        else:
            return 0
```

# Your goal: maximise the F2-score
F2 rewards catching real failures (recall) about 4x more than avoiding false alarms. When in
doubt, make the rule catch MORE failures: lower a threshold, or add an OR branch with another
useful feature. A few extra false alarms are an acceptable price for catching more failures.

Current rule: F2={{cur_f2}} | missed failures (OVR)={{cur_ovr}} | false alarms (FA)={{cur_fa}}
{{alerts}}{{seed_section}}
# Line: {{line_name}}

# Most useful features for THIS line (ranked by SHAP on the teacher, most important first)
{{anchors_text}}

# Feature value ranges (min / mean / max)
{{feature_stats_text}}

# Cases the current rule gets WRONG (fix these — missed failures first)
{{worst_cases}}

# Critic feedback
{{feedback}}

# Best rule so far (F2={{best_f2}})
{{best_rule}}

# Constraints (a violation makes the rule invalid)
- Nested if/else only (NO elif). Only the features listed above. No imports, no loops.
- Return only 0 or 1. Use 2 to 5 conditions.

# Output these three sections exactly
[Start of Justification]
One or two sentences: what you changed and why it should raise F2.
[End of Justification]
[Start of Changes]
One sentence naming the feature and the threshold you changed.
[End of Changes]
[Start of Rule]
```python
def rule(x):
    if x["max_line_rho"] >= 0.95:
        return 1
    else:
        return 0
```
[End of Rule]
```

---

## 2. Critic

The Critic supervises the search. It reads the current rule and its metrics and returns one
to three concrete, threshold-level suggestions to raise the F2-score, prioritizing the
recovery of missed failures.

### Tokens

| Token | Meaning |
|-------|---------|
| `{{current_rule}}` | The rule produced in the current iteration. |
| `{{cur_f2}}`, `{{cur_ovr}}`, `{{cur_fa}}` | Metrics of the current rule. |
| `{{oscillation_warning}}` | Warning string injected when oscillation is detected (forces a single small adjustment). |
| `{{anchors_summary}}` | Condensed list of the most discriminative features and their suggested thresholds. |
| `{{worst_cases}}` | Misclassified rows (false negatives first). |
| `{{best_f2}}`, `{{best_rule}}` | The best F2 and rule found so far. |

### Template (`prompts/critic_prompt.md`)

```text
You are the CRITIC in a dual-LLM framework. You review a small if/else failure-prediction
rule (a small decision tree) and tell the generator how to raise its F2-score.

# Remember how the rule works
Nested `if/else` = a small decision tree. AND = nest inside `if`; OR = nest inside `else`.
F2 rewards catching failures (recall) far more than avoiding false alarms, so raising F2
usually means catching MORE failures.

# Current rule
{{current_rule}}
Metrics: F2={{cur_f2}} | missed failures (OVR)={{cur_ovr}} | false alarms (FA)={{cur_fa}}
{{oscillation_warning}}
# Most useful features for this line (most important first)
{{anchors_summary}}

# Cases the rule gets wrong (missed failures first)
{{worst_cases}}

# Best rule so far (F2={{best_f2}})
{{best_rule}}

# How to advise the generator
- If F2 = 0: the rule never predicts failure. Tell it to lower thresholds toward the
  "failures median" so it starts catching failures.
- If it misses failures (OVR high): lower a threshold, or add an OR branch (`if ... return 1`
  inside the else) using the next useful feature.
- If F2 is already decent: suggest one small threshold change that catches one more of the
  missed-failure cases above.

Give 1 to 3 concrete suggestions. Each names a feature, its current threshold, and the new
value. Change at most 2 conditions at once.

# Quick validity check
No elif, no imports, returns only 0/1, uses only the listed features. Say PASS or FAIL.
```

---

## 3. Syntax Repair

A strictly syntactic failsafe. If the Generator returns a rule that violates the structural
constraints (imports, `elif`, non-binary return, unknown features, or invalid Python), the
exception is caught and injected here for a surgical correction. This is the `Repair` step in
Algorithm 1, inspired by LLM self-debugging.

### Tokens

| Token | Meaning |
|-------|---------|
| `{{error_msg}}` | The exact validation/compilation error raised for the rule. |
| `{{bad_output}}` | The invalid output produced by the Generator. |

### Template (`prompts/repair_prompt.md`)

```text
The following rule is not valid yet.

Error:
{{error_msg}}

Fix it and output ONLY valid Python wrapped in tags, exactly like this:

[Start of Rule]
```python
def rule(x):
    if x["max_line_rho"] >= 0.95:
        return 1
    else:
        return 0
```
[End of Rule]

Requirements:
- define exactly: def rule(x):
- return only 0 or 1
- no imports, no helper functions, no loops
- no elif (use nested if/else)
- use only the allowed grid features

Bad output:
{{bad_output}}
```
