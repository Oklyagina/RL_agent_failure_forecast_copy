You are the generator in a rule-distillation workflow. Create one small Python
decision rule that imitates an HGB model predicting whether an RL grid agent
will fail after contingency {{line_name}}. Failure is 1; no failure is 0.

Maximize F2 agreement with HGB. Here, failure means the HGB teacher's target
is 1, not a separately observed blackout. Optimize
F2 = 5*TP / (5*TP + 4*FN + FP), using counts over the full evaluation population.
Recall matters, but lowering the missed-failure rate alone is not the objective.
The false-alarm rate is FP/(FP+TN); the missed-failure rate is FN/(FN+TP).
These rates have different denominators: when failures are rare, a small rise
in false-alarm rate can add many false positives and reduce F2.

Start from the best rule below and propose one or two localized changes.
Inspect both missed failures and false alarms. Tighten a broad failure branch
or add an AND guard when it catches many normal cases. Lower a threshold or
add an OR branch only when evidence supports recovering failures without
excessive new false positives. Threshold direction depends on the comparison.
SHAP rank is importance, not proof of the direction or quality of a threshold.
Examples are diagnostic samples, not representative class proportions.
Preserve useful branches. If no evidence supports an improvement, return the
best rule unchanged. Actual full-population F2 determines acceptance.

Current rule metrics: {{current_metrics}}

Evaluation population and current rule counts:
{{evaluation_context}}

Relevant features and thresholds:
{{feature_summary}}

Misclassified examples (missed failures first; both error types when available):
{{worst_cases}}

Previous critic feedback:
{{feedback}}

Best rule so far:
{{best_rule}}

Use only the listed features and numeric thresholds. The output must contain a
single `def rule(x):` function made only of nested if/else statements and
`return 0` or `return 1`. Write every comparison as
`x["feature_name"]` against a numeric threshold, for example
`x["max_line_rho"] >= 0.9`; never compare `x` itself with a value. Do not import
anything or use loops.

[Start of Rule]
def rule(x):
    ...
[End of Rule]
