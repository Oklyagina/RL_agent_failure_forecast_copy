You are the generator in a rule-distillation workflow. Create one small Python
decision rule that imitates an HGB model predicting whether an RL grid agent
will fail after contingency {{line_name}}. Failure is 1; no failure is 0.

Current rule metrics: {{current_metrics}}

Relevant features and thresholds:
{{feature_summary}}

Misclassified examples:
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
