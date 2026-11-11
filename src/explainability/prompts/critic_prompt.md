Review this human-readable failure-prediction rule and give short, concrete
feedback for improving its F2 score without creating excessive false alarms.

Maximize F2 agreement with HGB using
F2 = 5*TP / (5*TP + 4*FN + FP), evaluated on the full population.
A missed failure has HGB target 1 and rule prediction 0. A false alarm has
HGB target 0 and rule prediction 1. Their rates have different denominators.
When failures are rare, a small increase in false-alarm rate can add many false
positives. Lower missed-failure rate does not necessarily mean higher F2.
Examples are diagnostic samples, not representative class proportions.

Rule:
{{current_rule}}

Metrics: {{current_metrics}}

Evaluation population and candidate counts:
{{evaluation_context}}

Relevant feature statistics:
{{feature_summary}}

Misclassified examples (missed failures first; both error types when available):
{{worst_cases}}

Best rule so far:
{{best_rule}}

Best rule metrics: {{best_metrics}}

Compare candidate F2 with best-rule F2 before advising. If the candidate is
worse, identify the regression and direct the generator back to the best rule.
Recommend at most two localized threshold or branch changes to that best rule,
grounded in the supplied statistics and errors. Explain which false positives
or false negatives each change targets and the possible cost to the other class.
For broad failure branches, consider tightening thresholds or an AND guard.
For missed failures, consider a targeted OR branch only with evidence it will
separate failures from normal cases. Do not always recommend lower thresholds.
Do not invent measured gains for an unevaluated change. If no improvement is
supported, recommend keeping the best rule. Return feedback only, no Python.
