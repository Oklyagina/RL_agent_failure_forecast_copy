Review this human-readable failure-prediction rule and give short, concrete
feedback for improving its F2 score without creating excessive false alarms.

Rule:
{{current_rule}}

Metrics: {{current_metrics}}

Relevant feature statistics:
{{feature_summary}}

Misclassified examples:
{{worst_cases}}

Best rule so far:
{{best_rule}}

Recommend at most two specific threshold or branch changes. Return feedback
only; do not return Python code.
