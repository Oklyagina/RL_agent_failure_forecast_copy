Repair the invalid rule below while preserving its intended decisions.

Validation error:
{{error_msg}}

Invalid output:
{{bad_output}}

Return exactly one function using nested if/else statements, numeric feature
comparisons, and `return 0` or `return 1`. No imports, loops, assignments, or
additional prose. Write every comparison as
`x["feature_name"]` against a numeric threshold, for example
`x["max_line_rho"] >= 0.9`; never compare `x` itself with a value.

[Start of Rule]
def rule(x):
    ...
[End of Rule]
