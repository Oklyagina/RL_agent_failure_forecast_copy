# LLM prompts

The distillation workflow uses the same three LLM roles as the legacy code,
all backed by the one model configured in `llm_config.json`:

- `prompts/generator_prompt.md` asks for a compact `if/else` rule using the
  selected HGB features, class statistics, and current errors.
- `prompts/critic_prompt.md` requests focused feedback for improving fidelity,
  with extra weight on detecting failures.
- `prompts/repair_prompt.md` converts malformed output back into the restricted
  rule syntax.

Prompt files are ordinary Markdown templates. `dual_llm.py` fills their
`{{field}}` placeholders and sends each result through `LLMClient`. Generated
source is accepted only after `rule_predictor.validate_rule` confirms that it
contains one `rule(x)` function made from feature comparisons, `if/else`
branches, and binary returns.
