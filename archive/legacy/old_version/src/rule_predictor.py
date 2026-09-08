from __future__ import annotations

import ast
import datetime
import glob
import os
import sys
import textwrap
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Ensure project root is on the path so curriculumagent and src/ imports work
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
for _p in [_ROOT_DIR, _SRC_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Project imports
try:
    from config import CFG
    from utils import compute_grid_stats
    from training_enn import get_uncertainty
    from collect_data import get_features_with_history
except ImportError:
    CFG = None
    compute_grid_stats = None
    get_uncertainty = None
    get_features_with_history = None


# ─────────────────────────────────────────────────────────────────────────────
# Feature descriptions — plain English strings for console output
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_TEXT: Dict[str, str] = {
    "sum_load_p":            "total active power load at t",
    "sum_load_q":            "total reactive power load at t",
    "sum_gen_p":             "total active power generation at t",
    "var_line_rho":          "variance of line loading (rho) at t",
    "avg_line_rho":          "average line loading (rho) at t",
    "max_line_rho":          "maximum line loading (rho) at t",
    "nb_rho_ge_0.95":        "number of lines with rho >= 0.95 at t",
    "aleatoric_load_p_mean": "mean aleatoric uncertainty for active load",
    "aleatoric_load_q_mean": "mean aleatoric uncertainty for reactive load",
    "aleatoric_gen_p_mean":  "mean aleatoric uncertainty for active generation",
    "load_gen_ratio":        "load-to-generation ratio at t",
    "epistemic_before":      "epistemic uncertainty at t",
    "epistemic_after":       "epistemic uncertainty at t+12",
    "fcast_sum_load_p":      "forecasted total active power load at t+12",
    "fcast_sum_load_q":      "forecasted total reactive power load at t+12",
    "fcast_sum_gen_p":       "forecasted total active power generation at t+12",
    "fcast_var_line_rho":    "forecasted variance of line loading (rho) at t+12",
    "fcast_avg_line_rho":    "forecasted average line loading (rho) at t+12",
    "fcast_max_line_rho":    "forecasted maximum line loading (rho) at t+12",
    "fcast_nb_rho_ge_0.95":  "forecasted number of lines with rho >= 0.95 at t+12",
}

# AST comparison operators mapped to plain text
OP_TEXT: Dict[type, str] = {
    ast.GtE:   ">=",
    ast.Gt:    ">",
    ast.LtE:   "<=",
    ast.Lt:    "<",
    ast.Eq:    "=",
    ast.NotEq: "!=",
}

# Physical units per feature (None = dimensionless)
FEATURE_UNITS: Dict[str, Optional[str]] = {
    "sum_load_p":       "MW",
    "sum_load_q":       "MVAR",
    "sum_gen_p":        "MW",
    "fcast_sum_load_p": "MW",
    "fcast_sum_load_q": "MVAR",
    "fcast_sum_gen_p":  "MW",
}


# ─────────────────────────────────────────────────────────────────────────────
# AST TRANSLATOR — Python rule -> plain English sentence
# ─────────────────────────────────────────────────────────────────────────────

def _extract_feature_name(node: ast.expr) -> str:
    """Extracts the feature name from a subscript node such as x["max_line_rho"]."""
    if isinstance(node, ast.Subscript):
        sl = node.slice
        if isinstance(sl, ast.Constant):
            return str(sl.value)
        if isinstance(sl, ast.Index):           # Python <= 3.8
            inner = sl.value                    # type: ignore[attr-defined]
            if isinstance(inner, ast.Constant):
                return str(inner.value)
            if isinstance(inner, ast.Str):
                return inner.s                  # type: ignore[attr-defined]
        if isinstance(sl, ast.Str):
            return sl.s                         # type: ignore[attr-defined]
    return "unknown_feature"


def _extract_numeric(node: ast.expr) -> float:
    """Extracts a numeric literal from an AST node."""
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Num):               # Python <= 3.7
        return float(node.n)                    # type: ignore[attr-defined]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_extract_numeric(node.operand)
    return float("nan")


def _format_value(val: float, unit: Optional[str]) -> str:
    """Formats a numeric threshold with its optional physical unit."""
    if unit:
        v = f"{val:.0f}" if abs(val) >= 10 else f"{val:.2f}"
        return f"{v} {unit}"
    return f"{val:.4g}"


def _parse_condition(node: ast.expr) -> str:
    """
    Converts a single AST condition node (Compare or BoolOp AND)
    into a plain English string.
    """
    if isinstance(node, ast.Compare):
        feat      = _extract_feature_name(node.left)
        op_text   = OP_TEXT.get(type(node.ops[0]), "?")
        val       = _extract_numeric(node.comparators[0])
        feat_desc = FEATURE_TEXT.get(feat, feat)
        unit      = FEATURE_UNITS.get(feat)
        return f"the {feat_desc} is {op_text} {_format_value(val, unit)}"

    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        parts = [_parse_condition(v) for v in node.values]
        if len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"
        return ", ".join(parts[:-1]) + f", and {parts[-1]}"

    return "unknown condition"


def _collect_paths(
    node: ast.stmt,
    current: List[str],
    paths: List[List[str]],
) -> None:
    """
    Traverses the AST depth-first. Every path that reaches 'return 1'
    is recorded in paths as a list of condition strings.
    """
    if isinstance(node, ast.If):
        cond = _parse_condition(node.test)
        for stmt in node.body:
            _collect_paths(stmt, current + [cond], paths)
        for stmt in node.orelse:
            _collect_paths(stmt, current, paths)
    elif isinstance(node, ast.Return):
        val = None
        if isinstance(node.value, ast.Constant):
            val = node.value.value
        elif isinstance(node.value, ast.Num):   # type: ignore[attr-defined]
            val = node.value.n                  # type: ignore[attr-defined]
        if val == 1:
            paths.append(list(current))


def translate_rule_to_sentence(rule_code: str, line_name: str) -> str:
    """
    Converts a Python rule function into a plain English sentence.

    Each distinct path that returns 1 becomes an "or if" clause.
    Multiple AND conditions within the same path are joined with "while ... and".

    Example output:
      "Following a contingency on line 41_48_131, the RL agent is predicted
       to fail to provide a recommendation that solves a congestion problem
       if the maximum line loading (rho) at t is >= 0.82, or if the
       forecasted maximum line loading (rho) at t+12 is >= 0.66 while the
       epistemic uncertainty at t is >= 0.77 and the forecasted total
       active power load at t+12 is <= 643 MW."
    """
    line_name = line_name.replace("line_", "")
    try:
        tree = ast.parse(textwrap.dedent(rule_code).strip())
    except SyntaxError as e:
        return f"[PARSE ERROR for line {line_name}]: {e}"

    paths: List[List[str]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "rule":
            for stmt in node.body:
                _collect_paths(stmt, [], paths)

    if not paths:
        return (
            f"Following a contingency on line {line_name}, "
            f"the rule never predicts a failure."
        )

    # Build one phrase per path, then join with "or if"
    path_phrases: List[str] = []
    for path in paths:
        if len(path) == 1:
            path_phrases.append(path[0])
        elif len(path) == 2:
            path_phrases.append(f"{path[0]} while {path[1]}")
        else:
            joined = f"{path[0]} while {path[1]}"
            for extra in path[2:]:
                joined += f" and {extra}"
            path_phrases.append(joined)

    if len(path_phrases) == 1:
        condition_str = f"if {path_phrases[0]}"
    else:
        condition_str = f"if {path_phrases[0]}"
        for phrase in path_phrases[1:]:
            condition_str += f", or if {phrase}"

    return (
        f"Following a contingency on line {line_name}, "
        f"the RL agent is predicted to fail to provide a recommendation "
        f"that solves a problem {condition_str}."
    )


def translate_rule_code(rule_code: str, line_name: str) -> str:
    """Standalone helper: Python rule code -> plain English sentence. No Grid2Op required."""
    return translate_rule_to_sentence(rule_code, line_name)


# ─────────────────────────────────────────────────────────────────────────────
# RULE LOADER & EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

def _load_rule_fn(rule_path: str) -> Optional[Any]:
    """Reads and compiles a rule file, returning the callable rule function."""
    try:
        with open(rule_path, "r", encoding="utf-8") as f:
            code = f.read()
        if "def rule" not in code:
            return None
        local_env: Dict[str, Any] = {}
        exec(textwrap.dedent(code), {}, local_env)  # noqa: S102
        return local_env.get("rule")
    except Exception as e:
        print(f"[WARN] rule_predictor: could not load {rule_path}: {e}")
        return None


def _load_rule_code(rule_path: str) -> str:
    """Returns the raw Python source of a rule file."""
    try:
        with open(rule_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _apply_rule(rule_fn: Any, features: Dict[str, float]) -> int:
    """Applies rule(x) to the feature dictionary. Returns 0 or 1."""
    try:
        import pandas as pd  # noqa: PLC0415
        result = rule_fn(pd.Series(features))
        return 1 if int(result) == 1 else 0
    except Exception as e:
        print(f"[WARN] rule_predictor: rule evaluation failed: {e}")
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# FORECAST PIPELINE — mirrors the forecast block in analyze_disconnection_effect
# ─────────────────────────────────────────────────────────────────────────────

def _run_forecast(
    obs: Any,
    observations_array: List[Any],
    model_predict: Any,
    model_aleatoric: Any,
    model_enn: Any,
    cfg: Any,
    get_features_with_history_fn: Callable,
    get_uncertainty_fn: Callable,
    compute_grid_stats_fn: Callable,
) -> Dict[str, Any]:
    """
    Replicates the forecast block from analyze_disconnection_effect:
      1. Epistemic uncertainty at t=0  (epistemic_before)
      2. Forecast load_p/q and gen_p at t+12  (model_predict + model_aleatoric)
      3. Real power flow at t+12 via obs._forecasted_inj + obs.simulate()
      4. Grid statistics at t+12 via compute_grid_stats
      5. Epistemic uncertainty at t+12  (epistemic_after)

    Returns a dict with all uncertainty values and fcast_grid_stats.
    """
    out: Dict[str, Any] = {
        "epistemic_before":      float("nan"),
        "epistemic_after":       float("nan"),
        "aleatoric_load_p_mean": float("nan"),
        "aleatoric_load_q_mean": float("nan"),
        "aleatoric_gen_p_mean":  float("nan"),
        "fcast_grid_stats":      {},
    }
    enn_input_dim = int(getattr(model_enn, "input_dim", cfg.ENN_INPUT_DIM))

    # 1. Epistemic uncertainty at t=0
    try:
        obs_vect = obs.to_vect()[:enn_input_dim].reshape(1, -1)
        out["epistemic_before"] = float(get_uncertainty_fn(model_enn, obs_vect))
    except Exception as e:
        print(f"[WARN] rule_predictor: epistemic_before failed: {e}")

    # 2. Forecast at t+12 + aleatoric uncertainty
    try:
        x_t12  = get_features_with_history_fn(observations_array, obs)
        y_pred = model_predict.predict([x_t12])[0]
        z_pred = model_aleatoric.predict([x_t12])[0]

        load_p_t12 = y_pred[:cfg.NO_LOADS]
        load_q_t12 = y_pred[cfg.NO_LOADS: cfg.NO_LOADS * 2]
        gen_p_t12  = np.clip(y_pred[cfg.NO_LOADS * 2:], cfg.GEN_MIN, cfg.GEN_MAX)

        y_sigma = np.sqrt(np.expm1(np.clip(z_pred, 0.0, None)) + 1e-12)
        out["aleatoric_load_p_mean"] = float(np.mean(y_sigma[:cfg.NO_LOADS]))
        out["aleatoric_load_q_mean"] = float(np.mean(y_sigma[cfg.NO_LOADS: cfg.NO_LOADS * 2]))
        out["aleatoric_gen_p_mean"]  = float(np.mean(y_sigma[cfg.NO_LOADS * 2:]))

        # 3. Real power flow at t+12 via obs.simulate()
        now_ts  = obs.get_time_stamp()
        next_ts = now_ts + datetime.timedelta(minutes=60)   # 12 steps x 5 min

        obs_copy = obs.copy()
        obs_copy._forecasted_inj = [
            (now_ts,  {"injection": {
                "load_p": obs.load_p, "load_q": obs.load_q,
                "prod_p": obs.gen_p,  "prod_v": obs.gen_v,
            }}),
            (next_ts, {"injection": {
                "load_p": load_p_t12, "load_q": load_q_t12,
                "prod_p": gen_p_t12,  "prod_v": obs.gen_v,
            }}),
        ]

        sim_obs, _, _, _ = obs_copy.simulate(obs._obs_env._helper_action_env({}))

        # 4. Grid statistics at t+12
        out["fcast_grid_stats"] = compute_grid_stats_fn(sim_obs)

        # 5. Epistemic uncertainty at t+12
        sim_vect = sim_obs.to_vect()[:enn_input_dim].reshape(1, -1)
        out["epistemic_after"] = float(get_uncertainty_fn(model_enn, sim_vect))

    except Exception as e:
        print(f"[WARN] rule_predictor: forecast pipeline failed: {e}")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class RulePredictor:
    """
    Applies LLM symbolic rules to live Grid2Op observations.

    Loads all available rules from the specified temperature folder and,
    for a given observation and line name, runs the full forecast pipeline
    internally to extract all features before applying the rule.

    Parameters
    ----------
    rules_dir : str
        Directory containing line_*/best_rule.py subfolders.
        Example: "llm_rules_results/temp_0.5"

    model_predict : sklearn estimator
        Mean forecaster for load and generation (HBGB_36.pkl).

    model_aleatoric : sklearn estimator
        Aleatoric uncertainty model (HBGB_36_aleatoric.pkl).

    model_enn : torch.nn.Module
        Evidential Neural Network for epistemic uncertainty (enn_36.pth).

    cfg : config object, optional
        Project configuration (src/config.py). Defaults to the imported CFG.

    compute_grid_stats_fn : callable, optional
        compute_grid_stats from src/utils.py. Defaults to the imported function.

    get_uncertainty_fn : callable, optional
        get_uncertainty from src/training_enn.py. Defaults to the imported function.

    get_features_with_history_fn : callable, optional
        get_features_with_history from src/collect_data.py. Defaults to the imported function.

    observations_array : list, optional
        Accumulated observations for the current episode. Can be the same list
        object used by the caller — it is updated automatically between calls.
    """

    def __init__(
        self,
        rules_dir: str,
        model_predict: Optional[Any] = None,
        model_aleatoric: Optional[Any] = None,
        model_enn: Optional[Any] = None,
        cfg: Optional[Any] = None,
        compute_grid_stats_fn: Optional[Callable] = None,
        get_uncertainty_fn: Optional[Callable] = None,
        get_features_with_history_fn: Optional[Callable] = None,
        observations_array: Optional[List[Any]] = None,
    ):
        self.rules_dir        = rules_dir
        self.model_predict    = model_predict
        self.model_aleatoric  = model_aleatoric
        self.model_enn        = model_enn

        # Fall back to project-level imports if not provided explicitly
        self._cfg                        = cfg or CFG
        self._compute_grid_stats         = compute_grid_stats_fn or compute_grid_stats
        self._get_uncertainty            = get_uncertainty_fn or get_uncertainty
        self._get_features_with_history  = get_features_with_history_fn or get_features_with_history

        # Shared reference to the caller's observation list — stays in sync automatically
        self.observations_array: List[Any] = observations_array if observations_array is not None else []

        self._rules: Dict[str, Any] = {}
        self._codes: Dict[str, str] = {}
        self._load_all_rules()

        if not self._has_forecast_models:
            print(
                "[WARN] rule_predictor: forecast models not provided — "
                "fcast_* and epistemic_* features will be NaN. "
                "Pass model_predict, model_aleatoric and model_enn to RulePredictor()."
            )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def _has_forecast_models(self) -> bool:
        """True when all components required to run the forecast are available."""
        return all([
            self.model_predict               is not None,
            self.model_aleatoric             is not None,
            self.model_enn                   is not None,
            self._cfg                        is not None,
            self._compute_grid_stats         is not None,
            self._get_uncertainty            is not None,
            self._get_features_with_history  is not None,
        ])

    @property
    def available_lines(self) -> List[str]:
        """Sorted list of line names for which a rule has been loaded."""
        return sorted(self._rules.keys())

    # ── Rule loading ──────────────────────────────────────────────────────────

    def _load_all_rules(self) -> None:
        """Scans rules_dir for best_rule.py files and loads each one."""
        pattern = os.path.join(self.rules_dir, "line_*", "best_rule.py")
        files   = sorted(glob.glob(pattern))
        if not files:
            print(f"[WARN] rule_predictor: no rules found in '{self.rules_dir}'")
            return
        for path in files:
            line_name = os.path.basename(os.path.dirname(path)).replace("line_", "")
            fn   = _load_rule_fn(path)
            code = _load_rule_code(path)
            if fn is not None:
                self._rules[line_name] = fn
                self._codes[line_name] = code
        print(
            f"[INFO] rule_predictor: loaded {len(self._rules)} rules "
            f"from '{self.rules_dir}': {self.available_lines}"
        )

    # ── Main API ──────────────────────────────────────────────────────────────

    def predict(self, obs: Any, line_name: str) -> Dict[str, Any]:
        """
        Receives a live Grid2Op observation and a line name, runs the forecast
        pipeline internally, extracts all features, and applies the symbolic rule.

        Parameters
        ----------
        obs : Grid2Op observation
            Current observation at timestep t.
        line_name : str
            Target line name, e.g. "41_48_131" or "line_41_48_131".

        Returns
        -------
        dict with:
            "line_name"  : str
            "prediction" : int   (0 = no failure predicted, 1 = failure predicted)
            "features"   : dict  (all extracted feature values)
            "sentence"   : str   (LaTeX-ready explanation for the paper)
            "rule_code"  : str   (Python source of the applied rule)
            "has_rule"   : bool
        """
        line_name = line_name.replace("line_", "")

        # Grid statistics at t=0
        gs = self._compute_grid_stats(obs) if self._compute_grid_stats else {}
        sum_load_p = float(gs.get("sum_load_p", np.sum(obs.load_p) if hasattr(obs, "load_p") else float("nan")))
        sum_gen_p  = float(gs.get("sum_gen_p",  np.sum(obs.gen_p)  if hasattr(obs, "gen_p")  else float("nan")))

        features: Dict[str, float] = {
            "sum_load_p":            sum_load_p,
            "sum_load_q":            float(gs.get("sum_load_q",      float("nan"))),
            "sum_gen_p":             sum_gen_p,
            "var_line_rho":          float(gs.get("var_line_rho",    float("nan"))),
            "avg_line_rho":          float(gs.get("avg_line_rho",    float("nan"))),
            "max_line_rho":          float(gs.get("max_line_rho",    float("nan"))),
            "nb_rho_ge_0.95":        float(gs.get("nb_rho_ge_0.95", float("nan"))),
            "load_gen_ratio":        sum_load_p / (sum_gen_p + 1e-6),
            # Initialised to NaN — filled in by the forecast pipeline below
            "aleatoric_load_p_mean": float("nan"),
            "aleatoric_load_q_mean": float("nan"),
            "aleatoric_gen_p_mean":  float("nan"),
            "epistemic_before":      float("nan"),
            "epistemic_after":       float("nan"),
            "fcast_sum_load_p":      float("nan"),
            "fcast_sum_load_q":      float("nan"),
            "fcast_sum_gen_p":       float("nan"),
            "fcast_var_line_rho":    float("nan"),
            "fcast_avg_line_rho":    float("nan"),
            "fcast_max_line_rho":    float("nan"),
            "fcast_nb_rho_ge_0.95":  float("nan"),
        }

        # Run the forecast pipeline to obtain t+12 features
        if self._has_forecast_models:
            fc  = _run_forecast(
                obs=obs,
                observations_array=self.observations_array,
                model_predict=self.model_predict,
                model_aleatoric=self.model_aleatoric,
                model_enn=self.model_enn,
                cfg=self._cfg,
                get_features_with_history_fn=self._get_features_with_history,
                get_uncertainty_fn=self._get_uncertainty,
                compute_grid_stats_fn=self._compute_grid_stats,
            )
            fgs = fc["fcast_grid_stats"]
            features.update({
                "epistemic_before":      fc["epistemic_before"],
                "epistemic_after":       fc["epistemic_after"],
                "aleatoric_load_p_mean": fc["aleatoric_load_p_mean"],
                "aleatoric_load_q_mean": fc["aleatoric_load_q_mean"],
                "aleatoric_gen_p_mean":  fc["aleatoric_gen_p_mean"],
                "fcast_sum_load_p":      float(fgs.get("sum_load_p",      float("nan"))),
                "fcast_sum_load_q":      float(fgs.get("sum_load_q",      float("nan"))),
                "fcast_sum_gen_p":       float(fgs.get("sum_gen_p",       float("nan"))),
                "fcast_var_line_rho":    float(fgs.get("var_line_rho",    float("nan"))),
                "fcast_avg_line_rho":    float(fgs.get("avg_line_rho",    float("nan"))),
                "fcast_max_line_rho":    float(fgs.get("max_line_rho",    float("nan"))),
                "fcast_nb_rho_ge_0.95":  float(fgs.get("nb_rho_ge_0.95", float("nan"))),
            })
        else:
            pass  # Warning already shown in __init__

        return self._build_result(line_name, features)

    def sentence_only(self, line_name: str) -> str:
        """Returns the translated rule sentence without running any observation."""
        line_name = line_name.replace("line_", "")
        if line_name not in self._codes:
            return f"No rule available for line {line_name}."
        return translate_rule_to_sentence(self._codes[line_name], line_name)

    def all_sentences(self) -> Dict[str, str]:
        """
        Returns a dict of {line_name: sentence} for all loaded rules.
        Useful for generating the rule table in the paper.
        """
        return {ln: self.sentence_only(ln) for ln in self.available_lines}

    # ── Internal helper ───────────────────────────────────────────────────────

    def _build_result(self, line_name: str, features: Dict[str, float]) -> Dict[str, Any]:
        """Assembles the result dict from a feature set and a line name."""
        if line_name not in self._rules:
            return {
                "line_name":  line_name,
                "prediction": 0,
                "features":   features,
                "sentence":   f"No rule available for line {line_name}.",
                "rule_code":  "",
                "has_rule":   False,
            }
        prediction = _apply_rule(self._rules[line_name], features)
        sentence   = translate_rule_to_sentence(self._codes[line_name], line_name)
        return {
            "line_name":  line_name,
            "prediction": prediction,
            "features":   features,
            "sentence":   sentence,
            "rule_code":  self._codes[line_name],
            "has_rule":   True,
        }
