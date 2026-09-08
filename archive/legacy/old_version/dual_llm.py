"""
dual_llm.py
===========
Dual-LLM extraction of interpretable failure-prediction rules for power grids.

This module distils a "black-box" Histogram Gradient Boosting (HGB) failure predictor into
a compact, human-readable `if/else` rule for each disconnected transmission line. It is the
implementation behind Section IV-B ("Generation of Interpretable Rules") and Table III of the
paper.

Pipeline, per transmission line
-------------------------------
1. Build the teacher target: label every sample with the HGB classifier's prediction.
2. Sequential train/test split (with adaptive / leave-one-out fallbacks for sparse lines).
3. SHAP feature selection: rank the input features by their SHAP importance on the HGB and
   keep only the top features for this line (see compute_shap_rankings.py, which writes
   shap_rankings.json). A small per-line sweep k in {3,...,8} chooses the subset size.
4. Data-driven seed rules: from class-separated statistics, build single-feature, AND, and
   OR candidate rules over the selected features and keep the best (no LLM calls, fast).
5. Generator-Critic-Repair refinement loop: an LLM proposes an improved rule, a Repair LLM
   fixes any syntax/structure violations, and a Critic LLM returns targeted feedback. The
   three roles are the SAME base model queried with different prompts (loaded from prompts/).
6. Scoring and selection: candidates are ranked directly by F2-score (recall-oriented), with
   a guard against degenerate "predict-everything" rules. Rules are SELECTED on the training
   set and the finally retained rule is REPORTED on the held-out test set, so the numbers are
   directly comparable to the DT and HGB hold-out results.

The whole procedure is repeated over several random seeds; Table III reports the mean and
standard deviation of the pooled (micro-averaged) test metrics over those seeds.

Run order
---------
  1) python compute_shap_rankings.py        # once -> shap_rankings.json
  2) python dual_llm.py                      # runs all seeds, writes results/ and summary

Requires: numpy, pandas, scikit-learn, joblib, requests. The LLM endpoint is configured in
llm_config_inesctec.json (api_url, api_key, model).
"""

import os
import re
import json
import math
import time
import random
import traceback
from typing import Dict, List, Optional, Any

import joblib
import numpy as np
import pandas as pd
import requests

from sklearn.metrics import accuracy_score, fbeta_score, confusion_matrix

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# =========================================================================== #
#  CONFIGURATION
# =========================================================================== #

HERE = os.path.dirname(os.path.abspath(__file__))

# --- inputs ---------------------------------------------------------------- #
DATA_PATH = os.path.join(HERE, "uncertainty_disconnection_analysis.csv")  # feature table
HGB_MODEL_PATH = os.path.join(HERE, "classifier_full_uncertainty.pkl")    # teacher classifier
LLM_CONFIG_FILE = os.path.join(HERE, "llm_config_inesctec.json")          # {api_url, api_key, model}
SHAP_RANKINGS_FILE = os.path.join(HERE, "shap_rankings.json")             # from compute_shap_rankings.py

PROMPTS_DIR = os.path.join(HERE, "prompts")
GENERATOR_PROMPT_FILE = os.path.join(PROMPTS_DIR, "generator_prompt.md")
CRITIC_PROMPT_FILE = os.path.join(PROMPTS_DIR, "critic_prompt.md")
REPAIR_PROMPT_FILE = os.path.join(PROMPTS_DIR, "repair_prompt.md")

# --- outputs --------------------------------------------------------------- #
OUTPUT_DIR = os.path.join(HERE, "results")

# --- experiment settings --------------------------------------------------- #
SEEDS = [42, 7, 13, 21, 100, 123, 2024, 555, 999, 31415]  # 10 random seeds for mean +/- std
TEMPERATURE = 0.7        # fixed LLM sampling temperature (tau in the paper)
ITERATIONS = 20          # Generator-Critic iterations per line
K_VALUES = (3, 4, 5, 6, 7, 8)  # per-line top-k feature-subset sweep

# --- per-line split thresholds (handle the rarer contingencies gracefully) - #
MIN_POS_TRAIN = 10       # standard split needs at least this many failures in train
MIN_POS_TEST = 5         # ... and this many in test
MIN_TOTAL_SAMPLES = 20
MIN_POS_STANDARD = MIN_POS_TRAIN + MIN_POS_TEST   # >= this -> standard sequential split
MIN_POS_LOO = 4          # < this many failures -> leave-one-out over the failure samples
MIN_TOTAL_SAMPLES_SPARSE = 6

# --- balanced window for the generator prompt ------------------------------ #
WINDOW_SIZE = 120
MAX_POSITIVE_IN_WINDOW = 40
NEGATIVE_RATIO = 2.0     # 2 normal samples per failure sample

# --- prompt-size caps ------------------------------------------------------ #
MAX_WORST_CASES = 8
MAX_RULE_CHARS = 12000
MAX_FEEDBACK_CHARS = 3000
MAX_TEXT_CHARS = 2000

# --- LLM call -------------------------------------------------------------- #
API_TIMEOUT = 180
API_RETRIES = 3
SLEEP_BETWEEN_RETRIES = 3

# --- features -------------------------------------------------------------- #
# Full feature vector seen by the HGB teacher (21 features). `line_id_encoded` is the
# categorical line identifier; it is NOT exposed to the rules (the rules are per-line).
FEATURES = [
    "line_id_encoded",
    "sum_load_p", "sum_load_q", "sum_gen_p",
    "var_line_rho", "avg_line_rho", "max_line_rho", "nb_rho_ge_0.95",
    "aleatoric_load_p_mean", "aleatoric_load_q_mean", "aleatoric_gen_p_mean",
    "load_gen_ratio",
    "epistemic_before", "epistemic_after",
    "fcast_sum_load_p", "fcast_sum_load_q", "fcast_sum_gen_p",
    "fcast_var_line_rho", "fcast_avg_line_rho", "fcast_max_line_rho",
    "fcast_nb_rho_ge_0.95",
]
FEATURES_FOR_LLM = [f for f in FEATURES if f != "line_id_encoded"]  # features the rules may use

# Short descriptions shown in the generator prompt.
FEATURE_DESCRIPTIONS = {
    "sum_load_p": "Total active load.", "sum_load_q": "Total reactive load.",
    "sum_gen_p": "Total active generation.", "var_line_rho": "Variance of line loading rho.",
    "avg_line_rho": "Average line loading rho.", "max_line_rho": "Maximum line loading rho.",
    "nb_rho_ge_0.95": "Number of lines with rho >= 0.95.",
    "aleatoric_load_p_mean": "Mean aleatoric uncertainty, active load.",
    "aleatoric_load_q_mean": "Mean aleatoric uncertainty, reactive load.",
    "aleatoric_gen_p_mean": "Mean aleatoric uncertainty, generation.",
    "load_gen_ratio": "Load / generation ratio.",
    "epistemic_before": "Epistemic uncertainty at t.",
    "epistemic_after": "Epistemic uncertainty at t+12 (pre-disconnection).",
    "fcast_sum_load_p": "Forecast total active load at t+12.",
    "fcast_sum_load_q": "Forecast total reactive load at t+12.",
    "fcast_sum_gen_p": "Forecast total active generation at t+12.",
    "fcast_var_line_rho": "Forecast variance of rho at t+12.",
    "fcast_avg_line_rho": "Forecast average rho at t+12.",
    "fcast_max_line_rho": "Forecast maximum rho at t+12.",
    "fcast_nb_rho_ge_0.95": "Forecast number of lines with rho >= 0.95 at t+12.",
}

# The 10 contingencies (disconnected lines) and their integer ids for the teacher.
LINE_MAP: Dict[str, int] = {
    "34_35_110": 0, "39_41_121": 1, "41_48_131": 2, "43_44_125": 3, "44_45_126": 4,
    "48_50_136": 5, "48_53_141": 6, "54_58_154": 7, "62_58_180": 8, "62_63_160": 9,
}


# =========================================================================== #
#  SMALL HELPERS
# =========================================================================== #

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def safe_float(x):
    try:
        return float("nan") if x is None else float(x)
    except Exception:
        return float("nan")


def clip_text(t, n):
    return "" if t is None else str(t)[:n]


def to_json(d):
    return json.dumps(d, ensure_ascii=False, indent=2, default=str)


# Prompt templates are cached after first read and filled with {{token}} substitution.
_PROMPT_CACHE: Dict[str, str] = {}


def load_prompt(path):
    if path not in _PROMPT_CACHE:
        with open(path, "r", encoding="utf-8") as f:
            _PROMPT_CACHE[path] = f.read()
    return _PROMPT_CACHE[path]


def fill_template(template, fields):
    out = template
    for k, v in fields.items():
        out = out.replace("{{" + k + "}}", str(v))
    return out.strip()


def load_llm_config():
    with open(LLM_CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_shap_rankings():
    """Per-line feature ranking produced by compute_shap_rankings.py. If missing, the seed
    phase falls back to ranking features by class-separated median difference."""
    if os.path.exists(SHAP_RANKINGS_FILE):
        with open(SHAP_RANKINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    print("[warn] shap_rankings.json not found - using median-difference feature order")
    return {}


SHAP_RANKINGS = load_shap_rankings()


# =========================================================================== #
#  METRICS AND SCORING
# =========================================================================== #

def compute_metrics(y_true, y_pred):
    """Accuracy, F2-score, false-alarm rate (FA) and oversight/miss rate (OVR)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "fa": fp / (fp + tn + 1e-12),      # FA = FP / (TN + FP)
        "ovr": fn / (fn + tp + 1e-12),     # OVR = FN / (TP + FN)
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def score_rule(m):
    """Rank rules directly by F2-score (recall-oriented), with two guards:
    - a rule that never predicts failure (F2 = 0) is worthless;
    - a rule that fires on almost everything (FA >= 0.90) is degenerate."""
    if m["f2"] == 0.0:
        return -1.0
    if m["fa"] >= 0.90:
        return -0.90
    return m["f2"]


# =========================================================================== #
#  DATA PREPARATION
# =========================================================================== #

def prepare_dataframe():
    """Load the feature table, encode the line id, and attach the teacher (HGB) label."""
    df = pd.read_csv(DATA_PATH)
    # load-to-generation ratio (guard against division by zero)
    df["load_gen_ratio"] = df.apply(
        lambda r: (r["sum_load_p"] / r["sum_gen_p"]) if r.get("sum_gen_p", 0) != 0 else 0.0, axis=1)

    df["line_id_encoded"] = df["line_disconnected"].map(LINE_MAP)
    if df["line_id_encoded"].isna().any():
        unknown = sorted(df.loc[df["line_id_encoded"].isna(), "line_disconnected"].dropna().unique())
        raise ValueError(f"line_disconnected values not in LINE_MAP: {unknown}")
    df["line_id_encoded"] = df["line_id_encoded"].astype(int)

    # the rules imitate the HGB teacher, so the target is the HGB prediction
    model = joblib.load(HGB_MODEL_PATH)
    df["target"] = model.predict(df[FEATURES].copy()).astype(int)
    return df


# =========================================================================== #
#  TRAIN / TEST SPLITS  (per line)
# =========================================================================== #
# Lines differ a lot in how many failures they contain, so the split strategy adapts:
#   - standard : enough failures -> a single sequential split
#   - adaptive : few failures   -> a looser sequential split that still leaves >=1 in test
#   - loo      : very few        -> leave-one-out over the failure samples

def sequential_split(df_line):
    df_line = df_line.sort_index().reset_index(drop=True)
    if len(df_line) < MIN_TOTAL_SAMPLES:
        raise RuntimeError(f"not enough samples: {len(df_line)}")
    for frac in np.linspace(0.60, 0.85, 11):
        cut = int(len(df_line) * frac)
        tr, te = df_line.iloc[:cut].copy(), df_line.iloc[cut:].copy()
        if int(tr["target"].sum()) >= MIN_POS_TRAIN and int(te["target"].sum()) >= MIN_POS_TEST:
            return tr, te
    raise RuntimeError("no valid sequential split")


def adaptive_split(df_line):
    df_line = df_line.sort_index().reset_index(drop=True)
    n_pos = int(df_line["target"].sum())
    if len(df_line) < MIN_TOTAL_SAMPLES_SPARSE:
        raise RuntimeError(f"sparse line too small: {len(df_line)}")
    for frac in np.linspace(0.50, 0.90, 17):
        cut = int(len(df_line) * frac)
        tr, te = df_line.iloc[:cut].copy(), df_line.iloc[cut:].copy()
        if int(tr["target"].sum()) >= max(2, n_pos - 2) and int(te["target"].sum()) >= 1:
            return tr, te
    pos = df_line.index[df_line["target"] == 1].tolist()
    if len(pos) >= 2:
        tr, te = df_line.iloc[:pos[-1]].copy(), df_line.iloc[pos[-1]:].copy()
        if int(tr["target"].sum()) >= 1 and int(te["target"].sum()) >= 1:
            return tr, te
    raise RuntimeError("adaptive split failed")


def leave_one_out_splits(df_line):
    df_line = df_line.sort_index().reset_index(drop=True)
    folds = []
    for idx in df_line.index[df_line["target"] == 1].tolist():
        te = df_line.loc[[idx]].copy()
        tr = df_line.drop(index=idx).copy()
        if len(tr) > 0:
            folds.append((tr, te))
    if not folds:
        raise RuntimeError("no LOO folds")
    return folds


# =========================================================================== #
#  WINDOW AND CLASS-SEPARATED STATISTICS
# =========================================================================== #

def build_balanced_window(train_df):
    """A balanced (2:1 normal:failure) sample of the training data, shown to the generator."""
    pos = train_df[train_df["target"] == 1]
    neg = train_df[train_df["target"] == 0]
    if len(pos) == 0:
        return train_df.head(min(WINDOW_SIZE, len(train_df))).copy()
    pos_n = min(len(pos), MAX_POSITIVE_IN_WINDOW)
    pos_s = pos.sample(n=pos_n, random_state=random.randint(0, 1_000_000))
    neg_n = min(len(neg), max(1, int(math.ceil(pos_n * NEGATIVE_RATIO))))
    neg_s = neg.sample(n=neg_n, random_state=random.randint(0, 1_000_000)) if neg_n else neg.head(0)
    window = pd.concat([pos_s, neg_s], axis=0)
    fill = max(0, WINDOW_SIZE - len(window))
    if fill > 0:
        rest = train_df.drop(index=window.index, errors="ignore")
        if len(rest) > 0:
            window = pd.concat([window, rest.sample(n=min(len(rest), fill),
                                                    random_state=random.randint(0, 1_000_000))])
    return window.sort_index().copy()


def feature_stats(window, cols):
    out = {}
    for c in cols:
        s = pd.to_numeric(window[c], errors="coerce")
        out[c] = {"min": safe_float(s.min()), "mean": safe_float(s.mean()), "max": safe_float(s.max())}
    return out


def compute_anchors(train_df, cols):
    """Per-feature class-separated medians/quartiles and a suggested threshold (midpoint of
    the failure and normal medians). Features with no class separation are dropped. Returned
    ordered by absolute median difference (most discriminative first)."""
    anchors = {}
    pos = train_df[train_df["target"] == 1]
    neg = train_df[train_df["target"] == 0]
    for f in cols:
        sp = pd.to_numeric(pos[f], errors="coerce").dropna()
        sn = pd.to_numeric(neg[f], errors="coerce").dropna()
        if len(sp) == 0 or len(sn) == 0:
            continue
        anchors[f] = {
            "pos_p50": round(float(sp.median()), 4),
            "neg_p50": round(float(sn.median()), 4),
            "median_diff": round(float(sp.median() - sn.median()), 4),
            "suggested_threshold": round(float((sp.median() + sn.median()) / 2.0), 4),
        }
    return dict(sorted(anchors.items(), key=lambda kv: abs(kv[1]["median_diff"]), reverse=True))


def restrict_to_shap(anchors, shap_feats, min_keep=4):
    """Keep only the SHAP-selected features (in SHAP order), backfilling from the
    median-difference order if too few of them have computable anchors."""
    selected = {f: anchors[f] for f in shap_feats if f in anchors}
    if len(selected) < min_keep:
        for f, v in anchors.items():
            if f not in selected:
                selected[f] = v
            if len(selected) >= min_keep:
                break
    return selected


# =========================================================================== #
#  RULE CODE GENERATION (seeds), VALIDATION AND EVALUATION
# =========================================================================== #

def _and_rule(conds):
    """Conjunction: all conditions must hold -> 1 (nested if inside if)."""
    ind = "    "
    lines, depth = ["def rule(x):"], 0
    for f, op, t in conds:
        lines.append(f'{ind*(depth+1)}if x["{f}"] {op} {t}:'); depth += 1
    lines.append(f"{ind*(depth+1)}return 1")
    for _ in conds:
        depth -= 1
        lines.append(f"{ind*(depth+1)}else:")
        lines.append(f"{ind*(depth+2)}return 0")
        if depth == 0:
            break
    return "\n".join(lines)


def _or_rule(conds):
    """Disjunction: any condition true -> 1 (each check inside the previous else)."""
    ind = "    "
    lines, depth = ["def rule(x):"], 0
    for f, op, t in conds:
        lines.append(f'{ind*(depth+1)}if x["{f}"] {op} {t}:')
        lines.append(f"{ind*(depth+2)}return 1")
        lines.append(f"{ind*(depth+1)}else:")
        depth += 1
    lines.append(f"{ind*(depth+1)}return 0")
    return "\n".join(lines)


RULE_FEATURE_RE = re.compile(r'x\[(?:"|\')([^"\']+)(?:"|\')\]')


def validate_rule(code):
    """Reject unsafe or out-of-spec rules (imports, loops, elif, unknown features, ...)."""
    if not code or "def rule" not in code:
        return False, "missing rule function"
    for bad in ["import ", "__import__", "open(", "exec(", "eval(", "os.", "sys.",
                "subprocess", "pickle", "joblib", "line_id_encoded"]:
        if bad in code:
            return False, f"forbidden token: {bad}"
    unknown = sorted(f for f in set(RULE_FEATURE_RE.findall(code)) if f not in FEATURES_FOR_LLM)
    if unknown:
        return False, f"disallowed feature(s): {unknown}"
    try:
        compile(code, "<rule>", "exec")
    except Exception as e:
        return False, f"compile error: {e}"
    return True, ""


def evaluate_rule(code, X):
    """Run a validated rule over the rows of X and return 0/1 predictions."""
    env: Dict[str, Any] = {}
    exec(code, {}, env)
    fn = env["rule"]
    return np.asarray([1 if int(fn(row)) == 1 else 0 for _, row in X.iterrows()], dtype=int)


def compute_seed_rules(X, y, anchors, k_values=K_VALUES):
    """Data-driven candidate rules (no LLM): for each subset size k in {3..8}, build
    single-feature, AND-pair, OR-pair and OR-triple rules over the top-k features and keep
    the best by F2. OR rules raise recall, which is the main lever for F2. The returned best
    seed bootstraps the LLM loop; the rest are shown to the generator as a baseline."""
    feats = list(anchors.keys())
    cands: List[Dict[str, Any]] = []

    def _try(code, k, kind):
        ok, _ = validate_rule(code)
        if not ok:
            return
        try:
            m = compute_metrics(y, evaluate_rule(code, X))
        except Exception:
            return
        cands.append({"rule_code": code, "score": score_rule(m), "k": k, "kind": kind, **m})

    for k in k_values:
        top = feats[:k]
        if not top:
            continue
        # single feature, thresholds scanned around the suggested midpoint
        for f in top:
            a = anchors[f]
            op = ">=" if a["median_diff"] > 0 else "<="
            for d in (-0.20, -0.10, 0.0, 0.10, 0.20):
                _try(_and_rule([(f, op, round(a["suggested_threshold"] * (1 + d), 6))]), k, "uni")
        # pairs (AND and OR) among the strongest features
        m_top = min(4, len(top))
        for i in range(m_top):
            for j in range(i + 1, m_top):
                f1, f2 = top[i], top[j]
                a1, a2 = anchors[f1], anchors[f2]
                op1 = ">=" if a1["median_diff"] > 0 else "<="
                op2 = ">=" if a2["median_diff"] > 0 else "<="
                for d1 in (-0.10, 0.0, 0.10):
                    for d2 in (-0.10, 0.0, 0.10):
                        t1 = round(a1["suggested_threshold"] * (1 + d1), 6)
                        t2 = round(a2["suggested_threshold"] * (1 + d2), 6)
                        _try(_and_rule([(f1, op1, t1), (f2, op2, t2)]), k, "and")
                        _try(_or_rule([(f1, op1, t1), (f2, op2, t2)]), k, "or")
        # OR over the top-3 features (strong recall booster)
        if len(top) >= 3:
            conds = [(f, ">=" if anchors[f]["median_diff"] > 0 else "<=",
                      anchors[f]["suggested_threshold"]) for f in top[:3]]
            _try(_or_rule(conds), k, "or3")

    # de-duplicate and keep the 10 best by F2
    seen, best = set(), []
    for c in sorted(cands, key=lambda x: x["score"], reverse=True):
        if c["rule_code"] not in seen:
            seen.add(c["rule_code"]); best.append(c)
        if len(best) >= 10:
            break
    return best


def worst_cases(X, y_true, y_pred, cols, max_items=MAX_WORST_CASES):
    """Misclassified rows (false negatives first) projected onto the selected features."""
    rows = []
    for i in range(len(X)):
        if int(y_true[i]) != int(y_pred[i]):
            row = {c: round(safe_float(X.iloc[i][c]), 4) for c in cols if c in X.columns}
            row["y_true"] = int(y_true[i]); row["y_pred"] = int(y_pred[i])
            row["error"] = "FN" if int(y_true[i]) == 1 else "FP"
            rows.append(row)
    return sorted(rows, key=lambda r: 0 if r["error"] == "FN" else 1)[:max_items]


# =========================================================================== #
#  PROMPT BUILDERS
# =========================================================================== #

def _fmt(m, key):
    return "n/a" if (not m or m.get(key) is None) else f"{m[key]:.3f}"


def generator_prompt(line_name, stats, anchors, cur, best, feedback, wrong, alerts="", seeds=None):
    anchors_text = ""
    for f, v in anchors.items():
        direction = "higher in failures" if v["median_diff"] > 0 else "lower in failures"
        anchors_text += (f"  {f} ({FEATURE_DESCRIPTIONS.get(f, '')}) -> {direction}; "
                         f"failures_median={v['pos_p50']}, normal_median={v['neg_p50']}, "
                         f"suggested_threshold={v['suggested_threshold']}\n")
    stats_text = "".join(
        f"  {f}: min={stats[f]['min']:.3f} mean={stats[f]['mean']:.3f} max={stats[f]['max']:.3f}\n"
        for f in anchors if f in stats)
    seed_section = ""
    if seeds:
        seed_section = "\n# Data-driven starting rules (baseline - try to beat the best F2)\n"
        for s in seeds:
            seed_section += f"  F2={s['f2']:.3f} (kind={s['kind']}, k={s['k']}):\n  {s['rule_code']}\n\n"
    fields = {
        "cur_f2": _fmt(cur, "f2"), "cur_ovr": _fmt(cur, "ovr"), "cur_fa": _fmt(cur, "fa"),
        "alerts": ("\n" + alerts + "\n") if alerts else "",
        "seed_section": seed_section, "line_name": line_name,
        "anchors_text": anchors_text or "None", "feature_stats_text": stats_text or "None",
        "worst_cases": to_json(wrong) if wrong else "None (no errors)",
        "feedback": clip_text(feedback, 1200) if feedback else "None",
        "best_f2": _fmt(best, "f2"),
        "best_rule": clip_text(best.get("rule_code"), MAX_RULE_CHARS) if best and best.get("rule_code") else "None",
    }
    return fill_template(load_prompt(GENERATOR_PROMPT_FILE), fields)


def critic_prompt(rule_code, cur, anchors, best, wrong, oscillation=""):
    summary = ""
    for f, v in list(anchors.items())[:6]:
        direction = "higher in failures" if v["median_diff"] > 0 else "lower in failures"
        summary += (f"  {f}: failures_median={v['pos_p50']}, normal_median={v['neg_p50']}, "
                    f"suggested_threshold={v['suggested_threshold']} ({direction})\n")
    fields = {
        "current_rule": clip_text(rule_code, MAX_RULE_CHARS),
        "cur_f2": _fmt(cur, "f2"), "cur_ovr": _fmt(cur, "ovr"), "cur_fa": _fmt(cur, "fa"),
        "oscillation_warning": ("\n" + oscillation + "\n") if oscillation else "",
        "anchors_summary": summary, "worst_cases": to_json(wrong) if wrong else "None",
        "best_f2": _fmt(best, "f2"),
        "best_rule": clip_text(best.get("rule_code"), MAX_RULE_CHARS) if best and best.get("rule_code") else "None",
    }
    return fill_template(load_prompt(CRITIC_PROMPT_FILE), fields)


def repair_prompt(bad_output, error_msg):
    return fill_template(load_prompt(REPAIR_PROMPT_FILE),
                         {"error_msg": str(error_msg), "bad_output": str(bad_output)})


# Extract the three tagged sections from an LLM response.
TAGS = {
    "justification": r"\[Start of Justification\](.*?)\[End of Justification\]",
    "changes": r"\[Start of Changes\](.*?)\[End of Changes\]",
    "rule": r"\[Start of Rule\](.*?)\[End of Rule\]",
}


def extract_section(text, name):
    m = re.search(TAGS[name], text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_rule(text):
    """Pull the `def rule(x): ...` block out of an LLM response (tagged or fenced)."""
    if not text:
        return ""
    tagged = extract_section(text, "rule")
    if tagged and "def rule" in tagged:
        text = tagged
    for block in re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE):
        if "def rule" in block:
            return block.strip()
    m = re.search(r"(def\s+rule\s*\(\s*\w+\s*\)\s*:[\s\S]*)", text)
    return m.group(1).strip() if m else ""


# =========================================================================== #
#  LLM CALL
# =========================================================================== #

def call_llm(prompt, temperature):
    """POST a single-message chat completion to the configured endpoint, with retries."""
    cfg = load_llm_config()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {cfg['api_key']}"}
    payload = {"model": cfg["model"], "messages": [{"role": "user", "content": prompt}],
               "temperature": temperature, "max_tokens": cfg.get("max_tokens", 4000)}
    for attempt in range(1, API_RETRIES + 1):
        try:
            r = requests.post(cfg["api_url"], headers=headers, json=payload,
                              verify=cfg.get("verify_ssl", True), timeout=cfg.get("timeout", API_TIMEOUT))
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
            content = r.json()["choices"][0]["message"]["content"]
            return content if isinstance(content, str) else str(content)
        except Exception as e:
            if attempt == API_RETRIES:
                raise RuntimeError(f"LLM call failed after {API_RETRIES} attempts: {e}") from e
            time.sleep(SLEEP_BETWEEN_RETRIES)


# =========================================================================== #
#  PER-LINE EXPERIMENT
# =========================================================================== #

def run_line(df_line, line_name, line_id, temperature, line_dir, split_mode):
    """Dispatch on the split strategy. For LOO, run each fold and pool the test predictions."""
    ensure_dir(line_dir)
    if split_mode == "loo":
        folds = leave_one_out_splits(df_line)
        results = [_run_on_split(tr, te, line_name, line_id, temperature,
                                 os.path.join(line_dir, f"fold_{i}"))
                   for i, (tr, te) in enumerate(folds)]
        best = dict(max(results, key=lambda r: (r["sel_score"] if r["sel_score"] is not None else -9)))
        yt, yp = [], []
        for r in results:
            yt += r["y_test"]; yp += r["y_pred"]
        best["y_test"], best["y_pred"] = yt, yp          # pooled over folds
        best["line_name"], best["split_mode"] = line_name, "loo"
        return best
    tr, te = (adaptive_split(df_line) if split_mode == "adaptive" else sequential_split(df_line))
    return _run_on_split(tr, te, line_name, line_id, temperature, line_dir, split_mode)


def _run_on_split(train_df, test_df, line_name, line_id, temperature, line_dir, split_mode="standard"):
    ensure_dir(line_dir)
    X_test, y_test = test_df[FEATURES_FOR_LLM].copy(), test_df["target"].astype(int).values

    # ---- features: anchors restricted to the SHAP-selected set for this line ------------ #
    anchors = compute_anchors(train_df, FEATURES_FOR_LLM)
    shap_feats = SHAP_RANKINGS.get(line_name)
    if shap_feats:
        anchors = restrict_to_shap(anchors, shap_feats, min_keep=4)
    sel_features = list(anchors.keys())

    # ---- selection set = TRAIN; the test set is only used for the final report ---------- #
    X_sel, y_sel = train_df[FEATURES_FOR_LLM].copy(), train_df["target"].astype(int).values

    # ---- data-driven seed rules (no LLM) ------------------------------------------------ #
    seeds = compute_seed_rules(X_sel, y_sel, anchors, K_VALUES)
    best_rule, best, best_score = "", None, float("-inf")
    if seeds:
        top = seeds[0]
        best_rule = top["rule_code"]
        best = {**{k: top[k] for k in ("acc", "f2", "fa", "ovr", "tp", "fp", "fn", "tn")},
                "rule_code": top["rule_code"]}
        best_score = top["score"]
        print(f"  [seed] {line_name}: F2={top['f2']:.3f} ovr={top['ovr']:.3f} fa={top['fa']:.3f} "
              f"(kind={top['kind']}, k={top['k']})")

    # ---- Generator - Critic - Repair refinement loop ------------------------------------ #
    prev, prev_feedback, recent_ovr, recent_fa = None, "", [], []
    for it in range(1, ITERATIONS + 1):
        stats = feature_stats(build_balanced_window(train_df), sel_features)

        # warn the generator if the rule is oscillating between misses and false alarms
        alerts = ""
        if len(recent_ovr) >= 3 and any(v > 0.5 for v in recent_ovr[-4:]) and any(v > 0.5 for v in recent_fa[-4:]):
            alerts = ("Oscillation: the rule is swinging between missing failures and too many "
                      "alarms. Make ONE small threshold change (<=10%). Do not flip conditions.")

        bp = evaluate_rule(best_rule, X_sel) if best_rule else np.zeros_like(y_sel)
        wrong = worst_cases(X_sel, y_sel, bp, sel_features)

        gp = generator_prompt(line_name, stats, anchors, prev or best, best, prev_feedback,
                              wrong, alerts, seeds[:3] if it <= 3 else None)
        try:
            raw = call_llm(gp, temperature)
            rule_code = clip_text(extract_rule(raw), MAX_RULE_CHARS)

            ok, err = validate_rule(rule_code)
            if not ok:  # one repair attempt
                rule_code = extract_rule(call_llm(repair_prompt(raw, err), temperature))
                ok, err = validate_rule(rule_code)
                if not ok:
                    prev, prev_feedback = None, f"Previous candidate invalid: {err}. Fix it."
                    continue

            m = compute_metrics(y_sel, evaluate_rule(rule_code, X_sel))
            s = score_rule(m)
            feedback = clip_text(call_llm(critic_prompt(rule_code, m, anchors, best,
                                                        worst_cases(X_sel, y_sel, evaluate_rule(rule_code, X_sel),
                                                                    sel_features), alerts), temperature),
                                 MAX_FEEDBACK_CHARS)

            if s > best_score:  # keep the best rule (selected on TRAIN)
                best_score, best_rule = s, rule_code
                best = {**m, "rule_code": rule_code}

            recent_ovr.append(m["ovr"]); recent_fa.append(m["fa"])
            recent_ovr, recent_fa = recent_ovr[-6:], recent_fa[-6:]
            prev, prev_feedback = m, feedback
        except Exception as e:
            prev_feedback = f"Previous iteration failed: {str(e)[:300]}. Repair and keep compact."

    # ---- final report on the held-out TEST set ------------------------------------------ #
    y_pred = evaluate_rule(best_rule, X_test) if best_rule else np.zeros_like(y_test)
    test_metrics = compute_metrics(y_test, y_pred) if best_rule else {k: None for k in
                   ("acc", "f2", "fa", "ovr", "tp", "fp", "fn", "tn")}

    with open(os.path.join(line_dir, "best_rule.py"), "w", encoding="utf-8") as f:
        f.write(best_rule or "# no valid rule found\n")

    return {
        "line_name": line_name, "split_mode": split_mode,
        "sel_score": None if best_score == float("-inf") else float(best_score),
        "sel_f2": (best or {}).get("f2"),
        **test_metrics,                       # reported metrics are on TEST
        "y_test": list(map(int, y_test)), "y_pred": list(map(int, y_pred)),
    }


# =========================================================================== #
#  MULTI-SEED ORCHESTRATION AND GLOBAL (MICRO-AVERAGED) METRICS
# =========================================================================== #

def pooled_metrics(per_line):
    """Micro-average across all lines by pooling their per-sample test predictions. This is
    the headline number, directly comparable to the global DT and HGB."""
    yt, yp = [], []
    for r in per_line:
        yt += r["y_test"]; yp += r["y_pred"]
    return compute_metrics(np.array(yt), np.array(yp)) if yt else None


def run_seed(seed, df):
    random.seed(seed)
    np.random.seed(seed)
    seed_dir = os.path.join(OUTPUT_DIR, f"seed_{seed}")
    ensure_dir(seed_dir)

    per_line = []
    for line_name, line_id in LINE_MAP.items():
        df_line = df[df["line_disconnected"] == line_name].copy()
        n_pos = int(df_line["target"].sum())
        if n_pos < 2:
            print(f"  [skip] {line_name}: {n_pos} failure(s)")
            continue
        mode = ("loo" if n_pos < MIN_POS_LOO else "adaptive" if n_pos < MIN_POS_STANDARD else "standard")
        print(f"  [{line_name}] failures={n_pos} split={mode}")
        try:
            per_line.append(run_line(df_line, line_name, line_id, TEMPERATURE,
                                     os.path.join(seed_dir, f"line_{line_name}"), mode))
        except Exception as e:
            print(f"  [error] {line_name}: {e}"); traceback.print_exc()

    if per_line:
        pd.DataFrame([{k: v for k, v in r.items() if k not in ("y_test", "y_pred")} for r in per_line]
                     ).to_csv(os.path.join(seed_dir, "per_line.csv"), index=False)
    gm = pooled_metrics(per_line)
    if gm:
        print(f"  [seed {seed}] F2={gm['f2']*100:.1f} Acc={gm['acc']*100:.1f} "
              f"OVR={gm['ovr']*100:.1f} FA={gm['fa']*100:.1f}")
    return gm


def main():
    ensure_dir(OUTPUT_DIR)
    print(f"seeds={SEEDS} | tau={TEMPERATURE} | iterations={ITERATIONS} | k-sweep={K_VALUES}")

    rows = []
    for seed in SEEDS:
        print(f"\n================= seed {seed} =================")
        gm = run_seed(seed, prepare_dataframe())
        if gm:
            rows.append({**gm, "seed": seed})

    if not rows:
        print("\nNo results produced.")
        return

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUTPUT_DIR, "per_seed.csv"), index=False)

    print("\n========= Table III (mean +/- std over seeds, %) =========")
    summary = {"n_seeds": len(res)}
    for key, label in [("f2", "F2-score"), ("acc", "Accuracy"), ("ovr", "OVR"), ("fa", "FA")]:
        vals = res[key].values * 100.0
        mean, std = float(vals.mean()), (float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"))
        summary[f"{key}_mean"], summary[f"{key}_std"] = mean, std
        print(f"  {label:9s}: {mean:.1f} +/- {std:.1f}")
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'summary.json')}")


if __name__ == "__main__":
    main()
