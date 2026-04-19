from pathlib import Path
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# CONFIGURATION
DATASET_PATH = Path("AMT_Anomaly_Dataset.csv")
FEATURES = ["delay_ms", "clutch_temp_c", "torque_var_pct", "rpm_diff"]
SEVERITY_LABELS = ["Normal", "Moderate", "Severe"]
ACTION_LABELS = [
    "NO_ACTION",
    "SHIFT_TIMING_ADJUST",
    "CLUTCH_PRESSURE_RECALIB",
    "TORQUE_REDISTRIBUTION",
    "RPM_SYNC",]

RANDOM_STATE = 42
TEST_SIZE = 0.20
TIMING_REPEATS = 50


# -------------------------------------------------------------
# GENERAL UTILITIES
def safe_int_dict(series: pd.Series) -> Dict[str, int]:
    counts = series.value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def average_inference_time_ms(predict_fn, X_batch: np.ndarray, repeats: int = 50) -> float:
    """Average inference time for severity + action on the same batch."""
    for _ in range(3):
        predict_fn(X_batch)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict_fn(X_batch)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.mean(times))


def fp_fn_table(y_true: List[str], y_pred: List[str], labels: List[str]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        rows.append(
            {
                "class": label,
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
            }
        )
    return pd.DataFrame(rows)


def print_confusion_matrix(title: str, y_true: List[str], y_pred: List[str], labels: List[str]) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"Actual_{x}" for x in labels], columns=[f"Pred_{x}" for x in labels])
    print(f"\n{title}")
    print(cm_df.to_string())


def dominant_feature_action(
        row: np.ndarray,
        thresholds: Dict[str, Dict[str, float]],
        severity: str,
) -> str:
    if severity == "Normal":
        return "NO_ACTION"

    feature_names = FEATURES
    action_map = {
        "delay_ms": "SHIFT_TIMING_ADJUST",
        "clutch_temp_c": "CLUTCH_PRESSURE_RECALIB",
        "torque_var_pct": "TORQUE_REDISTRIBUTION",
        "rpm_diff": "RPM_SYNC",
    }

    scores = {}
    for idx, feature in enumerate(feature_names):
        t_nm = thresholds[feature]["normal_to_moderate"]
        t_ms = thresholds[feature]["moderate_to_severe"]

        if severity == "Moderate":
            base = t_nm
            scale = max(t_ms - t_nm, 1e-6)
        else:
            base = t_ms
            scale = max(t_ms - t_nm, 1e-6)

        scores[feature] = (float(row[idx]) - base) / scale

    best_feature = max(scores, key=scores.get)
    return action_map[best_feature]


def print_metrics_block(
        name: str,
        y_true_sev: List[str],
        y_pred_sev: List[str],
        y_true_act: List[str],
        y_pred_act: List[str],
        inference_ms: float,
        batch_size: int,
) -> Dict[str, float]:
    sev_acc = accuracy_score(y_true_sev, y_pred_sev)
    act_acc = accuracy_score(y_true_act, y_pred_act)
    sev_f1 = f1_score(y_true_sev, y_pred_sev, average="weighted")
    act_f1 = f1_score(y_true_act, y_pred_act, average="weighted")

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)
    print(f"Inference time (average for {batch_size} test records): {inference_ms:.3f} ms")
    print(f"Per-record inference time: {inference_ms / batch_size:.4f} ms")
    print(f"Severity accuracy: {sev_acc * 100:.2f}%")
    print(f"Action accuracy:   {act_acc * 100:.2f}%")

    print("\n-- Severity Classification Report (test set) --")
    print(classification_report(y_true_sev, y_pred_sev, labels=SEVERITY_LABELS, zero_division=0))
    print_confusion_matrix("Severity Confusion Matrix", y_true_sev, y_pred_sev, SEVERITY_LABELS)
    print("\nSeverity FP/FN Summary")
    print(fp_fn_table(y_true_sev, y_pred_sev, SEVERITY_LABELS).to_string(index=False))

    print("\n-- Action Classification Report (test set) --")
    print(classification_report(y_true_act, y_pred_act, labels=ACTION_LABELS, zero_division=0))
    print_confusion_matrix("Action Confusion Matrix", y_true_act, y_pred_act, ACTION_LABELS)
    print("\nAction FP/FN Summary")
    print(fp_fn_table(y_true_act, y_pred_act, ACTION_LABELS).to_string(index=False))

    return {
        "sev_acc": sev_acc,
        "act_acc": act_acc,
        "sev_f1": sev_f1,
        "act_f1": act_f1,
        "time_ms": inference_ms,
    }


# ------------------------------------------------------------
# DATA PREPARATION
def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("Dataset not found. ")

    df = pd.read_csv(path)
    required_columns = FEATURES + ["severity", "action"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df.copy()


# -------------------------------------------------------------
# APPROACH 1 - MACHINE LEARNING
def build_ml_models(X_train: np.ndarray, y_train_sev: np.ndarray, y_train_act: np.ndarray):
    sev_encoder = LabelEncoder()
    act_encoder = LabelEncoder()

    y_train_sev_enc = sev_encoder.fit_transform(y_train_sev)
    y_train_act_enc = act_encoder.fit_transform(y_train_act)

    rf_sev = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        min_samples_leaf=1)
    rf_act = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        min_samples_leaf=1)

    rf_sev.fit(X_train, y_train_sev_enc)
    rf_act.fit(X_train, y_train_act_enc)

    def predict_fn(X_batch: np.ndarray) -> Tuple[List[str], List[str]]:
        pred_sev = sev_encoder.inverse_transform(rf_sev.predict(X_batch)).tolist()
        pred_act = act_encoder.inverse_transform(rf_act.predict(X_batch)).tolist()
        return pred_sev, pred_act

    return predict_fn


# ------------------------------------------------------------
# APPROACH 2 - FUZZY LOGIC
def _safe_quantile(series: pd.Series, q: float, fallback: float) -> float:
    if series.empty:
        return float(fallback)
    return float(series.quantile(q))


def _safe_median(series: pd.Series, fallback: float) -> float:
    if series.empty:
        return float(fallback)
    return float(series.median())


def _ordered4(a: float, b: float, c: float, d: float) -> List[float]:
    vals = sorted([float(a), float(b), float(c), float(d)])
    for i in range(1, 4):
        if vals[i] < vals[i - 1]:
            vals[i] = vals[i - 1]
    return vals


def _ordered3(a: float, b: float, c: float) -> List[float]:
    vals = sorted([float(a), float(b), float(c)])
    for i in range(1, 3):
        if vals[i] < vals[i - 1]:
            vals[i] = vals[i - 1]
    return vals


def calibrate_thresholds(train_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    thresholds: Dict[str, Dict[str, float]] = {}
    normal_df = train_df[train_df["severity"] == "Normal"]
    moderate_df = train_df[train_df["severity"] == "Moderate"]
    severe_df = train_df[train_df["severity"] == "Severe"]

    for feature in FEATURES:
        med_n = _safe_median(normal_df[feature], train_df[feature].quantile(0.25))
        med_m = _safe_median(moderate_df[feature], train_df[feature].median())
        med_s = _safe_median(severe_df[feature], train_df[feature].quantile(0.75))

        nm = (med_n + med_m) / 2.0
        ms = (med_m + med_s) / 2.0

        if ms <= nm:
            nm = float(train_df[feature].quantile(0.40))
            ms = float(train_df[feature].quantile(0.70))
            if ms <= nm:
                ms = nm + 1e-6

        thresholds[feature] = {
            "normal_to_moderate": float(nm),
            "moderate_to_severe": float(ms),
        }

    return thresholds


def build_fuzzy_predictor(train_df: pd.DataFrame):
    thresholds = calibrate_thresholds(train_df)

    normal_df = train_df[train_df["severity"] == "Normal"]
    moderate_df = train_df[train_df["severity"] == "Moderate"]
    severe_df = train_df[train_df["severity"] == "Severe"]

    feature_ranges = {}
    for feature in FEATURES:
        global_min = float(train_df[feature].min())
        global_max = float(train_df[feature].max())
        padding = max((global_max - global_min) * 0.05, 1.0)
        feature_ranges[feature] = np.linspace(global_min - padding, global_max + padding, 500)

    severity_range = np.linspace(0, 10, 501)

    delay = ctrl.Antecedent(feature_ranges["delay_ms"], "delay")
    temp = ctrl.Antecedent(feature_ranges["clutch_temp_c"], "temp")
    torque = ctrl.Antecedent(feature_ranges["torque_var_pct"], "torque")
    rpm = ctrl.Antecedent(feature_ranges["rpm_diff"], "rpm")
    severity_out = ctrl.Consequent(severity_range, "severity_out")

    def add_memberships(var, feature_name: str):
        s_all = train_df[feature_name]
        s_n = normal_df[feature_name]
        s_m = moderate_df[feature_name]
        s_s = severe_df[feature_name]

        all_min = float(s_all.min())
        all_max = float(s_all.max())
        low_end = _safe_quantile(s_n, 0.75, s_all.quantile(0.35))
        mid_left = _safe_quantile(s_m, 0.25, s_all.quantile(0.45))
        mid_peak = _safe_median(s_m, s_all.quantile(0.55))
        mid_right = _safe_quantile(s_m, 0.75, s_all.quantile(0.70))
        high_start = _safe_quantile(s_s, 0.25, s_all.quantile(0.75))

        var["low"] = fuzz.trapmf(var.universe, _ordered4(all_min, all_min, low_end, mid_left))
        var["medium"] = fuzz.trimf(var.universe, _ordered3(mid_left, mid_peak, high_start))
        var["high"] = fuzz.trapmf(var.universe, _ordered4(mid_right, high_start, all_max, all_max))

    add_memberships(delay, "delay_ms")
    add_memberships(temp, "clutch_temp_c")
    add_memberships(torque, "torque_var_pct")
    add_memberships(rpm, "rpm_diff")

    severity_out["normal"] = fuzz.trimf(severity_range, [0, 0, 4])
    severity_out["moderate"] = fuzz.trimf(severity_range, [2.5, 5, 7.5])
    severity_out["severe"] = fuzz.trimf(severity_range, [6, 10, 10])

    rules = [
        ctrl.Rule(delay["low"] & temp["low"] & torque["low"] & rpm["low"], severity_out["normal"]),
        ctrl.Rule(delay["medium"], severity_out["moderate"]),
        ctrl.Rule(temp["medium"], severity_out["moderate"]),
        ctrl.Rule(torque["medium"], severity_out["moderate"]),
        ctrl.Rule(rpm["medium"], severity_out["moderate"]),
        ctrl.Rule(delay["high"], severity_out["severe"]),
        ctrl.Rule(temp["high"], severity_out["severe"]),
        ctrl.Rule(torque["high"], severity_out["severe"]),
        ctrl.Rule(rpm["high"], severity_out["severe"]),
        ctrl.Rule(delay["medium"] & temp["medium"], severity_out["severe"]),
        ctrl.Rule(delay["medium"] & rpm["medium"], severity_out["severe"]),
        ctrl.Rule(temp["medium"] & torque["medium"], severity_out["severe"]),
        ctrl.Rule(torque["medium"] & rpm["medium"], severity_out["severe"]),
    ]

    system = ctrl.ControlSystem(rules)

    def predict_single(row: np.ndarray) -> Tuple[str, str]:
        sim = ctrl.ControlSystemSimulation(system)
        sim.input["delay"] = float(row[0])
        sim.input["temp"] = float(row[1])
        sim.input["torque"] = float(row[2])
        sim.input["rpm"] = float(row[3])

        try:
            sim.compute()
            score = float(sim.output["severity_out"])
        except Exception:
            score = 0.0

        if score < 3.5:
            severity = "Normal"
        elif score < 6.5:
            severity = "Moderate"
        else:
            severity = "Severe"

        action = dominant_feature_action(row, thresholds, severity)
        return severity, action

    def predict_fn(X_batch: np.ndarray) -> Tuple[List[str], List[str]]:
        pairs = [predict_single(row) for row in X_batch]
        pred_sev = [p[0] for p in pairs]
        pred_act = [p[1] for p in pairs]
        return pred_sev, pred_act

    return predict_fn


# ------------------------------------------------------------
# APPROACH 3 - RULE-BASED SYSTEM
def build_rule_based_predictor(train_df: pd.DataFrame):
    thresholds = calibrate_thresholds(train_df)

    def predict_single(row: np.ndarray) -> Tuple[str, str]:
        severe_hits = 0
        moderate_hits = 0

        for idx, feature in enumerate(FEATURES):
            value = float(row[idx])
            if value >= thresholds[feature]["moderate_to_severe"]:
                severe_hits += 1
            if value >= thresholds[feature]["normal_to_moderate"]:
                moderate_hits += 1

        if severe_hits >= 2 or (severe_hits >= 1 and moderate_hits >= 3):
            severity = "Severe"
        elif moderate_hits >= 1:
            severity = "Moderate"
        else:
            severity = "Normal"

        action = dominant_feature_action(row, thresholds, severity)
        return severity, action

    def predict_fn(X_batch: np.ndarray) -> Tuple[List[str], List[str]]:
        pairs = [predict_single(row) for row in X_batch]
        pred_sev = [p[0] for p in pairs]
        pred_act = [p[1] for p in pairs]
        return pred_sev, pred_act

    return predict_fn


# -----------------------------------------------------------
# MAIN
def main() -> None:
    df = load_dataset(DATASET_PATH)

    X = df[FEATURES].values
    y_severity = df["severity"].astype(str).values
    y_action = df["action"].astype(str).values

    print("-" * 70)
    print(" AMT Anomaly Detection & Correction")
    print("-" * 70)
    print(f"Features used: {FEATURES}")
    print(f"Severity distribution: {safe_int_dict(df['severity'])}")
    print(f"Action distribution:   {safe_int_dict(df['action'])}")

    # Fair evaluation: every approach uses the same train/test split.
    # Combined stratification preserves both severity and action proportions.
    stratify_labels = (df["severity"].astype(str) + "|" + df["action"].astype(str)).values

    X_train, X_test, y_train_sev, y_test_sev, y_train_act, y_test_act, idx_train, idx_test = train_test_split(
        X,
        y_severity,
        y_action,
        np.arange(len(df)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_labels,)

    train_df = df.iloc[idx_train].reset_index(drop=True)
    print(f"\nTrain records: {len(train_df)} | Test records: {len(X_test)}")
    print("All three approaches below are evaluated on the SAME test set.")

    results = []

    # Approach 1 - Machine Learning
    ml_predict = build_ml_models(X_train, y_train_sev, y_train_act)
    ml_pred_sev, ml_pred_act = ml_predict(X_test)
    ml_time_ms = average_inference_time_ms(ml_predict, X_test, repeats=TIMING_REPEATS)
    ml_metrics = print_metrics_block(
        "APPROACH 1: Machine Learning - Random Forest",
        y_test_sev.tolist(),
        ml_pred_sev,
        y_test_act.tolist(),
        ml_pred_act,
        ml_time_ms,
        len(X_test),
    )
    results.append(("Machine Learning", ml_metrics))

    # Approach 2 - Fuzzy Logic
    fuzzy_predict = build_fuzzy_predictor(train_df)
    fuzzy_pred_sev, fuzzy_pred_act = fuzzy_predict(X_test)
    fuzzy_time_ms = average_inference_time_ms(fuzzy_predict, X_test, repeats=TIMING_REPEATS)
    fuzzy_metrics = print_metrics_block(
        "APPROACH 2: Fuzzy Logic System",
        y_test_sev.tolist(),
        fuzzy_pred_sev,
        y_test_act.tolist(),
        fuzzy_pred_act,
        fuzzy_time_ms,
        len(X_test),
    )
    results.append(("Fuzzy Logic", fuzzy_metrics))

    # Approach 3 - Rule-Based
    rule_predict = build_rule_based_predictor(train_df)
    rule_pred_sev, rule_pred_act = rule_predict(X_test)
    rule_time_ms = average_inference_time_ms(rule_predict, X_test, repeats=TIMING_REPEATS)
    rule_metrics = print_metrics_block(
        "APPROACH 3: Rule-Based System",
        y_test_sev.tolist(),
        rule_pred_sev,
        y_test_act.tolist(),
        rule_pred_act,
        rule_time_ms,
        len(X_test),
    )
    results.append(("Rule-Based", rule_metrics))

    # Comparison summary
    summary_rows = []
    times = [metrics["time_ms"] for _, metrics in results]
    max_time = max(times)
    min_time = min(times)

    for approach_name, metrics in results:
        if max_time == min_time:
            speed_score = 1.0
        else:
            speed_score = 1.0 - ((metrics["time_ms"] - min_time) / (max_time - min_time))

        suitability_score = (
                0.45 * metrics["sev_f1"]
                + 0.45 * metrics["act_f1"]
                + 0.10 * speed_score)

        summary_rows.append(
            {
                "Approach": approach_name,
                "Severity Accuracy (%)": round(metrics["sev_acc"] * 100, 2),
                "Action Accuracy (%)": round(metrics["act_acc"] * 100, 2),
                "Severity F1": round(metrics["sev_f1"], 4),
                "Action F1": round(metrics["act_f1"], 4),
                "Inference Time (ms)": round(metrics["time_ms"], 4),
                "Suitability Score": round(suitability_score, 4),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["Suitability Score", "Inference Time (ms)"],
        ascending=[False, True],
    )

    print("\n" + "-" * 70)
    print("FAIR COMPARISON SUMMARY")
    print("-" * 70)
    print(summary_df.to_string(index=False))

    best = summary_df.iloc[0]
    print("\nRecommended approach for real-time AMT deployment:")
    print(
        f"{best['Approach']} | Suitability Score = {best['Suitability Score']:.4f} | "
        f"Inference Time = {best['Inference Time (ms)']:.4f} ms")


if __name__ == "__main__":
    main()
