# -*- coding: utf-8 -*-
"""
PACE OCI PFT retrieval using an SVD-XGBoost framework.

This script trains taxon-specific retrieval models for phytoplankton functional
types using matched hyperspectral Rrs, chlorophyll-a, and pigment-derived PFT
concentrations.

Expected Excel sheets:
    - Rrs_PACE: station ID in the first column, Rrs bands in the remaining columns
    - 藻种数据: PFT concentration columns
    - Chla: a column named "Chla"

Example:
    python pft_retrieval_svd_xgboost.py --input data/input.xlsx --output results
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


RANDOM_STATE = 42
N_SPLITS = 5
MIN_CONCENTRATION = 0.001
EPS = 1e-10


ALGAE_CONFIGS = {
    "Prasinophytes": {
        "target_column": 1,
        "bands": [415, 422, 432, 437, 447, 455, 457, 465, 467, 477, 482, 615, 650, 665],
    },
    "Dinoflagellates": {
        "target_column": 2,
        "bands": [415, 418, 422, 432, 437, 442, 447, 450, 452, 455, 457, 465, 467,
                  472, 475, 477, 480, 482, 485, 580, 586, 615, 632, 635, 650, 665],
    },
    "Cryptophytes": {
        "target_column": 3,
        "bands": [430, 432, 450, 452, 455, 482, 580, 586, 615, 632, 635, 665],
    },
    "Chrysophytes": {
        "target_column": 4,
        "bands": [413, 415, 437, 440, 445, 447, 450, 452, 455, 465,
                  470, 475, 477, 480, 580, 583, 630, 632],
    },
    "Prymnesiophytes": {
        "target_column": 5,
        "bands": [422, 432, 447, 450, 452, 455, 457, 472, 477, 586, 588, 615, 635, 665],
    },
    "Chlorophytes": {
        "target_column": 6,
        "bands": [415, 418, 422, 432, 437, 442, 447, 455, 465, 467, 472, 477, 482, 615, 650, 665],
    },
    "Cyanobacteria": {
        "target_column": 7,
        "bands": [432, 442, 455, 475, 482, 615, 620, 652, 663, 665],
    },
    "Diatoms": {
        "target_column": 8,
        "bands": [418, 422, 432, 442, 447, 450, 452, 455, 457, 472,
                  477, 480, 482, 580, 586, 588, 615, 632, 635, 665],
    },
}


XGB_PARAM_GRIDS = {
    "high": {
        "n_estimators": [100, 200],
        "max_depth": [3, 4],
        "learning_rate": [0.05, 0.10],
        "min_child_weight": [3],
        "reg_alpha": [0.1, 1.0],
    },
    "medium": {
        "n_estimators": [100],
        "max_depth": [3],
        "learning_rate": [0.05],
        "min_child_weight": [5],
        "reg_alpha": [1.0],
    },
    "low": {
        "n_estimators": [100],
        "max_depth": [2, 3],
        "learning_rate": [0.03],
        "min_child_weight": [5],
        "reg_alpha": [1.0, 5.0],
    },
}


ALGAE_PARAM_GROUP = {
    "Cryptophytes": "high",
    "Diatoms": "high",
    "Dinoflagellates": "high",
    "Chlorophytes": "medium",
    "Chrysophytes": "medium",
    "Prasinophytes": "low",
    "Cyanobacteria": "low",
    "Prymnesiophytes": "low",
}


def get_k_candidates(n_samples: int) -> List[int]:
    if n_samples < 80:
        return [5, 8, 12, 15, 20]
    if n_samples < 120:
        return [8, 12, 15, 20, 25]
    return [8, 12, 15, 20, 25, 30, 35, 40, 50]


def get_xgb_grid(algae_key: str, n_samples: int) -> Dict[str, List[float]]:
    group = ALGAE_PARAM_GROUP.get(algae_key, "medium")
    grid = dict(XGB_PARAM_GRIDS[group])
    grid["k"] = get_k_candidates(n_samples)
    return grid


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    value = np.corrcoef(x, y)[0, 1]
    if np.isnan(value) or np.isinf(value):
        return 0.0

    return float(value)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(100 * np.mean(np.abs((y_true - y_pred) / (y_true + 1e-12))))


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_true_log: np.ndarray | None = None,
    y_pred_log: np.ndarray | None = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.nan_to_num(y_pred, nan=1e-6, posinf=1e6, neginf=1e-6), 1e-6, 1e6)

    metrics = {
        "R2_linear": r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": mape(y_true, y_pred),
    }

    if y_true_log is not None and y_pred_log is not None and len(y_true_log) >= 2:
        y_pred_log = np.nan_to_num(
            y_pred_log,
            nan=float(np.nanmedian(y_true_log)),
            posinf=float(np.nanmax(y_true_log)),
            neginf=float(np.nanmin(y_true_log)),
        )
        metrics["R2_log"] = r2_score(y_true_log, y_pred_log)
    else:
        metrics["R2_log"] = np.nan

    return metrics


def smearing_factor(y_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    residuals = np.asarray(y_log, dtype=float) - np.asarray(y_pred_log, dtype=float)
    if len(residuals) == 0:
        return 1.0
    return float(np.median(10 ** residuals))


def replace_negative_rrs(rrs: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    values = rrs.copy()
    negative = values < 0

    if not negative.any():
        return values

    positive = ~negative
    if positive.sum() < 2:
        return values

    interpolator = interp1d(
        wavelengths[positive],
        values[positive],
        kind="linear",
        fill_value="extrapolate",
    )
    values[negative] = interpolator(wavelengths[negative])
    values[negative] = np.maximum(values[negative], values[positive].min() * 0.1)

    return values


def generate_band_features(rrs: np.ndarray, wavelengths: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    features = []
    names = []
    n_bands = rrs.shape[1]

    ordered_pairs = [(i, j) for i in range(n_bands) for j in range(n_bands) if i != j]
    unordered_pairs = [(i, j) for i in range(n_bands) for j in range(i + 1, n_bands)]

    for i in range(n_bands):
        features.append(rrs[:, i])
        names.append(f"Rrs_{int(wavelengths[i])}")

    for i, j in ordered_pairs:
        features.append(rrs[:, i] / (rrs[:, j] + EPS))
        names.append(f"Rrs_{int(wavelengths[i])}_div_Rrs_{int(wavelengths[j])}")

    for i, j in ordered_pairs:
        features.append(rrs[:, i] - rrs[:, j])
        names.append(f"Rrs_{int(wavelengths[i])}_minus_Rrs_{int(wavelengths[j])}")

    for i, j in unordered_pairs:
        features.append(rrs[:, i] + rrs[:, j])
        names.append(f"Rrs_{int(wavelengths[i])}_plus_Rrs_{int(wavelengths[j])}")

    for i, j in ordered_pairs:
        ratio = rrs[:, i] / (rrs[:, j] + EPS)
        diff = rrs[:, i] - rrs[:, j]
        summ = rrs[:, i] + rrs[:, j]

        features.append(rrs[:, i] / (ratio + EPS))
        names.append(f"Rrs_{int(wavelengths[i])}_div_ratio_{int(wavelengths[i])}_{int(wavelengths[j])}")

        features.append(rrs[:, i] / (diff + EPS))
        names.append(f"Rrs_{int(wavelengths[i])}_div_diff_{int(wavelengths[i])}_{int(wavelengths[j])}")

        features.append(rrs[:, i] / (summ + EPS))
        names.append(f"Rrs_{int(wavelengths[i])}_div_sum_{int(wavelengths[i])}_{int(wavelengths[j])}")

        features.append(ratio / (diff + EPS))
        names.append(f"ratio_div_diff_{int(wavelengths[i])}_{int(wavelengths[j])}")

        features.append(ratio / (summ + EPS))
        names.append(f"ratio_div_sum_{int(wavelengths[i])}_{int(wavelengths[j])}")

        features.append(diff / (summ + EPS))
        names.append(f"diff_div_sum_{int(wavelengths[i])}_{int(wavelengths[j])}")

    matrix = np.column_stack(features)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=1e10, neginf=-1e10)

    return matrix, names


def concentration_holdout_split(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort samples by concentration and reserve every fourth sample for holdout validation."""
    y = np.asarray(y, dtype=float)
    holdout = np.zeros(len(y), dtype=bool)
    groups = np.zeros(len(y), dtype=int)

    if len(y) < 4:
        return holdout, groups

    sorted_index = np.argsort(y)

    for group_id, start in enumerate(range(0, len(y), 4)):
        block = sorted_index[start:start + 4]
        groups[block] = group_id
        if len(block) == 4:
            holdout[block[-1]] = True

    return ~holdout, holdout


def select_features(
    x_train: np.ndarray,
    y_train_log: np.ndarray,
    max_features: int,
    max_collinearity: float = 0.90,
) -> np.ndarray:
    correlations = np.array([abs(safe_corr(x_train[:, i], y_train_log)) for i in range(x_train.shape[1])])
    ranked = np.argsort(correlations)[::-1]

    selected = []
    for idx in ranked:
        if len(selected) >= max_features:
            break

        if not selected:
            selected.append(idx)
            continue

        keep = all(abs(safe_corr(x_train[:, idx], x_train[:, j])) <= max_collinearity for j in selected)
        if keep:
            selected.append(idx)

    if not selected:
        selected = [int(np.argmax(correlations))]

    return np.array(selected, dtype=int)


def fit_svd_xgb(
    x_train: np.ndarray,
    chla_train: np.ndarray,
    y_train: np.ndarray,
    params: Dict[str, float],
) -> Dict:
    y_train_log = np.log10(y_train + 1e-6)
    k = min(int(params["k"]), x_train.shape[1])

    selected_columns = select_features(x_train, y_train_log, max_features=k)
    x_selected = x_train[:, selected_columns]

    band_scaler = StandardScaler()
    x_scaled = band_scaler.fit_transform(x_selected)

    max_components = max(1, min(k - 1, len(y_train) - 1, 15))
    svd_probe = TruncatedSVD(n_components=max_components, random_state=RANDOM_STATE)
    svd_probe.fit(x_scaled)

    variance = np.cumsum(svd_probe.explained_variance_ratio_)
    n_components = int(np.searchsorted(variance, 0.98) + 1)
    n_components = max(1, min(n_components, max_components))

    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    x_svd = svd.fit_transform(x_scaled)

    chla_scaler = StandardScaler()
    chla_scaled = chla_scaler.fit_transform(np.asarray(chla_train).reshape(-1, 1))

    x_final = np.column_stack([x_svd, chla_scaled])

    model = XGBRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        min_child_weight=float(params["min_child_weight"]),
        reg_alpha=float(params["reg_alpha"]),
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(x_final, y_train_log)

    train_pred_log = model.predict(x_final)

    return {
        "selected_columns": selected_columns,
        "band_scaler": band_scaler,
        "svd": svd,
        "chla_scaler": chla_scaler,
        "model": model,
        "smearing": smearing_factor(y_train_log, train_pred_log),
        "params": params,
        "n_components": n_components,
    }


def predict_svd_xgb(package: Dict, x: np.ndarray, chla: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_selected = x[:, package["selected_columns"]]
    x_scaled = package["band_scaler"].transform(x_selected)
    x_svd = package["svd"].transform(x_scaled)
    chla_scaled = package["chla_scaler"].transform(np.asarray(chla).reshape(-1, 1))

    x_final = np.column_stack([x_svd, chla_scaled])
    y_pred_log = package["model"].predict(x_final)
    y_pred = (10 ** y_pred_log) * package["smearing"]
    y_pred = np.clip(np.nan_to_num(y_pred, nan=1e-6, posinf=1e6, neginf=1e-6), 1e-6, 1e6)

    return y_pred, y_pred_log


def evaluate_params(
    params: Dict[str, float],
    x_dev: np.ndarray,
    chla_dev: np.ndarray,
    y_dev: np.ndarray,
) -> Dict[str, float]:
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for fold_id, (train_idx, val_idx) in enumerate(cv.split(x_dev), start=1):
        package = fit_svd_xgb(x_dev[train_idx], chla_dev[train_idx], y_dev[train_idx], params)
        y_pred, y_pred_log = predict_svd_xgb(package, x_dev[val_idx], chla_dev[val_idx])

        metrics = calculate_metrics(
            y_dev[val_idx],
            y_pred,
            y_true_log=np.log10(y_dev[val_idx] + 1e-6),
            y_pred_log=y_pred_log,
        )
        metrics["fold"] = fold_id
        rows.append(metrics)

    fold_table = pd.DataFrame(rows)
    return {
        "params": params,
        "CV_RMSE": fold_table["RMSE"].mean(),
        "CV_MAE": fold_table["MAE"].mean(),
        "CV_MAPE": fold_table["MAPE"].mean(),
        "CV_R2_log": fold_table["R2_log"].mean(),
    }


def tune_model(
    algae_key: str,
    x_dev: np.ndarray,
    chla_dev: np.ndarray,
    y_dev: np.ndarray,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    grid = get_xgb_grid(algae_key, len(y_dev))
    rows = [evaluate_params(params, x_dev, chla_dev, y_dev) for params in ParameterGrid(grid)]

    table = pd.DataFrame(rows)
    table["params_json"] = table["params"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    best_idx = table["CV_RMSE"].idxmin()

    return table.loc[best_idx, "params"], table.drop(columns=["params"])


def prepare_taxon_data(
    algae_key: str,
    config: Dict,
    df_rrs: pd.DataFrame,
    df_pft: pd.DataFrame,
    df_chla: pd.DataFrame,
) -> Dict:
    stations = df_rrs.iloc[:, 0].values
    wavelengths = df_rrs.columns[1:].astype(float).values
    rrs_all = df_rrs.iloc[:, 1:].values.astype(float)

    target = df_pft.iloc[:, config["target_column"]].values.astype(float)
    chla = df_chla["Chla"].values.astype(float)

    target_bands = np.array(config["bands"], dtype=float)
    band_index = [int(np.argmin(np.abs(wavelengths - band))) for band in target_bands]
    matched_wavelengths = wavelengths[band_index]
    rrs = rrs_all[:, band_index]

    valid = (
        (target >= MIN_CONCENTRATION)
        & np.isfinite(target)
        & (chla > 0)
        & np.isfinite(chla)
        & np.isfinite(rrs).all(axis=1)
        & ((rrs < 0).sum(axis=1) <= 5)
    )

    rrs = rrs[valid].copy()
    for i in range(len(rrs)):
        rrs[i] = replace_negative_rrs(rrs[i], matched_wavelengths)

    feature_matrix, feature_names = generate_band_features(rrs, matched_wavelengths)
    dev_mask, holdout_mask = concentration_holdout_split(target[valid])

    return {
        "algae": algae_key,
        "x": feature_matrix,
        "y": target[valid],
        "chla": chla[valid],
        "stations": stations[valid],
        "dev_mask": dev_mask,
        "holdout_mask": holdout_mask,
        "bands": target_bands,
        "matched_wavelengths": matched_wavelengths,
        "feature_names": feature_names,
    }


def process_taxon(
    algae_key: str,
    config: Dict,
    df_rrs: pd.DataFrame,
    df_pft: pd.DataFrame,
    df_chla: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    data = prepare_taxon_data(algae_key, config, df_rrs, df_pft, df_chla)

    x = data["x"]
    y = data["y"]
    chla = data["chla"]

    dev_mask = data["dev_mask"]
    holdout_mask = data["holdout_mask"]

    if dev_mask.sum() < N_SPLITS * 3 or holdout_mask.sum() < 5:
        raise ValueError(f"Insufficient samples for {algae_key}")

    best_params, cv_table = tune_model(algae_key, x[dev_mask], chla[dev_mask], y[dev_mask])
    package = fit_svd_xgb(x[dev_mask], chla[dev_mask], y[dev_mask], best_params)

    holdout_pred, holdout_pred_log = predict_svd_xgb(package, x[holdout_mask], chla[holdout_mask])
    holdout_metrics = calculate_metrics(
        y[holdout_mask],
        holdout_pred,
        y_true_log=np.log10(y[holdout_mask] + 1e-6),
        y_pred_log=holdout_pred_log,
    )

    dev_pred, dev_pred_log = predict_svd_xgb(package, x[dev_mask], chla[dev_mask])
    dev_metrics = calculate_metrics(
        y[dev_mask],
        dev_pred,
        y_true_log=np.log10(y[dev_mask] + 1e-6),
        y_pred_log=dev_pred_log,
    )

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{algae_key}_svd_xgboost.pkl"
    joblib.dump(
        {
            "algae": algae_key,
            "best_params": best_params,
            "model_package": package,
            "bands": data["bands"],
            "matched_wavelengths": data["matched_wavelengths"],
            "feature_names": data["feature_names"],
        },
        model_path,
    )

    summary = {
        "algae": algae_key,
        "n_samples": len(y),
        "n_development": int(dev_mask.sum()),
        "n_holdout": int(holdout_mask.sum()),
        "best_params": json.dumps(best_params),
        "Dev_R2_log": dev_metrics["R2_log"],
        "Dev_MAE": dev_metrics["MAE"],
        "Dev_RMSE": dev_metrics["RMSE"],
        "Dev_MAPE": dev_metrics["MAPE"],
        "Holdout_R2_log": holdout_metrics["R2_log"],
        "Holdout_MAE": holdout_metrics["MAE"],
        "Holdout_RMSE": holdout_metrics["RMSE"],
        "Holdout_MAPE": holdout_metrics["MAPE"],
        "model_file": str(model_path),
    }

    prediction_table = pd.DataFrame(
        {
            "algae": algae_key,
            "station": data["stations"][holdout_mask],
            "observed": y[holdout_mask],
            "predicted": holdout_pred,
            "absolute_error": np.abs(y[holdout_mask] - holdout_pred),
            "relative_error_percent": np.abs(y[holdout_mask] - holdout_pred) / (y[holdout_mask] + 1e-12) * 100,
        }
    )

    cv_table.insert(0, "algae", algae_key)

    return summary, cv_table, prediction_table


def run(input_file: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df_rrs = pd.read_excel(input_file, sheet_name="Rrs_PACE")
    df_pft = pd.read_excel(input_file, sheet_name="藻种数据")
    df_chla = pd.read_excel(input_file, sheet_name="Chla")

    summaries = []
    cv_tables = []
    prediction_tables = []

    for algae_key, config in ALGAE_CONFIGS.items():
        print(f"Processing {algae_key}...")

        try:
            summary, cv_table, prediction_table = process_taxon(
                algae_key,
                config,
                df_rrs,
                df_pft,
                df_chla,
                output_dir,
            )
            summaries.append(summary)
            cv_tables.append(cv_table)
            prediction_tables.append(prediction_table)

        except Exception as exc:
            print(f"Skipped {algae_key}: {exc}")

    if not summaries:
        raise RuntimeError("No model was successfully trained.")

    pd.DataFrame(summaries).to_excel(output_dir / "model_summary.xlsx", index=False)
    pd.concat(cv_tables, ignore_index=True).to_excel(output_dir / "cross_validation_results.xlsx", index=False)
    pd.concat(prediction_tables, ignore_index=True).to_excel(output_dir / "holdout_predictions.xlsx", index=False)

    print(f"Results saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SVD-XGBoost models for PACE OCI PFT retrieval.")
    parser.add_argument("--input", required=True, type=Path, help="Input Excel file.")
    parser.add_argument("--output", default=Path("results"), type=Path, help="Output directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output)
