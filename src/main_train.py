import os
import sys
import csv
import argparse
import pandas as pd

from joblib import dump
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import train_xgb_regr, train_xgb_class, train_svm_regr, train_svm_classif
from utils import y_columns, get_x_y_labels, col_names_switch

ROOT = Path(__file__).parent

OUT_PATH = ROOT / "outputs"
IN_PATH  = ROOT / ".." / "raw" / "Grain"

SUPPORTED_MODELS = ("xgb", "svm")

# ─────────────────────────────── helpers ────────────────────────────────────

def group_wavelengths(data: pd.DataFrame, beams_step: int) -> pd.DataFrame:
    """Average spectral bands into groups of size `beams_step`."""
    if beams_step == 1:
        return data

    data = data.select_dtypes(include="number")
    grouped = {
        f"group_{i // beams_step}": data.iloc[:, i : i + beams_step].mean(axis=1)
        for i in range(0, data.shape[1], beams_step)
    }
    return pd.DataFrame(grouped)


def build_output_dirs(base_out: Path, freq_tag: str, sub: str, model_type: str) -> tuple[Path, Path]:
    """Return (csv_dir, models_dir) and create them if they don't exist."""
    csv_dir    = base_out / model_type / "performance" / freq_tag / sub
    models_dir = base_out / model_type /  "trained_models"  / freq_tag / sub
    csv_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    return csv_dir, models_dir


def save_model_and_indices(model, selected_indices, models_dir: Path, stem: str) -> None:
    dump(model, models_dir / f"{stem}.joblib")
    with open(models_dir / f"{stem}_selected_freq.csv", "w", newline="") as f:
        csv.writer(f).writerow(selected_indices)


# ─────────────────────────── data loading ───────────────────────────────────

def load_dataset(path: Path, sheet_name: str):
    """Load train/val DataFrames from an Excel sheet."""
    if sheet_name == "DATASET":
        data     = pd.read_excel(path, sheet_name=sheet_name)
        val_data = pd.read_excel(path, sheet_name="VALID")
        val_data.columns = val_data.columns.map(str)
    else:
        data     = pd.read_excel(path, sheet_name=sheet_name, header=1)
        val_data = None

    data.columns = data.columns.map(str)
    return data, val_data


def split_train_val(
    data: pd.DataFrame,
    val_data: pd.DataFrame,
    random_state: int = 42,
    frac: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if val_data is not None:
        return data, val_data
    train = data.sample(frac=frac, random_state=random_state)
    return train, data.drop(train.index)


# ─────────────────────── feature / label extraction ─────────────────────────

def prepare_splits(
    dataset: str,
    sheet_name: str,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    beams_step: int,
    train_on_X_first_freq: int,
    col_idx_offset: int = 0,
) -> tuple[dict, dict, dict, dict]:
    """Build per-target train/val feature and label dicts."""
    col_names = (
        y_columns[sheet_name] if dataset == "08_CornGrain" else y_columns["ALL_FOSS"]
    )
    print("\tTarget columns:", col_names)

    X_train, X_val, y_train, y_val = {}, {}, {}, {}

    for col_idx, y_col in enumerate(col_names):
        y_col_act = y_col if y_col in y_columns["DATASET"] else col_names_switch[y_col]

        # Raw features + labels
        X_train[y_col_act], y_train[y_col_act] = get_x_y_labels(
            train_data, y_col, train_on_X_first_freq
        )
        val_y_col = (
            y_columns["VALID"][col_idx]
            if dataset == "08_CornGrain" and sheet_name == "DATASET"
            else y_col
        )
        X_val[y_col_act], y_val[y_col_act] = get_x_y_labels(
            val_data, val_y_col, train_on_X_first_freq
        )

        # Wavelength grouping
        X_train[y_col_act] = group_wavelengths(X_train[y_col_act], beams_step)
        X_val[y_col_act]   = group_wavelengths(X_val[y_col_act],   beams_step)

        # Filter zero-valued labels (only if they're the minority)
        n_train = len(y_train[y_col_act])
        if n_train > len(train_data) / 2:
            mask_tr = y_train[y_col_act] != 0
            mask_vl = y_val[y_col_act]   != 0
            X_train[y_col_act] = X_train[y_col_act][mask_tr]
            y_train[y_col_act] = y_train[y_col_act][mask_tr]
            X_val[y_col_act]   = X_val[y_col_act][mask_vl]
            y_val[y_col_act]   = y_val[y_col_act][mask_vl]

    return X_train, X_val, y_train, y_val


# ────────────────────────── main data pipeline ──────────────────────────────

def main(
    dataset: str,
    sheet_names: list[str],
    path: Path,
    beams_step: int,
    train_on_X_first_freq: int,
) -> tuple[dict, dict, dict, dict]:
    X_train_all, X_val_all, y_train_all, y_val_all = {}, {}, {}, {}

    for sheet_name in sheet_names:
        data, val_data = load_dataset(path, sheet_name)
        train_data, val_data = split_train_val(data, val_data)

        X_tr, X_vl, y_tr, y_vl = prepare_splits(
            dataset, sheet_name, train_data, val_data, beams_step, train_on_X_first_freq
        )
        X_train_all.update(X_tr)
        X_val_all.update(X_vl)
        y_train_all.update(y_tr)
        y_val_all.update(y_vl)

    return X_train_all, X_val_all, y_train_all, y_val_all


# ──────────────────────────── model training ────────────────────────────────

def get_model_fns(model_type: str) -> tuple:
    """Return (regression_fn, classification_fn) for the chosen model type."""
    if model_type == "xgb":
        return train_xgb_regr, train_xgb_class
    if model_type == "svm":
        return train_svm_regr, train_svm_classif
    raise ValueError(f"Unknown model type '{model_type}'. Choose from: {SUPPORTED_MODELS}")


def train_regression(
    X_tr, X_vl, y_tr, y_vl,
    y_col: str,
    csv_dir: Path,
    models_dir: Path,
    beams_step: int,
    train_regr_fn,
    model_type: str,
    sub: str,
) -> None:
    print(f"\nTraining {model_type.upper()} regression — target: {y_col}  [{sub}]")
    results = train_regr_fn(X_tr, X_vl, y_tr, y_vl, y_col, str(csv_dir), beams_step)
    save_model_and_indices(results["model"], results["selected_indices"],
                           models_dir, f"{y_col}_regr")


def train_classification(
    X_tr, X_vl, labels_tr, labels_vl,
    csv_dir: Path,
    models_dir: Path,
    beams_step: int,
    train_class_fn,
    model_type: str,
    sub: str,
) -> None:
    print(f"\nTraining {model_type.upper()} classification  [{sub}]")
    results = train_class_fn(X_tr, X_vl, labels_tr, labels_vl, str(csv_dir), beams_step)
    save_model_and_indices(results["model"], results["selected_indices"],
                           models_dir, "class")


def make_dataset_labels(data_dict: dict, y_col: str) -> pd.Series:
    return pd.Series(
        [ds for ds in data_dict for _ in range(len(data_dict[ds][y_col]))],
        name="dataset_label",
    )


# ──────────────────────── multi-dataset training ────────────────────────────

def train_multi_dataset(
    X_train_all, X_val_all, y_train_all, y_val_all,
    freq_tag: str,
    beams_step: int,
    model_type: str,
) -> None:
    print("─── Training a unified model on all families ───")
    train_regr_fn, train_class_fn = get_model_fns(model_type)
    sub = f"multi_dataset_training_{model_type}"
    csv_dir, models_dir = build_output_dirs(OUT_PATH, freq_tag, sub, model_type)

    first_ds = next(iter(y_train_all))
    for y_col in y_train_all[first_ds]:
        X_tr = pd.concat(X_train_all[ds][y_col] for ds in X_train_all)
        y_tr = pd.concat(y_train_all[ds][y_col] for ds in y_train_all)
        X_vl = pd.concat(X_val_all[ds][y_col]   for ds in X_val_all)
        y_vl = pd.concat(y_val_all[ds][y_col]   for ds in y_val_all)

        train_regression(X_tr, X_vl, y_tr, y_vl, y_col,
                         csv_dir, models_dir, beams_step,
                         train_regr_fn, model_type, sub)

        if y_col in ("DM", "SS"):
            labels_tr = make_dataset_labels(X_train_all, y_col)
            labels_vl = make_dataset_labels(X_val_all,   y_col)
            train_classification(X_tr, X_vl, labels_tr, labels_vl,
                                 csv_dir, models_dir, beams_step,
                                 train_class_fn, model_type, sub)


# ────────────────────── per-family training ─────────────────────────────────

DATASETS_TO_TRAIN = {"Barley", "Soybean"}  # TODO: remove filter when ready


def train_per_family(
    X_train_all, X_val_all, y_train_all, y_val_all,
    freq_tag: str,
    beams_step: int,
    model_type: str,
) -> None:
    print("─── Training a model for each family ───")
    train_regr_fn, _ = get_model_fns(model_type)

    for dataset_name in X_train_all:
        if dataset_name not in DATASETS_TO_TRAIN:
            continue

        csv_dir, models_dir = build_output_dirs(
            OUT_PATH, freq_tag, dataset_name, model_type
        )
        # One sub-folder per dataset inside the models dir
        dataset_models_dir = models_dir / dataset_name
        dataset_models_dir.mkdir(parents=True, exist_ok=True)

        for y_col, X_tr in X_train_all[dataset_name].items():
            X_vl = X_val_all[dataset_name][y_col]
            y_tr = y_train_all[dataset_name][y_col]
            y_vl = y_val_all[dataset_name][y_col]

            if X_tr.empty or X_vl.empty:
                continue

            train_regression(X_tr, X_vl, y_tr, y_vl, y_col,
                             csv_dir, dataset_models_dir, beams_step,
                             train_regr_fn, model_type, dataset_name)


# ─────────────────────────────── entry point ────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train regression/classification models on NIR data.")
    parser.add_argument(
        "--model_type", choices=SUPPORTED_MODELS, default="xgb",
        help="Model backend to use (default: xgb).",
    )
    parser.add_argument(
        "--beams_step", type=int, default=1,
        help="Step size for grouping wavelengths (1 = no grouping).",
    )
    parser.add_argument(
        "--multi_dataset", action="store_true",
        help="Train a single unified model on all datasets instead of one per family.",
    )
    parser.add_argument(
        "--train_on_X_first_freq", type=int, default=-1,
        help="Use only the first N frequencies for training (-1 = all).",
    )
    return parser.parse_args()


def freq_tag(train_on_X_first_freq: int) -> str:
    return f"{train_on_X_first_freq}freqs" if train_on_X_first_freq > 0 else "all_freqs"


if __name__ == "__main__":
    args = parse_args()

    if args.beams_step < 1:
        raise ValueError("--beams_step must be >= 1.")

    sheet_names = ["DATASET"]
    tag = freq_tag(args.train_on_X_first_freq)

    X_train_all, X_val_all, y_train_all, y_val_all = {}, {}, {}, {}

    for dir_entry in IN_PATH.iterdir():
        if not dir_entry.is_dir():
            continue

        # Robustly extract dataset name: strip leading digits + underscore, trailing "Grain"
        raw_name = dir_entry.name.split("_", 1)[-1]
        dataset_name = raw_name.removesuffix("Grain")

        for file in dir_entry.iterdir():
            if file.suffix == ".xlsx" and file.stem.endswith("DATASET"):
                print(f"\nProcessing dataset: {dataset_name}  |  file: {file.name}")
                (
                    X_train_all[dataset_name],
                    X_val_all[dataset_name],
                    y_train_all[dataset_name],
                    y_val_all[dataset_name],
                ) = main(dir_entry.name, sheet_names, file, args.beams_step, args.train_on_X_first_freq)

    if args.multi_dataset:
        train_multi_dataset(
            X_train_all, X_val_all, y_train_all, y_val_all,
            tag, args.beams_step, args.model_type,
        )
    else:
        train_per_family(
            X_train_all, X_val_all, y_train_all, y_val_all,
            tag, args.beams_step, args.model_type,
        )
