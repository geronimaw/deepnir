import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

# from kan import KAN, ex_round
from sklearn.svm import SVC, SVR
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from xgboost import XGBRegressor, XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.append(str(Path(__file__).parent.parent))

from VRAI.preprocessing import SNV, SavitzkyGolaySmooth, MSC
from VRAI.utils.visual import plot_predictions, plot_cm


# Compute regression metrics
def compute_regression_metrics(y_true, y_pred):
    n = len(y_true)
    residuals = y_true - y_pred
    bias = residuals.mean()
    sep = (residuals ** 2).sum() / (n - 1)
    sepc = ((residuals - bias) ** 2).sum() / (n - 1)
    r2 = r2_score(y_true, y_pred)
    slope = np.corrcoef(y_true, y_pred)[0, 1] * (y_pred.std() / y_true.std())

    # Add MIN, MAX, AVG, STD for y_true and y_pred
    min_true = np.min(y_true)
    max_true = np.max(y_true)
    avg_true = np.mean(y_true)
    std_true = np.std(y_true)
    min_pred = np.min(y_pred)
    max_pred = np.max(y_pred)
    avg_pred = np.mean(y_pred)
    std_pred = np.std(y_pred)
    
    return {
        "SEP": sep, "Bias": bias, "SEPC": sepc, "R2": r2, "Slope": slope, "N": n,
        "Min_True": min_true, "Max_True": max_true, "Avg_True": avg_true, "Std_True": std_true,
        "Min_Pred": min_pred, "Max_Pred": max_pred, "Avg_Pred": avg_pred, "Std_Pred": std_pred}

def compute_classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Extract precision, recall, f1-score for each class
    metrics = {"Accuracy": accuracy}
    for cls, cls_report in report.items():
        if cls in ["accuracy", "macro avg", "weighted avg"]:
            continue  # Skip these entries
        metrics[f"Precision_{cls}"] = cls_report["precision"]
        metrics[f"Recall_{cls}"] = cls_report["recall"]
        metrics[f"F1-Score_{cls}"] = cls_report["f1-score"]

    return metrics


def train_xgb(dtrain, dtest, y_val, y_col, outpath, prep=True):
    train_dict = {}

    #TODO: add preprocessing steps if prep=True

    # Set parameters for XGBoost. very few samples, so we need a simple model
    if "classification" in y_col:
        params = {
            "objective": "multi:softmax",
            "num_class": len(np.unique(y_val)),
            "max_depth": 2,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "mlogloss"
        }
        num_rounds = 50
    else:
        params = {
            "objective": "reg:squarederror",
            "max_depth": 2,
            "eta": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "rmse"
        }
        num_rounds = 100

    # Train the model: show evaluation metrics on validation set along with SEP, Bias, SEPC, R2, Slope, N (at the end)
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=num_rounds, evals=evals)
    
    # Make predictions: print rmse on validation set
    preds = model.predict(dtest)
    # y_val may not have ".values" if it's a single column
    y_val_array = y_val.ravel() if hasattr(y_val, 'values') else y_val.ravel()
    rmse = ((preds - y_val_array) ** 2).mean() ** 0.5
    print(f"Validation RMSE: {rmse}")
    metrics = compute_regression_metrics(y_val_array, preds)
    for key, value in metrics.items():
        train_dict["performance_" + key + "_" + y_col] = value
    
    # Plot predictions
    plot_predictions(y_val_array, preds, y_col, "xgb", "valid", outpath)

    # Make predictions on training set
    train_preds = model.predict(dtrain)
    train_rmse = ((train_preds - dtrain.get_label()) ** 2).mean
    print(f"Training RMSE: {train_rmse}")
    train_metrics = compute_regression_metrics(dtrain.get_label(), train_preds)
    for key, value in train_metrics.items():
        train_dict["train_" + key + "_" + y_col] = value

    # Save train_dict to csv
    df_output = pd.DataFrame([train_dict])
    with open(os.path.join(outpath, f"{y_col}_XGB.csv"), 'w') as f:
        df_output.to_csv(f, index=False)

    # Plot predictions
    plot_predictions(dtrain.get_label(), train_preds, y_col, "xgb", "calib", outpath)


def train_xgb_regr(X_train, X_val, y_train, y_val, y_col, outpath, beams_step=1):

    # Ensure numpy arrays
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()
    if isinstance(X_val, pd.DataFrame):
        X_val = X_val.to_numpy()
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy()
    if isinstance(y_val, pd.Series):
        y_val = y_val.to_numpy()

    train_dict = {}

    xgb_params = dict(
    booster="gbtree",
    objective="reg:squarederror",
    eval_metric="rmse",

    max_depth=3,
    min_child_weight=7,
    gamma=0.2,

    subsample=0.8,
    colsample_bytree=0.8,

    learning_rate=0.05,
    n_estimators=5000,

    reg_lambda=10,
    reg_alpha=0.2,

    tree_method="hist",
    random_state=42,

    )

    # ===== Feature selection model =====
    xgb_selector = XGBRegressor(
        **xgb_params
    )

    selector = SelectFromModel(
        estimator=xgb_selector,
        threshold="median"  # seleziona ~50% bande più importanti
    )

    # ===== Pipeline =====
    pipe = Pipeline([
        ('snv', SNV()),
        ('msc', MSC()),
        ('savgol', SavitzkyGolaySmooth(window_length=7, polyorder=2, deriv=0)),
        ('scaler', StandardScaler()),
        ('band_select', selector),
        ('model', XGBRegressor(
            **xgb_params
        ))
    ])

    # ===== Hyperparameters =====
    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.7, 0.9],
        "model__colsample_bytree": [0.7, 0.9]
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    reg = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        n_jobs=-1,
        scoring="neg_mean_absolute_error"
    )

    reg.fit(
        X_train, y_train.ravel(),
        )

    # ===== Feature selection info =====
    selector_fitted = reg.best_estimator_.named_steps['band_select']
    mask_selected = selector_fitted.get_support()
    selected_indices = np.where(mask_selected)[0]
    num_bands_selected = len(selected_indices)

    print(f"Numero bande selezionate da XGB: {num_bands_selected}")
    print("Indici bande selezionate:", selected_indices)
    print("Best params:", reg.best_params_)

    # ===== Validation =====
    pred = reg.predict(X_val)
    print("MAE:", mean_absolute_error(y_val, pred),
          "R2:", r2_score(y_val, pred))

    # ===== Final refit =====
    final_reg = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        n_jobs=-1,
        scoring="neg_mean_absolute_error"
    )

    final_reg.fit(
        np.vstack([X_train, X_val]),
        np.hstack([y_train, y_val])
    )

    # ===== Validation metrics =====
    preds = final_reg.predict(X_val)
    y_val_array = y_val.ravel()

    rmse = np.sqrt(((preds - y_val_array) ** 2).mean())
    print(f"Validation RMSE: {rmse}")

    metrics = compute_regression_metrics(y_val_array, preds)
    for key, value in metrics.items():
        train_dict[f"val_{key}_{y_col}"] = value

    if outpath is not None:
        plot_predictions(
            y_val_array,
            preds,
            y_col,
            "xgb",
            "validation",
            outpath,
            beams_step
        )

    # ===== Training metrics =====
    train_preds = final_reg.predict(X_train)
    train_rmse = np.sqrt(((train_preds - y_train.ravel()) ** 2).mean())
    print(f"Training RMSE: {train_rmse}")

    train_metrics = compute_regression_metrics(
        y_train.ravel(),
        train_preds
    )

    for key, value in train_metrics.items():
        train_dict[f"train_{key}_{y_col}"] = value

    # ===== Save CSV =====
    df_output = pd.DataFrame([train_dict])
    outname = f"{y_col}_regr_XGB_{'each' + str(beams_step) if beams_step > 1 else ''}.csv"

    if outpath is not None:
        df_output.to_csv(os.path.join(outpath, outname), index=False)

    if outpath is not None:
        plot_predictions(
            y_train.ravel(),
            train_preds,
            f"{y_col}_regr",
            "xgb",
            "train",
            outpath,
            num_feat_selected=num_bands_selected,
            beams_step=beams_step
        )
    
    if outpath is None:
        return {
            "model": final_reg,
            "selected_indices": selected_indices,
            "x_val": X_val[:, selected_indices],
            "y_val": y_val_array,
            "predictions": preds,
            "metrics": metrics,
            "train_metrics": train_metrics,
        }

def train_xgb_class(X_train, X_val, y_train, y_val, outpath, beams_step=1):

    # Ensure numpy arrays
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()
    if isinstance(X_val, pd.DataFrame):
        X_val = X_val.to_numpy()
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy()
    if isinstance(y_val, pd.Series):
        y_val = y_val.to_numpy()

    # convert y_train and y_val to integers if they are categorical
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train.ravel())
    y_val_enc = le.transform(y_val.ravel())
    class_names = le.classes_.astype(str)

    train_dict = {}

    xgb_params = dict(
    booster="gbtree",
    objective="multi:softmax",
    eval_metric="mlogloss",

    max_depth=3,
    min_child_weight=7,
    gamma=0.2,

    subsample=0.8,
    colsample_bytree=0.8,

    learning_rate=0.05,
    n_estimators=5000,

    reg_lambda=10,
    reg_alpha=0.2,

    tree_method="hist",
    random_state=42,

    )

    # ===== Feature selection model =====
    xgb_selector = XGBClassifier(
        **xgb_params
    )

    selector = SelectFromModel(
        estimator=xgb_selector,
        threshold="median"  # seleziona ~50% bande più importanti
    )

    # ===== Pipeline =====
    pipe = Pipeline([
        ('snv', SNV()),
        ('msc', MSC()),
        ('savgol', SavitzkyGolaySmooth(window_length=7, polyorder=2, deriv=0)),
        ('scaler', StandardScaler()),
        ('band_select', selector),
        ('model', XGBClassifier(
            **xgb_params
        ))
    ])

    # ===== Hyperparameters =====
    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.7, 0.9],
        "model__colsample_bytree": [0.7, 0.9]
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # grid search for classification: use scoring='accuracy'
    reg = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        n_jobs=-1,
        scoring='accuracy'
    )

    # fit the model with encoded labels
    reg.fit(
        X_train, y_train_enc,
        )

    # ===== Feature selection info =====
    selector_fitted = reg.best_estimator_.named_steps['band_select']
    mask_selected = selector_fitted.get_support()
    selected_indices = np.where(mask_selected)[0]
    num_bands_selected = len(selected_indices)

    print(f"Numero bande selezionate da XGB: {num_bands_selected}")
    print("Indici bande selezionate:", selected_indices)
    print("Best params:", reg.best_params_)

    # ===== Validation =====
    pred = reg.predict(X_val)
    print("Accuracy:", accuracy_score(y_val_enc, pred),
          "Classification Report:\n", classification_report(y_val_enc, pred, target_names=class_names))

    # ===== Final refit =====
    final_reg = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        n_jobs=-1,
        scoring='accuracy'
    )

    final_reg.fit(
        np.vstack([X_train, X_val]),
        np.hstack([y_train_enc, y_val_enc])
    )

    # ===== Validation metrics =====
    preds = final_reg.predict(X_val)
    y_val_array = y_val_enc.ravel()

    accuracy = (preds == y_val_array).mean()
    print(f"Validation Accuracy: {accuracy}")

    # Compute other classification metrics
    train_dict[f"val_Accuracy"] = accuracy
    train_dict[f"train_Accuracy"] = (final_reg.predict(X_train) == y_train_enc).mean()

    # Plot confusion matrix
    class_names = np.unique(y_train_enc).astype(str)
    plot_cm(confusion_matrix(y_val_array, preds), model='XGB', classes=class_names, acc=accuracy, outpath=outpath)
    
    # ===== Training metrics =====
    train_preds = final_reg.predict(X_train)
    train_accuracy = (train_preds == y_train_enc).mean()
    print(f"Training Accuracy: {train_accuracy}")

    train_metrics = compute_classification_metrics(
        y_train_enc,
        train_preds
    )

    for key, value in train_metrics.items():
        train_dict[f"train_{key}"] = value

    # ===== Save CSV =====
    df_output = pd.DataFrame([train_dict])
    outname = f"class_XGB_{'each' + str(beams_step) if beams_step > 1 else ''}.csv"

    df_output.to_csv(os.path.join(outpath, outname), index=False)


def train_svm_regr(X_train, X_val, y_train, y_val, y_col, outpath, beams_step=1):
    # Ensure inputs are numerical arrays
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()
    if isinstance(X_val, pd.DataFrame):
        X_val = X_val.to_numpy()
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy()
    if isinstance(y_val, pd.Series):
        y_val = y_val.to_numpy()

    train_dict = {}

    # modello di base per la selezione
    # RFECV: Recursive Feature Elimination with Cross-Validation
    selector = RFECV(
        estimator=SVR(kernel='linear'),  # sistema semplice per ranking delle bande
        step=1,
        cv=5,
        scoring='r2'  # Use R2 for regression tasks
    )

    pipe = Pipeline([
        ('snv', SNV()), # SNV: Standard Normal Variate
        ('msc', MSC()), # MSC: Multiplicative Scatter Correction
        ('savgol', SavitzkyGolaySmooth(window_length=7, polyorder=2, deriv=0)),
        ('scaler', StandardScaler()),
        ('band_select', selector),   # seleziona band in base all’importanza
        # ('pca', PCA(n_components=0.99, whiten=True)),  # opzionale dopo band select
        ('model', SVR(kernel='rbf')) # SVR: Support Vector Regression
    ])

    param_grid = {
        "model__C": [0.1, 1, 10, 100],
        "model__epsilon": [0.01, 0.1, 1.0],
        "model__gamma": ["scale", 0.1, 0.01, 0.001]
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    reg = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, scoring="neg_mean_absolute_error")
    reg.fit(X_train, y_train.ravel())

    # recuperare RFECV dal pipeline
    rfecv = reg.best_estimator_.named_steps['band_select']

    # quante feature sono state selezionate
    num_bands_selected = rfecv.n_features_
    print(f"Numero bande selezionate da RFECV: {num_bands_selected}")

    # boolean mask delle bande selezionate
    mask_selected = rfecv.support_

    # indice delle bande selezionate (es. 0..70)
    selected_indices = np.where(mask_selected)[0]
    print("Indici delle bande selezionate:", selected_indices)
    
    # Salva grafico importanza bande TODO: implementare importanza
    # almeno salvare valori in un file

    print("Best params:", reg.best_params_)
    pred = reg.predict(X_val)
    print("MAE:", mean_absolute_error(y_val, pred), "R2:", r2_score(y_val, pred))

    # Optional final refit on all data:
    final_reg = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, scoring="neg_mean_absolute_error")
    final_reg.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))

    # Make predictions: print rmse on validation set
    preds = final_reg.predict(X_val)

    y_val_array = y_val.ravel()
    rmse = ((preds - y_val_array) ** 2).mean() ** 0.5
    print(f"Validation RMSE: {rmse}")
    metrics = compute_regression_metrics(y_val_array, preds)
    for key, value in metrics.items():
        train_dict["val_" + key + "_" + y_col] = value

    # Plot predictions
    plot_predictions(y_val_array, preds, y_col, "svm", "validation", outpath, beams_step)

    # Make predictions on training set
    train_preds = final_reg.predict(X_train)
    train_rmse = ((train_preds - y_train.ravel()) ** 2).mean() ** 0.5
    print(f"Training RMSE: {train_rmse}")
    train_metrics = compute_regression_metrics(y_train.ravel(), train_preds)
    for key, value in train_metrics.items():
        train_dict["train_" + key + "_" + y_col] = value

    # Save train_dict to csv
    df_output = pd.DataFrame([train_dict])
    with open(os.path.join(outpath, f"{y_col}_regr_SVM_{'each' + str(beams_step) if beams_step > 1 else ''}.csv"), 'w') as f:
        df_output.to_csv(f, index=False)

    # Plot predictions
    plot_predictions(y_train.ravel(), train_preds, f"{y_col}_regr", "svm", "train", outpath,
                     num_feat_selected=num_bands_selected, beams_step=beams_step)

# def train_svm_classif(X_train, X_val, y_train, y_val, y_col, outpath):
def train_svm_classif(X_train, X_val, y_train, y_val, y_col, outpath, beams_step=1):
    # Ensure inputs are numerical arrays
    # wavelengths = np.array(X_train.columns)

    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()
    if isinstance(X_val, pd.DataFrame):
        X_val = X_val.to_numpy()
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy()
    if isinstance(y_val, pd.Series):
        y_val = y_val.to_numpy()

    train_dict = {}

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train.ravel())
    y_val_enc = le.transform(y_val.ravel())
    class_names = le.classes_.astype(str)

    # RFECV: Recursive Feature Elimination with Cross-Validation
    selector = RFECV(
        estimator=SVC(kernel='linear'),  # sistema semplice per ranking delle bande
        step=1,
        cv=5,
        scoring='accuracy'  # Use accuracy for classification tasks
    )

    pipe = Pipeline([
        ('snv', SNV()),
        ('msc', MSC()),
        ('savgol', SavitzkyGolaySmooth(window_length=7, polyorder=2, deriv=0)),
        ('scaler', StandardScaler()),
        ('band_select', selector),
        ('model', SVC(kernel='rbf'))
    ])

    param_grid = {
        "model__C": [0.1, 1, 10, 100],
        "model__gamma": ["scale", 0.1, 0.01, 0.001]
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    clf = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, scoring="accuracy")
    clf.fit(X_train, y_train_enc)
    
    # Inspect RFECV results
    rfecv = clf.best_estimator_.named_steps['band_select']
    num_bands_selected = rfecv.n_features_
    print(f"Numero bande selezionate da RFECV: {num_bands_selected}")
    mask_selected = rfecv.support_
    selected_indices = np.where(mask_selected)[0]
    print("Indici delle bande selezionate:", selected_indices)

    print("Best params:", clf.best_params_)
    pred = clf.predict(X_val)
    print("Accuracy:", accuracy_score(y_val_enc, pred))
    print(classification_report(y_val_enc, pred))

    # Make predictions: print accuracy on validation set
    preds = clf.predict(X_val)

    accuracy = accuracy_score(y_val_enc, preds)
    print(f"Validation Accuracy: {accuracy}")
    train_dict["val_Accuracy_" + y_col] = accuracy

    # Plot confusion matrix
    plot_cm(confusion_matrix(y_val_enc, pred), model='SVM', classes=class_names, acc=accuracy, y_col=y_col, outpath=outpath)

    # Plot predictions
    plot_predictions(y_val_enc, preds, y_col, "SVM", "valid", outpath, num_feat_selected=num_bands_selected, beams_step=beams_step)

    # Make predictions on training set
    train_preds = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train_enc, train_preds)
    print(f"Training Accuracy: {train_accuracy}")
    train_dict["train_Accuracy_" + y_col] = train_accuracy

    # Save train_dict to csv
    df_output = pd.DataFrame([train_dict])
    with open(os.path.join(outpath, f"{y_col}_SVM{'_each' + str(beams_step) if beams_step > 1 else ''}.csv"), 'w') as f:
        df_output.to_csv(f, index=False)


# KAN training function