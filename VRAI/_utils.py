import os
import numpy as np
import pandas as pd
import xgboost as xgb

import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from deep_nir.VRAI.preprocessing import SNV, SavitzkyGolaySmooth, MSC


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

def train_svm_regr(X_train, X_val, y_train, y_val, y_col, outpath):
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
    selector = RFECV(
        estimator=SVR(kernel='linear'),  # sistema semplice per ranking delle bande
        step=1,
        cv=5,
        scoring='r2'  # Use R2 for regression tasks
    )

    pipe = Pipeline([
        ('snv', SNV()),
        ('msc', MSC()),
        ('savgol', SavitzkyGolaySmooth(window_length=7, polyorder=2, deriv=0)),
        ('scaler', StandardScaler()),
        ('band_select', selector),   # seleziona band in base all’importanza
        ('pca', PCA(n_components=0.99, whiten=True)),  # opzionale dopo band select
        ('model', SVR(kernel='rbf'))
    ])

    param_grid = {
        "model__C": [0.1, 1, 10, 100],
        "model__epsilon": [0.01, 0.1, 1.0],
        "model__gamma": ["scale", 0.1, 0.01, 0.001]
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    reg = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, scoring="neg_mean_absolute_error")
    reg.fit(X_train, y_train.ravel())

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
    plot_predictions(y_val_array, preds, y_col, "svm", "validation", outpath)

    # Make predictions on training set
    train_preds = final_reg.predict(X_train)
    train_rmse = ((train_preds - y_train.ravel()) ** 2).mean() ** 0.5
    print(f"Training RMSE: {train_rmse}")
    train_metrics = compute_regression_metrics(y_train.ravel(), train_preds)
    for key, value in train_metrics.items():
        train_dict["train_" + key + "_" + y_col] = value

    # Save train_dict to csv
    df_output = pd.DataFrame([train_dict])
    with open(os.path.join(outpath, f"{y_col}_SVM.csv"), 'w') as f:
        df_output.to_csv(f, index=False)

    # Plot predictions
    plot_predictions(y_train.ravel(), train_preds, y_col, "svm", "train", outpath)

# def train_svm_classif(X_train, X_val, y_train, y_val, y_col, outpath):
def train_svm_classif(X_train, X_val, y_train, y_val, y_col, outpath):
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

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train.ravel())
    y_val_enc = le.transform(y_val.ravel())

    pipe = Pipeline([
        ('snv', SNV()),
        ('msc', MSC()),
        ('savgol', SavitzkyGolaySmooth(window_length=7, polyorder=2, deriv=0)),
        ('scaler', StandardScaler()),
        ('model', SVC(kernel='rbf'))
    ])

    param_grid = {
        "model__C": [0.1, 1, 10, 100],
        "model__gamma": ["scale", 0.1, 0.01, 0.001]
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    clf = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, scoring="accuracy")
    clf.fit(X_train, y_train_enc)

    print("Best params:", clf.best_params_)
    pred = clf.predict(X_val)
    print("Accuracy:", accuracy_score(y_val_enc, pred))
    print(classification_report(y_val_enc, pred))
    print(confusion_matrix(y_val_enc, pred))

    # Make predictions: print accuracy on validation set
    preds = clf.predict(X_val)

    accuracy = accuracy_score(y_val_enc, preds)
    print(f"Validation Accuracy: {accuracy}")
    train_dict["val_Accuracy_" + y_col] = accuracy

    # Plot predictions
    plot_predictions(y_val_enc, preds, y_col, "svm_classif", "validation", outpath)

    # Make predictions on training set
    train_preds = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train_enc, train_preds)
    print(f"Training Accuracy: {train_accuracy}")
    train_dict["train_Accuracy_" + y_col] = train_accuracy

    # Save train_dict to csv
    df_output = pd.DataFrame([train_dict])
    with open(os.path.join(outpath, f"{y_col}_SVM_Classification.csv"), 'w') as f:
        df_output.to_csv(f, index=False)
