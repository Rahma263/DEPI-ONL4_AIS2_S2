import os, joblib
import mlflow, mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from config import RANDOM_STATE, OUTPUT_DIR, EXPERIMENT_NAME
from utils.data_utils import read_and_clean_data
from utils.preprocessing import build_preprocessor
from utils.metrics import evaluate_and_plot
from utils.tracker import log_experiment


def get_feature_names(preprocessor):
    """Extract feature names after preprocessing for SHAP or analysis."""
    output_features = []
    for name, trans, cols in preprocessor.transformers_:
        if hasattr(trans, 'named_steps') and 'ohe' in trans.named_steps:
            ohe = trans.named_steps['ohe']
            output_features.extend(ohe.get_feature_names_out(cols))
        else:
            output_features.extend(cols)
    return output_features


def run_pipeline(dataset_path):
    # ---------- Data Preparation ----------
    df = read_and_clean_data(dataset_path)
    X, y = df.drop(columns=['Exited']), df['Exited']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE
    )

    num_features = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']
    cat_features = ['Gender', 'Geography']
    ready_features = list(set(X_train.columns) - set(num_features) - set(cat_features))

    preprocessor = build_preprocessor(num_features, cat_features, ready_features)

    # -------------------- MLflow Setup --------------------
    MLFLOW_DIR = os.path.join(OUTPUT_DIR, "mlruns")
    os.makedirs(MLFLOW_DIR, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ======================================================
    # 1️⃣ RANDOM FOREST MODEL
    # ======================================================
    model_rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
    pipe_rf = ImbPipeline([
        ('pre', preprocessor),
        ('smote', SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.7)),
        ('clf', model_rf)
    ])

    param_grid_rf = {
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [10, 15, 20],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search_rf = RandomizedSearchCV(pipe_rf, param_grid_rf, n_iter=10, scoring='f1', cv=cv, n_jobs=-1, random_state=RANDOM_STATE)

    with mlflow.start_run(run_name="RandomForest_Training") as run:
        print("🚀 Training RandomForest model...")
        search_rf.fit(X_train, y_train)
        best_rf = search_rf.best_estimator_

        run_dir = os.path.join(OUTPUT_DIR, run.info.run_id)
        os.makedirs(run_dir, exist_ok=True)

        y_pred = best_rf.predict(X_test)
        y_proba = best_rf.predict_proba(X_test)[:, 1]
        metrics_rf = evaluate_and_plot(y_test, y_pred, y_proba, 'RandomForest', run_dir)

        feature_names = get_feature_names(best_rf.named_steps['pre'])
        feat_path = os.path.join(run_dir, "feature_names.txt")
        with open(feat_path, "w") as f:
            f.writelines("\n".join(feature_names))

        # log with mlflow
        mlflow.log_params(search_rf.best_params_)
        mlflow.log_metrics(metrics_rf)
        mlflow.log_artifacts(run_dir)
        mlflow.sklearn.log_model(best_rf, artifact_path="model_rf")
        joblib.dump(best_rf, os.path.join(run_dir, "best_rf_model.joblib"))

        print("✅ RandomForest model saved at:", run_dir)

        # Also record in tracker
        log_experiment(
            run_name="RandomForest_Training",
            params=search_rf.best_params_,
            metrics=metrics_rf,
            artifacts={"run_dir": run_dir},
            use_mlflow=False,   # already logged in mlflow above
            output_dir=OUTPUT_DIR
        )

    # ======================================================
    # 2️ LOGISTIC REGRESSION MODEL
    # ======================================================
    pipe_lr = ImbPipeline([
        ('pre', preprocessor),
        ('smote', SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.7)),
        ('clf', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
    ])

    with mlflow.start_run(run_name="LogisticRegression_Training") as run:
        print(" Training Logistic Regression model...")
        pipe_lr.fit(X_train, y_train)
        y_pred_lr = pipe_lr.predict(X_test)
        y_proba_lr = pipe_lr.predict_proba(X_test)[:, 1]
        metrics_lr = evaluate_and_plot(y_test, y_pred_lr, y_proba_lr, 'LogisticRegression', OUTPUT_DIR)

        mlflow.log_params({'model': 'LogisticRegression'})
        mlflow.log_metrics(metrics_lr)
        mlflow.sklearn.log_model(pipe_lr, artifact_path="model_lr")
        joblib.dump(pipe_lr, os.path.join(OUTPUT_DIR, "best_lr_model.joblib"))

        print(" Logistic Regression model saved at:", OUTPUT_DIR)

        # record via tracker too
        log_experiment(
            run_name="LogisticRegression_Training",
            params={'model': 'LogisticRegression'},
            metrics=metrics_lr,
            artifacts={"plots": OUTPUT_DIR},
            use_mlflow=False,
            output_dir=OUTPUT_DIR
        )

    print("\n All models trained, tracked, and saved successfully!")
