import mlflow
from mlflow_utils import get_mlflow_experiment

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, ConfusionMatrixDisplay

import matplotlib
matplotlib.use("Agg")  # منع مشاكل Qt في بيئة headless
import matplotlib.pyplot as plt

if __name__ == "__main__":

    experiment = get_mlflow_experiment(experiment_name="testing_mlflow_1")
    if experiment is None:
        raise Exception("Experiment not found. Please create it first.")

    print("Name: {}".format(experiment.name))

    with mlflow.start_run(run_name="logging_images", experiment_id=experiment.experiment_id) as run:

        # إنشاء بيانات تجريبية
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                                   n_redundant=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

        # تدريب نموذج
        rfc = RandomForestClassifier(n_estimators=100, random_state=42)
        rfc.fit(X_train, y_train)
        y_pred = rfc.predict(X_test)

        # Precision-Recall curve
        fig_pr, ax_pr = plt.subplots()
        PrecisionRecallDisplay.from_predictions(y_test, y_pred, ax=ax_pr)
        ax_pr.set_title("Precision-Recall Curve")
        mlflow.log_figure(fig_pr, "metrics/precision_recall_curve.png")
        plt.close(fig_pr)  # اغلاق الشكل لتحرير الذاكرة

        # ROC curve
        fig_roc, ax_roc = plt.subplots()
        RocCurveDisplay.from_predictions(y_test, y_pred, ax=ax_roc)
        ax_roc.set_title("ROC Curve")
        mlflow.log_figure(fig_roc, "metrics/roc_curve.png")
        plt.close(fig_roc)

        # Confusion Matrix
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm)
        ax_cm.set_title("Confusion Matrix")
        mlflow.log_figure(fig_cm, "metrics/confusion_matrix.png")
        plt.close(fig_cm)

        # معلومات الـ run
        print("run_id: {}".format(run.info.run_id))
        print("experiment_id: {}".format(run.info.experiment_id))
        print("status: {}".format(run.info.status))
        print("start_time: {}".format(run.info.start_time))
        print("end_time: {}".format(run.info.end_time))
        print("lifecycle_stage: {}".format(run.info.lifecycle_stage))