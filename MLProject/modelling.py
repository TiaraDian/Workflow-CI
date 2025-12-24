import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

while mlflow.active_run() is not None:
    mlflow.end_run()

os.environ.pop("MLFLOW_RUN_ID", None)
os.environ.pop("MLFLOW_EXPERIMENT_ID", None)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

mlflow.set_tracking_uri(f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}")
mlflow.set_experiment("heart-disease-ci")

df = pd.read_csv(os.path.join(BASE_DIR, "Heart_Disease_Prediction_preprocessing.csv"))
X = df.drop(columns=["Heart Disease"]).astype("float64")
y = df["Heart Disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="logreg-heart-disease"):

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, artifact_path="model")

    print("Accuracy:", acc)
