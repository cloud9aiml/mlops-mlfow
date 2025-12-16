import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection  import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='cloud9aiml', repo_name='mlops-mlfow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/cloud9aiml/mlops-mlfow.mlflow")

mlflow.autolog()
experiment_name = "Remote_MLFlow_Server"

# 2. Create experiment if it does not exist
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id

# 3. Set the experiment
mlflow.set_experiment(experiment_name)

wine = load_wine()
x = wine.data
y = wine.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

max_depth = 10
n_estimators= 25


with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=42)
    rf.fit(x_train, y_train)

    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("max_depth",max_depth)
    mlflow.log_param("n_estimator",n_estimators)


    # Creating a confusion matrix

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matix!')

    plt.savefig("Confusion-matix.png")
    mlflow.log_artifact(__file__)

    print(accuracy)