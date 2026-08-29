import dagshub
import mlflow

mlflow.set_tracking_uri('https://dagshub.com/sujat-khan/MLFlow-DVC-Dagshub-Project.mlflow')


dagshub.init(repo_owner='sujat-khan', repo_name='MLFlow-DVC-Dagshub-Project', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)