# Databricks notebook source
# MAGIC %md
# MAGIC Remove the OpenAI Endpoint

# COMMAND ----------

# MAGIC %pip install mlflow[genai]>=2.9.0
# MAGIC %pip install --upgrade mlflow
# MAGIC %pip install --upgrade langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")
print(client.delete_endpoint("Email-OpenAI-Completion-Endpoint"))

# COMMAND ----------


