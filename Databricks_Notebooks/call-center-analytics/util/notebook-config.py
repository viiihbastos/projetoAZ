# Databricks notebook source
# MAGIC %md ## Set config

# COMMAND ----------

config = {}

# COMMAND ----------

# DBTITLE 1,Streaming checkpoint location
config["checkpoint_path"] = "/tmp/delta/tutorials/librispeech"

# COMMAND ----------

# DBTITLE 1,Database settings
config["database"] = "libri"
spark.sql(f"""create database if not exists {config["database"]}""")
spark.sql(f"""use {config["database"]}""")

# COMMAND ----------

# DBTITLE 1,mlflow settings
import mlflow
config["model_name"] = "libri"
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment('/Users/{}/libri'.format(username))

# COMMAND ----------

print(f"Set config: ", config)
