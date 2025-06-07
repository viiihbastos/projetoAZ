# Databricks notebook source
# MAGIC %md
# MAGIC install kaggle spark-nlp==4.0.0 git+https://github.com/databricks-academy/dbacademy@v1.0.13 git+https://github.com/databricks-industry-solutions/notebook-solution-companion@safe-print-html --quiet --disable-pip-version-check

# COMMAND ----------

# MAGIC %md
# MAGIC import re
# MAGIC import os
# MAGIC import json
# MAGIC import sys
# MAGIC
# MAGIC from pyspark.sql import functions as F
# MAGIC from pyspark.sql.window import Window
# MAGIC from pyspark.sql.types import *
# MAGIC
# MAGIC import numpy as np
# MAGIC from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
# MAGIC
# MAGIC from pyspark.ml import Pipeline
# MAGIC from pyspark.ml.classification import RandomForestClassifier,GBTClassifier
# MAGIC from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
# MAGIC from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# MAGIC
# MAGIC import sparknlp
# MAGIC from sparknlp.annotator import *
# MAGIC from sparknlp.common import *
# MAGIC from sparknlp.base import *
# MAGIC
# MAGIC from pyspark.ml import Pipeline
# MAGIC from pyspark.mllib.evaluation import MultilabelMetrics
# MAGIC from pyspark.sql.functions import lit,when,col,array,array_contains,array_remove,regexp_replace,size,when
# MAGIC from pyspark.sql.types import ArrayType,DoubleType,StringType
# MAGIC
# MAGIC from pyspark.ml.evaluation import MultilabelClassificationEvaluator
# MAGIC
# MAGIC import mlflow
# MAGIC import mlflow.spark
# MAGIC from mlflow.tracking import MlflowClient
# MAGIC
# MAGIC from pyspark.sql.functions import col, struct
# MAGIC from pyspark.sql.types import *

# COMMAND ----------

tmpdir = f"/dbfs/tmp/games_solutions/"
tmpdir_dbfs = f"/tmp/games_solutions"
catalog_name = f"games_solutions"
os.environ['tmpdir'] = tmpdir

# COMMAND ----------

# Check if the catalog already exists
catalog_exists = spark.sql(f"SHOW CATALOGS LIKE '{catalog_name}'").count() > 0

# Create the catalog if it does not exist
if not catalog_exists:
    _ = spark.sql(f"CREATE CATALOG {catalog_name}")

# Set catalog
_ = spark.sql(f"USE CATALOG {catalog_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC try:
# MAGIC   mlflow.set_experiment(f"/Users/{useremail}/gaming_experiment") # will try creating experiment if it doesn't exist; but when two notebooks with this code executes at the same time, could trigger a race-condition
# MAGIC except:
# MAGIC   pass

# COMMAND ----------

print(f"tmpdir: {tmpdir}")
print(f"Catalog Name: {catalog_name}")
