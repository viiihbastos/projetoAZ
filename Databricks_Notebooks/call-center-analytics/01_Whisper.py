# Databricks notebook source
# MAGIC %pip install --upgrade transformers

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Download and prepare data
# MAGIC %run ./util/data-prep

# COMMAND ----------

df = spark.read.table("paths_with_ids")
pandas_df = df.toPandas()
pandas_df

# COMMAND ----------

# MAGIC %md
# MAGIC # Load model pipeline

# COMMAND ----------

import torch
from transformers import pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-medium",
  chunk_length_s=30,
  device=device,
)


# COMMAND ----------


sample = pandas_df["path"][0]

prediction = pipe(sample)
print(prediction)

# COMMAND ----------

# MAGIC %md
# MAGIC # Create udf

# COMMAND ----------

broadcast_pipeline = spark.sparkContext.broadcast(pipe)

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import pandas_udf

@pandas_udf("string")
def transcribe_udf(paths: pd.Series) -> pd.Series:
  pipe = broadcast_pipeline.value
  transcriptions = [result['text'] for result in pipe(paths.to_list(), batch_size=1)]
  return pd.Series(transcriptions)

# COMMAND ----------

# MAGIC %md
# MAGIC # Run on sample

# COMMAND ----------

transcribed = df.limit(10).select(df.path, df.id, transcribe_udf(df.path))
transcribed.cache()

# COMMAND ----------

display(transcribed)

# COMMAND ----------

# MAGIC %md
# MAGIC # Run on full data

# COMMAND ----------

# TODO: dynamically set partitions
transcribed = df.repartition(16).select(df.path, df.id, transcribe_udf(df.path).alias('transcription')) 

transcribed.cache()

# COMMAND ----------

transcribed.write.mode("overwrite").saveAsTable("transcriptions")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM transcriptions

# COMMAND ----------

# MAGIC %md
# MAGIC # Using autoloader instead of references

# COMMAND ----------

binary_df = spark.read.table("binary_audio_with_ids")


# COMMAND ----------

one_df = binary_df.limit(1).cache()

# COMMAND ----------

display(one_df)

# COMMAND ----------

one_transcription = one_df.select(one_df.path, one_df.id, transcribe_udf(one_df.content))
display(one_transcription)

# COMMAND ----------


