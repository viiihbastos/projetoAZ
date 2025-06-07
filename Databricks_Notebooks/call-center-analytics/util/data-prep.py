# Databricks notebook source
# MAGIC %run ./notebook-config

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Download and extract source data
# MAGIC %sh 
# MAGIC cd /databricks/driver
# MAGIC wget https://www.openslr.org/resources/12/dev-clean.tar.gz
# MAGIC tar -zxvf dev-clean.tar.gz
# MAGIC cp -r LibriSpeech/ /dbfs/tutorials

# COMMAND ----------

# DBTITLE 1,Take a look at the path structure of our source data files
import os
from glob import glob
audio_files = [y for x in os.walk("/dbfs/tutorials/LibriSpeech/dev-clean/") for y in glob(os.path.join(x[0], '*.flac'))]
print(audio_files[:10])

# COMMAND ----------

# DBTITLE 1,Create path dataframe
import pandas as pd
pandas_df = pd.DataFrame(pd.Series(audio_files),columns=["path"])
df = spark.createDataFrame(pandas_df)
display(df.limit(10))

# COMMAND ----------

# DBTITLE 1,Add unique id and write to path table
df_with_ids = df.selectExpr("path", "uuid() as id")
df_with_ids.write.mode("overwrite").saveAsTable("paths_with_ids")

# COMMAND ----------

# MAGIC %md
# MAGIC # Create binary table

# COMMAND ----------

binary_df = spark.readStream.format("cloudFiles") \
  .option("cloudFiles.format", "binaryFile") \
  .option("recursiveFileLookup", "true") \
  .option("pathGlobFilter", "*.flac") \
  .load("/tutorials/LibriSpeech/dev-clean") \
  .repartition(32)

# COMMAND ----------

binary_df = binary_df.selectExpr("*", "uuid() as id")

# COMMAND ----------

binary_df.writeStream.format("delta")\
  .option("checkpointLocation", config["checkpoint_path"])\
  .trigger(once=True)\
  .toTable("binary_audio_with_ids")

# COMMAND ----------

df = spark.read.table("binary_audio_with_ids")
display(df.limit(10))

# COMMAND ----------


