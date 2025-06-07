# Databricks notebook source
# MAGIC %md 
# MAGIC #Ingest Customer Support Emails into the Bronze Delta Table 

# COMMAND ----------

# MAGIC %md
# MAGIC Customer support email application is integrated with Databricks platform. The components such as Azure LogicApps drop data into <a href="https://docs.databricks.com/en/connect/unity-catalog/volumes.html" target="_blank">Databricks Volumes</a>  or directly ingest into Bronze Delta tables. In this notebook, we are assuming that the raw emails are dropped into Volume and we are ingesting data into Bronze Delta Table using Autoloader.
# MAGIC
# MAGIC Data used for this solution are fake emails being generated manually based on the real world emails received by electricity suppliers for their business customers.
# MAGIC
# MAGIC As we are using Autoloader, the solution can be implemented either using batch processing or stream processing.
# MAGIC

# COMMAND ----------

# MAGIC %run ./_resources/00-setup

# COMMAND ----------

# MAGIC %md ## Ingest data from Volume into Bronze Delta table
# MAGIC

# COMMAND ----------

bronzeDF = (spark.readStream \
                .format("cloudFiles")
                .option("cloudFiles.format", "csv")
                .option("cloudFiles.schemaLocation", config['schema_path']) \
                .option("multiLine", "true") \
                .option("escape", "\"") \
                .option("quote","\"") \
                .option("header", "true") \
                .option("inferSchema","true") \
                .option("rescuedDataColumn", "_rescued_data") \
                .load(config['vol_data_landing']))

# COMMAND ----------

display(bronzeDF)

# COMMAND ----------

_ = (
  bronzeDF
    .writeStream
      .format('delta')
      .outputMode('append')
      .option('checkpointLocation', f"{config['checkpoint_path']}/emails_bronze")
      .toTable(config['table_emails_bronze'])
  )

# COMMAND ----------


