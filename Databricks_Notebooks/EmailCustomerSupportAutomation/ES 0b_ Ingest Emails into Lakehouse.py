# Databricks notebook source
# MAGIC %run "./ES 0a: Intro & Config"

# COMMAND ----------

# MAGIC %md ## Ingest customer emails into the Bronze Emails table
# MAGIC
# MAGIC Along with our customer, we have created sample emails based on the real world emails received by our electricity supplier for their business customers

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

_ = (
  bronzeDF
    .writeStream
      .format('delta')
      .outputMode('append')
      .option('checkpointLocation', f"{config['checkpoint_path']}/emails_bronze")
      .toTable(config['table_emails_bronze'])
  )

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from email_summary_llm_solution.email_llm.emails_bronze

# COMMAND ----------


