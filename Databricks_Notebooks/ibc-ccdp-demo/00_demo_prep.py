# Databricks notebook source
# DBTITLE 1,Step 1: Create catalog and database
_ = spark.sql("CREATE CATALOG IF NOT EXISTS ibc")

# COMMAND ----------

_ = spark.sql("CREATE DATABASE IF NOT EXISTS ibc.ccdp_demo")

# COMMAND ----------

# DBTITLE 1,Step 2: Import csv files and create tables
# MAGIC %md
# MAGIC * Click on Catalog in the left hand nav
# MAGIC * Click on +Add
# MAGIC * Click on Add Data
# MAGIC * Click on Create or modify table
# MAGIC * Drag in subscribers.csv
# MAGIC * Select 'ibc' for catalog <-- change to what you used above
# MAGIC * Select 'ccdp_demo' for database
# MAGIC * Click Create Table
# MAGIC * Repeat these steps for the svod.csv file

# COMMAND ----------

# DBTITLE 1,Step 3: Load data into dataframes
subscribers_df = spark.table('ibc.ccdp_demo.subscribers')
svod_df = spark.table('ibc.ccdp_demo.svod')

# COMMAND ----------

# DBTITLE 1,Step 4: Make adjustments to svod column names
svod_df = (svod_df.withColumnRenamed('Video Start Failure_L30', 'Video_Start_Failure_L30')
           .withColumnRenamed('High Rebuffering_L30', 'High_Rebuffering_L30'))

# COMMAND ----------

svod_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("ibc.ccdp_demo.svod") 
