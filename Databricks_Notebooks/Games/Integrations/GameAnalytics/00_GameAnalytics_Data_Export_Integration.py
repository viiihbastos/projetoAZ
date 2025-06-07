# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Integrating GameAnalytics Data Export
# MAGIC
# MAGIC <img src="./_resources/GameAnalytics Integration - Unity Catalog.png" width="800"/>
# MAGIC
# MAGIC Unity Catalog gives you capabilities to ensure full data governance, storing your main tables in the managed catalog/storage while ensuring secure access for specific cloud storage.
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox 
# MAGIC
# MAGIC ## Working with External Locations
# MAGIC
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/uc/external/uc-external-location.png" style="float:right; margin-left:10px" width="800"/>
# MAGIC
# MAGIC
# MAGIC Accessing external cloud storage is easily done using `External locations`.
# MAGIC
# MAGIC This can be done using 3 simple steps:
# MAGIC
# MAGIC
# MAGIC 1. First, create a Storage credential. It'll contain the IAM role/SP required to access your cloud storage
# MAGIC 2. Create an External location using your Storage credential. It can be any cloud location (a sub folder)
# MAGIC 3. Finally, Grant permissions to your users to access this Storage Credential

# COMMAND ----------

from config import *

# COMMAND ----------

dbutils.widgets.text("S3_PATH", S3_PATH)

# COMMAND ----------

# MAGIC %md-sandbox 
# MAGIC ### Step 1: Create the STORAGE CREDENTIAL
# MAGIC
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/uc/external/uc-external-location-1.png" style="float:right; margin-left:10px" width="700px"/>
# MAGIC
# MAGIC The first step is to create the `STORAGE CREDENTIAL`.
# MAGIC
# MAGIC To do that, we'll use Databricks Unity Catalog UI:
# MAGIC
# MAGIC 1. Open the Data Explorer in DBSQL
# MAGIC 1. Select the "Storage Credential" menu
# MAGIC 1. Click on "Create Credential"
# MAGIC 1. Fill your credential information: the name and IAM role you will be using
# MAGIC
# MAGIC
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/uc/external/uc-external-location-cred.png" width="400"/>

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Insert the storage credential name from the step above
# MAGIC DESCRIBE STORAGE CREDENTIAL `one_env_external_location`

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Create an external location

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE EXTERNAL LOCATION IF NOT EXISTS `game_analytics_data_export`
# MAGIC URL '${S3_PATH}' -- path to the S3 bucket used by GameAnalytics
# MAGIC WITH (STORAGE CREDENTIAL `one_env_external_location`) -- name of storage credential created in Step 1
# MAGIC COMMENT 's3 bucket for GameAnalytics DataSuite';

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Grant access to users

# COMMAND ----------

# MAGIC %sql
# MAGIC GRANT READ FILES, WRITE FILES ON EXTERNAL LOCATION `game_analytics_data_export` TO `account users`;

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Accessing the data
# MAGIC
# MAGIC That's all we have to do! Our users can now access the folder in SQL or python:

# COMMAND ----------

# MAGIC %sql
# MAGIC LIST '${S3_PATH}'
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- The subdirectory may be different. 
# MAGIC SELECT *
# MAGIC FROM json.`${S3_PATH}/raw/2024/09/26/237761`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC With Unity Catalog, you can easily secure access to external locations and grant access based on users/groups.
# MAGIC
# MAGIC This let you operate security at scale, cross workspace, and be ready to build data mesh setups.
