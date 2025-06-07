# Databricks notebook source
import dlt
from config import *
from pyspark.sql import functions as F 
from pyspark.sql.types import LongType, FloatType, StringType, TimestampType, BooleanType, StructType, StructField, DoubleType, DateType, IntegerType, ArrayType

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # Overview
# MAGIC
# MAGIC The Medallion Architecture is a data design pattern used to logically organize data in a lakehouse, with the goal of incrementally and progressively improving the structure and quality of data as it flows through each layer of the architecture. This architecture is divided into three layers: Bronze, Silver, and Gold
# MAGIC
# MAGIC - **Bronze Layer**: This layer is for raw data ingestion, storing data in its native format without any transformations. It acts as a dumping ground for raw data, ensuring no information is lost and providing a historical archive
# MAGIC
# MAGIC - **Silver Layer**: In this layer, data is cleansed and conformed. Minimal transformations and data cleansing rules are applied to ensure data quality and consistency. The Silver layer typically has more normalized data models, making it suitable for further processing and analysis.
# MAGIC
# MAGIC - **Gold Layer**: This layer contains curated, business-level aggregates of the Silver data. The data is in a format suitable for individual business projects or reports, often using de-normalized and read-optimized data models. The final layer of data transformations and data quality rules are applied here, making the data ready for consumption by BI tools and ML models.
# MAGIC
# MAGIC The benefits of using the Medallion Architecture for ELT pipelines include improved data quality, scalability, and query performance. By organizing data into optimized layers, the architecture ensures faster data retrieval and analysis, simplifies data governance, and provides granular access controls to ensure data security and compliance.
# MAGIC
# MAGIC This diagram shows how we will use the mediallian architecture to process data that is exported from GameAnlaytics to cloud storage.  We're going to add a **Raw** layer at the beginning of the pipeline to preserve the raw JSON as a string.
# MAGIC
# MAGIC <img src="./_resources/GameAnalytics Integration - Delta Live Tables.png" width="800"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Raw
# MAGIC
# MAGIC Loading data without transformations is the first step of a medallion ETL architecture, which ensures data integrity and provides a single source of truth. It allows data engineers to capture the original state of the data, which is crucial for auditing, debugging, and reprocessing if needed. Additionally, adding file metadata in the first table of a medallion ETL architecture provides significant benefits to data engineers. By capturing information like load timestamps, file paths, and filenames, engineers can easily track data to the source, troubleshoot issues, and enable efficient incremental processing in subsequent stages. This approach preserves the raw data's integrity while enhancing traceability and facilitating more robust data management throughout the pipeline.

# COMMAND ----------

raw_schema = StructType([
  StructField("value", StringType(), True, {"comment": "Payload of individual player events that GameAnalytics received and processes for your game"}),
  StructField("source_metadata", StructType([
        StructField("file_path", StringType(), True, {"comment": "File path of the input file."}),
        StructField("file_name", StringType(), True, {"comment": "Name of the input file along with its extension."}),
        StructField("file_size", LongType(), True, {"comment": "Length of the input file, in bytes."}),
        StructField("file_block_start", LongType(), True, {"comment": "Start offset of the block being read, in bytes."}),
        StructField("file_block_length", LongType(), True, {"comment": "Length of the block being read, in bytes."}),
        StructField("file_modification_time", TimestampType(), True, {"comment": "Last modification timestamp of the input file."})
    ]), True, {"comment": "Metadata information for input files"})
])

raw_comment = """
Payload data from GameAnalytics.
"""

@dlt.table(
  name="raw",
  comment=raw_comment,
  schema=raw_schema
  )
def raw():
  return (
    spark.readStream.format("cloudFiles")
      .option("cloudFiles.format", "text")
      .load(S3_PATH)
      .withColumn("source_metadata", F.col("_metadata"))
      )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze
# MAGIC
# MAGIC GameAnalytics provides the schemas for each event type, which can be accessed at this link. By leveraging these predefined schemas, data engineers can define the schema in Delta Live Tables (DLT). Enforcing a schema during the data ingestion process ensures data consistency and quality by validating incoming data against these predefined structures. This practice helps prevent errors and anomalies, making it easier to detect and resolve data issues early in the pipeline. Additionally, it facilitates better data governance and compliance by ensuring that all ingested data adheres to the expected format and standards, which is crucial for maintaining reliable and accurate datasets.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Event

# COMMAND ----------

payload_schema = StructType([
    StructField("user_meta", StringType(), True), # will be variant type
    StructField("ip", StringType(), True),
    StructField("game_id", LongType(), True),
    StructField("first_in_batch", BooleanType(), True),
    StructField("data", StringType(), True), # will be variant type
    StructField("country_code", StringType(), True),
    StructField("arrival_ts", LongType(), True)
])

events_schema = StructType([
  StructField("user_meta", StringType(), True, {"comment": "Player metadata for the game_id"}), # will be variant type
  StructField("ip", StringType(), True, {"comment": "IP address of the player"}),
  StructField("game_id", LongType(), True, {"comment": "Game's unique identifier"}),
  StructField("first_in_batch", BooleanType(), True, {"comment": ""}),
  StructField("data", StringType(), True, {"comment": "Event data"}), # will be variant type
  StructField("country_code", StringType(), True, {"comment": "Country code for the player's country based on events (please note this may change day on day if the player is travelling)"}),
  StructField("arrival_ts", LongType(), True, {"comment": "Timestamp for which the event arrived at GA (discrepancy might be for users being offline, for example)"}),
  StructField("file_path", StringType(), True, {"comment": "File path of the input file."}),
  StructField("file_modification_time", TimestampType(), True, {"comment": "Last modification timestamp of the input file."})
])


events_comment = """
The events table parses out top level keys of the JSON payload.
"""  

@dlt.table(
  name="events",
  comment=events_comment,
  schema=events_schema
  )
def events():
  return (
    dlt.read_stream("raw")
      .withColumn("value", F.from_json("value", payload_schema))
      .withColumn("user_meta", F.col("value.user_meta"))
      .withColumn("ip", F.col("value.ip"))
      .withColumn("game_id", F.col("value.game_id"))
      .withColumn("first_in_batch", F.col("value.first_in_batch"))
      .withColumn("data", F.col("value.data"))
      .withColumn("country_code", F.col("value.country_code"))
      .withColumn("arrival_ts", F.col("value.arrival_ts"))
      .withColumn("file_path", F.col("source_metadata.file_path"))
      .withColumn("file_modification_time", F.col("source_metadata.file_modification_time"))
      .drop("value", "source_metadata")
          )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver
# MAGIC
# MAGIC For the Silver layer, we want to clean, normalize, and merge data from the Bronze layer just enough that Data Scientists and Data Analysts can build fit-for-purpose datasets for their AI/ML models and BI dashboards.  Examples of this are
# MAGIC - Data Cleaning: Temporal Fields are exported by GameAnalytics in Unix time, so for a better developer experience and more accessible for downstream usage in BI tools, we are going to convert that to a timestamp
# MAGIC - Data Quality Enforcement: DLT expectations apply data quality checks on each record passing through a pipeline.  For fields such as progression_status where we know the categories, we can define those quality checks in code using the @dlt.expectations function.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Session End

# COMMAND ----------

session_end_user_meta_schema = StructType([
    StructField("attribution_partner", StringType(), True),
    StructField("cohort_month", LongType(), True),
    StructField("cohort_week", LongType(), True),
    StructField("first_build", StringType(), True),
    StructField("install_campaign", StringType(), True),
    StructField("install_hour", LongType(), True),
    StructField("install_adgroup", StringType(), True),
    StructField("install_publisher", StringType(), True),
    StructField("install_site", StringType(), True),
    StructField("install_ts", LongType(), True),
    StructField("is_converting", BooleanType(), True),
    StructField("is_paying", BooleanType(), True),
    StructField("origin", StringType(), True),
    StructField("pay_ft", LongType(), True),
    StructField("revenue", DoubleType(), True)
])

session_end_data_schema = StructType([
    StructField("v", IntegerType(), True),
    StructField("user_id", StringType(), True),
    StructField("session_num", IntegerType(), True),
    StructField("session_id", StringType(), True),
    StructField("sdk_version", StringType(), True),
    StructField("platform", StringType(), True),
    StructField("os_version", StringType(), True),
    StructField("manufacturer", StringType(), True),
    StructField("length", LongType(), True),
    StructField("device", StringType(), True),
    StructField("custom_01", StringType(), True),
    StructField("custom_02", StringType(), True),
    StructField("custom_03", StringType(), True),
    StructField("client_ts", LongType(), True),
    StructField("category", StringType(), True),
    StructField("build", StringType(), True)
])

session_end_schema = StructType([
    StructField("ip", StringType(), True, {"comment": "IP address of the player"}),
    StructField("game_id", LongType(), True, {"comment": "Game's unique identifier"}),
    StructField("first_in_batch", BooleanType(), True, {"comment": ""}),
    StructField("country_code", StringType(), True, {"comment": "Country code for the player's country based on events (please note this may change day on day if the player is travelling)"}),
    StructField("arrival_ts", TimestampType(), True, {"comment": "Timestamp for which the event arrived at GA (discrepancy might be for users being offline, for example)"}),
    StructField("install_campaign", StringType(), True, {"comment": ""}),
    StructField("install_adgroup", StringType(), True, {"comment": ""}),
    StructField("install_publisher", StringType(), True, {"comment": ""}),
    StructField("install_site", StringType(), True, {"comment": ""}),
    StructField("is_paying", BooleanType(), True, {"comment": ""}),
    StructField("origin", StringType(), True, {"comment": ""}),
    StructField("pay_ft", LongType(), True, {"comment": ""}),
    StructField("site_id", StringType(), True, {"comment": ""}),
    StructField("attribution_partner", StringType(), True, {"comment": ""}),
    StructField("revenue", DoubleType(), True, {"comment": ""}),
    StructField("cohort_month", StringType(), True, {"comment": "First day of the month the player installed the game	"}),
    StructField("is_converting", BooleanType(), True, {"comment": "Flag indicating whether it's the first time the player is making a payment (since we have history of it)	"}),
    StructField("cohort_week", StringType(), True, {"comment": "First day of the week the player installed the game	"}),
    StructField("first_build", StringType(), True, {"comment": ""}),
    StructField("install_ts", TimestampType(), True, {"comment": "Date the player installed the game"}),
    StructField("install_hour", TimestampType(), True, {"comment": ""}),
    StructField("session_id", StringType(), True, {"comment": "Session's unique identifier"}),
    StructField("os_version", StringType(), True, {"comment": "Device's OS version"}),
    StructField("client_ts", TimestampType(), True, {"comment": "Timestamp for which the event occurred"}),
    StructField("session_num", IntegerType(), True, {"comment": "Session number for that player"}),
    StructField("build", StringType(), True, {"comment": "Game build"}),
    StructField("user_id", StringType(), True, {"comment": "Device identifier of the player (note the same user_id might be linked to multiple game_ids)	"}),
    StructField("v", IntegerType(), True, {"comment": "Reflects the version of events coming in to the collectors."}),
    StructField("custom_01", StringType(), True, {"comment": "Custom field 1"}),
    StructField("custom_02", StringType(), True, {"comment": "Custom field 2"}),
    StructField("custom_03", StringType(), True, {"comment": "Custom field 3"}),
    StructField("category", StringType(), True, {"comment": ""}),
    StructField("sdk_version", StringType(), True, {"comment": "SDK version"}),
    StructField("length", LongType(), True, {"comment": "Length of that session in seconds"}),
    StructField("manufacturer", StringType(), True, {"comment": "Device's manufacturer"}),
    StructField("platform", StringType(), True, {"comment": "Platform e.g. ios, android	"}),
    StructField("device", StringType(), True, {"comment": "Device model"})
])

session_end_comment = """
Whenever a session is determined to be over the code should **always** attempt to add a session end event and submit all pending events immediately.

Only **one** session end event per session should be activated.

Refer to GameAnalytics [documentation](https://restapidocs.gameanalytics.com/?_gl=1*1bty7ed*_ga*MTQ5NjQzOTMwMy4xNzE0NzQ5NDY2*_ga_ML11XZTFE7*MTcyNjkyODgzNS4zMC4xLjE3MjY5MjkxNzYuMC4wLjA.#session-end) for more details.
"""  

@dlt.table(
  name="session_end",
  comment=session_end_comment,
  schema=session_end_schema
  )
def session_end():
  return (
    dlt.read_stream("events")
    .filter(F.col("data").contains('"category":"session_end"'))
    .withColumn("user_meta", F.from_json("user_meta", session_end_user_meta_schema))
    .withColumn("data", F.from_json("data", session_end_data_schema))
    .withColumn("arrival_ts", F.to_timestamp(F.from_unixtime(F.col("arrival_ts"))))
    .withColumn("install_campaign", F.col("user_meta.install_campaign"))
    .withColumn("install_site", F.col("user_meta.install_site"))
    .withColumn("is_paying", F.col("user_meta.is_paying"))
    .withColumn("origin", F.col("user_meta.origin"))
    .withColumn("pay_ft", F.col("user_meta.pay_ft"))
    .withColumn("attribution_partner", F.col("user_meta.attribution_partner"))
    .withColumn("revenue", F.col("user_meta.revenue"))
    .withColumn("cohort_month", F.from_unixtime(F.col("user_meta.cohort_month"), "yyyy-MM"))
    .withColumn("is_converting", F.col("user_meta.is_converting"))
    .withColumn("cohort_week", F.from_unixtime(F.col("user_meta.cohort_week")))
    .withColumn("first_build", F.col("user_meta.first_build"))
    .withColumn("install_ts", F.to_timestamp(F.from_unixtime(F.col("user_meta.install_ts"))))
    .withColumn("install_hour", F.to_timestamp(F.from_unixtime(F.col("user_meta.install_hour"))))
    .withColumn("session_id", F.col("data.session_id"))
    .withColumn("os_version", F.col("data.os_version"))
    .withColumn("client_ts", F.to_timestamp(F.from_unixtime(F.col("data.client_ts"))))
    .withColumn("session_num", F.col("data.session_num"))
    .withColumn("build", F.col("data.build"))
    .withColumn("user_id", F.col("data.user_id"))
    .withColumn("v", F.col("data.v"))
    .withColumn("custom_01", F.col("data.custom_01"))
    .withColumn("custom_02", F.col("data.custom_02"))
    .withColumn("custom_03", F.col("data.custom_03"))
    .withColumn("category", F.col("data.category"))
    .withColumn("sdk_version", F.col("data.sdk_version"))
    .withColumn("length", F.col("data.length"))
    .withColumn("manufacturer", F.col("data.manufacturer"))
    .withColumn("platform", F.col("data.platform"))
    .withColumn("device", F.col("data.device"))
    .drop("file_path","file_modification_time","user_meta","data")
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ### User

# COMMAND ----------

user_user_meta_schema = StructType([
    StructField("attribution_partner", StringType(), True),
    StructField("cohort_month", LongType(), True),
    StructField("cohort_week", LongType(), True),
    StructField("first_build", StringType(), True),
    StructField("install_campaign", StringType(), True),
    StructField("install_hour", LongType(), True),
    StructField("install_adgroup", StringType(), True),
    StructField("install_publisher", StringType(), True),
    StructField("install_site", StringType(), True),
    StructField("install_ts", LongType(), True),
    StructField("is_converting", BooleanType(), True),
    StructField("is_paying", BooleanType(), True),
    StructField("origin", StringType(), True),
    StructField("pay_ft", LongType(), True),
    StructField("revenue", DoubleType(), True)
])

user_data_schema = StructType([
    StructField("v", IntegerType(), True),
    StructField("user_id", StringType(), True),
    StructField("session_num", IntegerType(), True),
    StructField("session_id", StringType(), True),
    StructField("sdk_version", StringType(), True),
    StructField("platform", StringType(), True),
    StructField("os_version", StringType(), True),
    StructField("manufacturer", StringType(), True),
    StructField("device", StringType(), True),
    StructField("custom_01", StringType(), True),
    StructField("custom_02", StringType(), True),
    StructField("custom_03", StringType(), True),
    StructField("client_ts", LongType(), True),
    StructField("category", StringType(), True),
    StructField("build", StringType(), True),
])

user_schema = StructType([
    StructField("ip", StringType(), True, {"comment": "IP address of the player"}),
    StructField("game_id", LongType(), True, {"comment": "Game's unique identifier"}),
    StructField("first_in_batch", BooleanType(), True, {"comment": ""}),
    StructField("country_code", StringType(), True, {"comment": "Country code for the player's country based on events (please note this may change day on day if the player is travelling)"}),
    StructField("arrival_ts", TimestampType(), True, {"comment": "Timestamp for which the event arrived at GA (discrepancy might be for users being offline, for example)"}),
    StructField("install_campaign", StringType(), True, {"comment": ""}),
    StructField("install_adgroup", StringType(), True, {"comment": ""}),
    StructField("install_publisher", StringType(), True, {"comment": ""}),
    StructField("install_site", StringType(), True, {"comment": ""}),
    StructField("is_paying", BooleanType(), True, {"comment": ""}),
    StructField("origin", StringType(), True, {"comment": ""}),
    StructField("pay_ft", LongType(), True, {"comment": ""}),
    StructField("site_id", StringType(), True, {"comment": ""}),
    StructField("attribution_partner", StringType(), True, {"comment": ""}),
    StructField("revenue", DoubleType(), True, {"comment": ""}),
    StructField("cohort_month", StringType(), True, {"comment": "First day of the month the player installed the game	"}),
    StructField("is_converting", BooleanType(), True, {"comment": "Flag indicating whether it's the first time the player is making a payment (since we have history of it)	"}),
    StructField("cohort_week", StringType(), True, {"comment": "First day of the week the player installed the game	"}),
    StructField("first_build", StringType(), True, {"comment": ""}),
    StructField("install_ts", TimestampType(), True, {"comment": "Date the player installed the game"}),
    StructField("install_hour", TimestampType(), True, {"comment": ""}),
    StructField("session_id", StringType(), True, {"comment": "Session's unique identifier"}),
    StructField("os_version", StringType(), True, {"comment": "Device's OS version"}),
    StructField("client_ts", TimestampType(), True, {"comment": "Timestamp for which the event occurred"}),
    StructField("session_num", IntegerType(), True, {"comment": "Session number for that player"}),
    StructField("build", StringType(), True, {"comment": "Game build"}),
    StructField("user_id", StringType(), True, {"comment": "Device identifier of the player (note the same user_id might be linked to multiple game_ids)	"}),
    StructField("v", IntegerType(), True, {"comment": "Reflects the version of events coming in to the collectors."}),
    StructField("custom_01", StringType(), True, {"comment": "Custom field 1"}),
    StructField("custom_02", StringType(), True, {"comment": "Custom field 2"}),
    StructField("custom_03", StringType(), True, {"comment": "Custom field 3"}),
    StructField("category", StringType(), True, {"comment": ""}),
    StructField("sdk_version", StringType(), True, {"comment": "SDK version"}),
    StructField("manufacturer", StringType(), True, {"comment": "Device's manufacturer"}),
    StructField("platform", StringType(), True, {"comment": "Platform e.g. ios, android	"}),
    StructField("device", StringType(), True, {"comment": "Device model"})
])

user_comment = """
As session is the concept of **a user spending a period of time focused on a game**.

The user event acts like a session start. It should **always** be the first event in the first batch sent to the collectors and added each time a session starts.

Refer to GameAnalytics [documentation](https://restapidocs.gameanalytics.com/?json#user-session-start) for more details.
"""  

@dlt.table(
  name="user",
  comment=user_comment,
  schema=user_schema
  )
def user():
  return (
    dlt.read_stream("events")
    .filter(F.col("data").contains('category":"user"'))
    .withColumn("user_meta", F.from_json("user_meta", user_user_meta_schema))
    .withColumn("data", F.from_json("data", user_data_schema))
    .withColumn("arrival_ts", F.to_timestamp(F.from_unixtime(F.col("arrival_ts"))))
    .withColumn("install_campaign", F.col("user_meta.install_campaign"))
    .withColumn("install_site", F.col("user_meta.install_site"))
    .withColumn("is_paying", F.col("user_meta.is_paying"))
    .withColumn("origin", F.col("user_meta.origin"))
    .withColumn("pay_ft", F.col("user_meta.pay_ft"))
    .withColumn("attribution_partner", F.col("user_meta.attribution_partner"))
    .withColumn("revenue", F.col("user_meta.revenue"))
    .withColumn("cohort_month", F.from_unixtime(F.col("user_meta.cohort_month"), "yyyy-MM"))
    .withColumn("is_converting", F.col("user_meta.is_converting"))
    .withColumn("cohort_week", F.from_unixtime(F.col("user_meta.cohort_week")))
    .withColumn("first_build", F.col("user_meta.first_build"))
    .withColumn("install_ts", F.to_timestamp(F.from_unixtime(F.col("user_meta.install_ts"))))
    .withColumn("install_hour", F.to_timestamp(F.from_unixtime(F.col("user_meta.install_hour"))))
    .withColumn("session_id", F.col("data.session_id"))
    .withColumn("os_version", F.col("data.os_version"))
    .withColumn("client_ts", F.to_timestamp(F.from_unixtime(F.col("data.client_ts"))))
    .withColumn("session_num", F.col("data.session_num"))
    .withColumn("build", F.col("data.build"))
    .withColumn("user_id", F.col("data.user_id"))
    .withColumn("v", F.col("data.v"))
    .withColumn("custom_01", F.col("data.custom_01"))
    .withColumn("custom_02", F.col("data.custom_02"))
    .withColumn("custom_03", F.col("data.custom_03"))
    .withColumn("category", F.col("data.category"))
    .withColumn("sdk_version", F.col("data.sdk_version"))
    .withColumn("manufacturer", F.col("data.manufacturer"))
    .withColumn("platform", F.col("data.platform"))
    .withColumn("device", F.col("data.device"))
    .drop("file_path","file_modification_time","user_meta","data")
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Progression

# COMMAND ----------

progression_user_meta_schema = StructType([
    StructField("attribution_partner", StringType(), True),
    StructField("cohort_month", LongType(), True),
    StructField("cohort_week", LongType(), True),
    StructField("first_build", StringType(), True),
    StructField("install_campaign", StringType(), True),
    StructField("install_hour", LongType(), True),
    StructField("install_adgroup", StringType(), True),
    StructField("install_publisher", StringType(), True),
    StructField("install_site", StringType(), True),
    StructField("install_ts", LongType(), True),
    StructField("is_converting", BooleanType(), True),
    StructField("is_paying", BooleanType(), True),
    StructField("origin", StringType(), True),
    StructField("pay_ft", LongType(), True),
    StructField("revenue", DoubleType(), True)
])

progression_data_schema = StructType([
    StructField("v", IntegerType(), True),
    StructField("user_id", StringType(), True),
    StructField("session_num", IntegerType(), True),
    StructField("session_id", StringType(), True),
    StructField("sdk_version", StringType(), True),
    StructField("platform", StringType(), True),
    StructField("os_version", StringType(), True),
    StructField("manufacturer", StringType(), True),
    StructField("device", StringType(), True),
    StructField("client_ts", LongType(), True),
    StructField("category", StringType(), True),
    StructField("build", StringType(), True),
    StructField("ab_id", StringType(), True),
    StructField("ab_variant_id", StringType(), True),
    StructField("android_app_build", StringType(), True),
    StructField("android_app_signature", StringType(), True),
    StructField("android_app_version", StringType(), True),
    StructField("android_bundle_id", StringType(), True),
    StructField("android_channel_id", StringType(), True),
    StructField("android_id", StringType(), True),
    StructField("android_mac_md5", StringType(), True),
    StructField("android_mac_sha1", StringType(), True),
    StructField("attempt_num", LongType(), True),
    StructField("configuration_keys", ArrayType(StringType()), True),
    StructField("configurations", ArrayType(StringType()), True),
    StructField("connection_type", StringType(), True),
    StructField("engine_version", StringType(), True),
    StructField("event_id", StringType(), True),
    StructField("google_aid", StringType(), True),
    StructField("google_aid_src", StringType(), True),
    StructField("ios_app_build", StringType(), True),
    StructField("ios_app_version", StringType(), True),
    StructField("ios_att", StringType(), True),
    StructField("ios_bundle_id", StringType(), True),
    StructField("ios_idfa", StringType(), True),
    StructField("ios_idfv", StringType(), True),
    StructField("ios_testflight", BooleanType(), True),
    StructField("jailbroken", BooleanType(), True),
    StructField("limited_ad_tracking", BooleanType(), True),
    StructField("oaid", StringType(), True),
    StructField("score", LongType(), True),
    StructField("user_id_ext", StringType(), True),
    StructField("progression_status", StringType(), True),
    StructField("progression1", StringType(), True),
    StructField("progression2", StringType(), True),
    StructField("progression3", StringType(), True),
    StructField("custom_01", StringType(), True),
    StructField("custom_02", StringType(), True),
    StructField("custom_03", StringType(), True),
])

progression_schema = StructType([
    StructField("ip", StringType(), True, {"comment": "IP address of the player"}),
    StructField("game_id", LongType(), True, {"comment": "Game's unique identifier"}),
    StructField("first_in_batch", BooleanType(), True, {"comment": ""}),
    StructField("country_code", StringType(), True, {"comment": "Country code for the player's country based on events (please note this may change day on day if the player is travelling)"}),
    StructField("arrival_ts", TimestampType(), True, {"comment": "Timestamp for which the event arrived at GA (discrepancy might be for users being offline, for example)"}),
    StructField("install_campaign", StringType(), True, {"comment": ""}),
    StructField("install_adgroup", StringType(), True, {"comment": ""}),
    StructField("install_publisher", StringType(), True, {"comment": ""}),
    StructField("install_site", StringType(), True, {"comment": ""}),
    StructField("is_paying", BooleanType(), True, {"comment": ""}),
    StructField("origin", StringType(), True, {"comment": ""}),
    StructField("pay_ft", LongType(), True, {"comment": ""}),
    StructField("site_id", StringType(), True, {"comment": ""}),
    StructField("attribution_partner", StringType(), True, {"comment": ""}),
    StructField("revenue", DoubleType(), True, {"comment": ""}),
    StructField("cohort_month", StringType(), True, {"comment": "First day of the month the player installed the game	"}),
    StructField("is_converting", BooleanType(), True, {"comment": "Flag indicating whether it's the first time the player is making a payment (since we have history of it)	"}),
    StructField("cohort_week", StringType(), True, {"comment": "First day of the week the player installed the game	"}),
    StructField("first_build", StringType(), True, {"comment": ""}),
    StructField("install_ts", TimestampType(), True, {"comment": "Date the player installed the game"}),
    StructField("install_hour", TimestampType(), True, {"comment": ""}),
    StructField("session_id", StringType(), True, {"comment": "Session's unique identifier"}),
    StructField("os_version", StringType(), True, {"comment": "Device's OS version"}),
    StructField("client_ts", TimestampType(), True, {"comment": "Timestamp for which the event occurred"}),
    StructField("session_num", IntegerType(), True, {"comment": "Session number for that player"}),
    StructField("build", StringType(), True, {"comment": "Game build"}),
    StructField("user_id", StringType(), True, {"comment": "Device identifier of the player (note the same user_id might be linked to multiple game_ids)	"}),
    StructField("v", IntegerType(), True, {"comment": "Reflects the version of events coming in to the collectors."}),
    StructField("category", StringType(), True, {"comment": ""}),
    StructField("sdk_version", StringType(), True, {"comment": "SDK version"}),
    StructField("manufacturer", StringType(), True, {"comment": "Device's manufacturer"}),
    StructField("platform", StringType(), True, {"comment": "Platform e.g. ios, android	"}),
    StructField("device", StringType(), True, {"comment": "Device model"}),
    StructField("ab_id", StringType(), True, {"comment": "A/B Testing experiment identifier in case the player is participating in an A/B Test"}),
    StructField("ab_variant_id", StringType(), True, {"comment": "A/B Testing variant identifier in case the player is participating in an A/B Test"}),
    StructField("android_app_build", StringType(), True, {"comment": ""}),
    StructField("android_app_signature", StringType(), True, {"comment": ""}),
    StructField("android_app_version", StringType(), True, {"comment": ""}),
    StructField("android_bundle_id", StringType(), True, {"comment": ""}),
    StructField("android_channel_id", StringType(), True, {"comment": ""}),
    StructField("android_id", StringType(), True, {"comment": "Android id	"}), #find a better defintion
    StructField("android_mac_md5", StringType(), True, {"comment": ""}),
    StructField("android_mac_sha1", StringType(), True, {"comment": ""}),
    StructField("attempt_num", LongType(), True, {"comment": "The number of attempts for this event id (event_id e.g. \"Fail:Universe1:Planet1:Quest1\")"}),
    StructField("configuration_keys", ArrayType(StringType()), True, {"comment": ""}),
    StructField("configurations", ArrayType(StringType()), True, {"comment": ""}),
    StructField("connection_type", StringType(), True, {"comment": "connection, e.g. lan, wwan, wifi, offline	"}),
    StructField("engine_version", StringType(), True, {"comment": "engine version"}),
    StructField("google_aid", StringType(), True, {"comment": "Android advertising id	"}),
    StructField("google_aid_src", StringType(), True, {"comment": ""}),
    StructField("ios_app_build", StringType(), True, {"comment": ""}),
    StructField("ios_app_version", StringType(), True, {"comment": ""}),
    StructField("ios_att", StringType(), True, {"comment": ""}),
    StructField("ios_bundle_id", StringType(), True, {"comment": ""}),
    StructField("ios_idfa", StringType(), True, {"comment": "IOS identifier for advertisers"}),
    StructField("ios_idfv", StringType(), True, {"comment": "IOS identifier for vendor"}),
    StructField("ios_testflight", BooleanType(), True, {"comment": ""}),
    StructField("jailbroken", BooleanType(), True, {"comment": "whether the player has jailbreaking (process of removing all restrictions imposed on an IOS device) enabled"}),
    StructField("limited_ad_tracking", BooleanType(), True, {"comment": "if True, it means the player does not want to be targeted, preventing attribution of installs to any advertising source"}),
    StructField("oaid", StringType(), True, {"comment": ""}),
    StructField("score", LongType(), True, {"comment": "An optional player score for attempt. Only sent when Status is “Fail” or “Complete”."}),
    StructField("user_id_ext", StringType(), True, {"comment": ""}),
    StructField("progression_status", StringType(), True, {"comment": ""}),
    StructField("progression_1", StringType(), True, {"comment": ""}),
    StructField("progression_2", StringType(), True, {"comment": ""}),
    StructField("progression_3", StringType(), True, {"comment": ""}),
    StructField("custom_01", StringType(), True, {"comment": "Custom field 1"}),
    StructField("custom_02", StringType(), True, {"comment": "Custom field 2"}),
    StructField("custom_03", StringType(), True, {"comment": "Custom field 3"}),
    
])

progression_comment = """
Level attempts with Start, Fail & Complete event.

Refer to GameAnalytics [documentation](https://docs.gameanalytics.com/event-types/progression-events) for more details.
"""  

@dlt.table(
  name="progression",
  comment=progression_comment,
  schema=progression_schema
  )
@dlt.expect_or_fail("valid_progression_status", "progression_status IN (\"Start\", \"Complete\", \"Fail\")")
def progression():
  return (
    dlt.read_stream("events")
    .filter(F.col("data").contains('"category":"progression"'))
    .withColumn("user_meta", F.from_json("user_meta", progression_user_meta_schema))
    .withColumn("data", F.from_json("data", progression_data_schema))
    .withColumn("arrival_ts", F.to_timestamp(F.from_unixtime(F.col("arrival_ts"))))
    .withColumn("install_campaign", F.col("user_meta.install_campaign"))
    .withColumn("install_adgroup", F.col("user_meta.install_adgroup"))
    .withColumn("install_publisher", F.col("user_meta.install_publisher"))
    .withColumn("install_site", F.col("user_meta.install_site"))
    .withColumn("is_paying", F.col("user_meta.is_paying"))
    .withColumn("origin", F.col("user_meta.origin"))
    .withColumn("pay_ft", F.col("user_meta.pay_ft"))
    .withColumn("attribution_partner", F.col("user_meta.attribution_partner"))
    .withColumn("revenue", F.col("user_meta.revenue"))
    .withColumn("cohort_month", F.from_unixtime(F.col("user_meta.cohort_month"), "yyyy-MM"))
    .withColumn("is_converting", F.col("user_meta.is_converting"))
    .withColumn("cohort_week", F.from_unixtime(F.col("user_meta.cohort_week")))
    .withColumn("first_build", F.col("user_meta.first_build"))
    .withColumn("install_ts", F.to_timestamp(F.from_unixtime(F.col("user_meta.install_ts"))))
    .withColumn("install_hour", F.to_timestamp(F.from_unixtime(F.col("user_meta.install_hour"))))
    .withColumn("session_id", F.col("data.session_id"))
    .withColumn("os_version", F.col("data.os_version"))
    .withColumn("client_ts", F.to_timestamp(F.from_unixtime(F.col("data.client_ts"))))
    .withColumn("session_num", F.col("data.session_num"))
    .withColumn("build", F.col("data.build"))
    .withColumn("user_id", F.col("data.user_id"))
    .withColumn("v", F.col("data.v"))
    .withColumn("category", F.col("data.category"))
    .withColumn("sdk_version", F.col("data.sdk_version"))
    .withColumn("manufacturer", F.col("data.manufacturer"))
    .withColumn("platform", F.col("data.platform"))
    .withColumn("device", F.col("data.device"))
    .withColumn("ab_id", F.col("data.ab_id"))
    .withColumn("ab_variant_id", F.col("data.ab_variant_id"))
    .withColumn("android_app_build", F.col("data.android_app_build"))
    .withColumn("android_app_signature", F.col("data.android_app_signature"))
    .withColumn("android_app_version", F.col("data.android_app_version"))
    .withColumn("android_bundle_id", F.col("data.android_bundle_id"))
    .withColumn("android_channel_id", F.col("data.android_channel_id"))
    .withColumn("android_id", F.col("data.android_id"))
    .withColumn("android_mac_md5", F.col("data.android_mac_md5"))
    .withColumn("android_mac_sha1", F.col("data.android_mac_sha1"))
    .withColumn("attempt_num", F.col("data.attempt_num"))
    .withColumn("configuration_keys", F.col("data.configuration_keys"))
    .withColumn("configurations", F.col("data.configurations"))
    .withColumn("connection_type", F.col("data.connection_type"))
    .withColumn("engine_version", F.col("data.engine_version"))
    .withColumn("google_aid", F.col("data.google_aid"))
    .withColumn("google_aid_src", F.col("data.google_aid_src"))
    .withColumn("ios_app_build", F.col("data.ios_app_build"))
    .withColumn("ios_app_version", F.col("data.ios_app_version"))
    .withColumn("ios_att", F.col("data.ios_att"))
    .withColumn("ios_bundle_id", F.col("data.ios_bundle_id"))
    .withColumn("ios_idfa", F.col("data.ios_idfa"))
    .withColumn("ios_idfv", F.col("data.ios_idfv"))
    .withColumn("ios_testflight", F.col("data.ios_testflight"))
    .withColumn("jailbroken", F.col("data.jailbroken"))
    .withColumn("limited_ad_tracking", F.col("data.limited_ad_tracking"))
    .withColumn("oaid", F.col("data.oaid"))
    .withColumn("score", F.col("data.score"))
    .withColumn("user_id_ext", F.col("data.user_id_ext"))
    .withColumn("event_id_split",F.split(F.col("data.event_id"), ":"))
    .withColumn("progression_status", F.col("event_id_split").getItem(0))
    .withColumn("progression_1", F.col("event_id_split").getItem(1))
    .withColumn("progression_2", F.col("event_id_split").getItem(2))
    .withColumn("progression_3", F.col("event_id_split").getItem(3))
    .withColumn("custom_01", F.col("data.custom_01"))
    .withColumn("custom_02", F.col("data.custom_02"))
    .withColumn("custom_03", F.col("data.custom_03"))
    .drop("file_path","file_modification_time","user_meta","data","event_id_split")
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ads

# COMMAND ----------

ads_user_meta_schema = StructType([
    StructField("attribution_partner", StringType(), True),
    StructField("cohort_month", LongType(), True),
    StructField("cohort_week", LongType(), True),
    StructField("first_build", StringType(), True),
    StructField("install_campaign", StringType(), True),
    StructField("install_hour", LongType(), True),
    StructField("install_adgroup", StringType(), True),
    StructField("install_publisher", StringType(), True),
    StructField("install_site", StringType(), True),
    StructField("install_ts", LongType(), True),
    StructField("is_converting", BooleanType(), True),
    StructField("is_paying", BooleanType(), True),
    StructField("origin", StringType(), True),
    StructField("pay_ft", LongType(), True),
    StructField("revenue", DoubleType(), True)
])

ads_data_schema = StructType([
    StructField("ab_id", StringType(), True),
    StructField("ab_variant_id", StringType(), True),
    StructField("ad_action", StringType(), True),
    StructField("ad_placement", StringType(), True),
    StructField("ad_sdk_name", StringType(), True),
    StructField("ad_type", StringType(), True),
    StructField("android_app_build", StringType(), True),
    StructField("android_app_signature", StringType(), True),
    StructField("android_app_version", StringType(), True),
    StructField("android_bundle_id", StringType(), True),
    StructField("android_channel_id", StringType(), True),
    StructField("android_id", StringType(), True),
    StructField("android_mac_md5", StringType(), True),
    StructField("android_mac_sha1", StringType(), True),
    StructField("build", StringType(), True),
    StructField("category", StringType(), True),
    StructField("client_ts", LongType(), True),
    StructField("configuration_keys", ArrayType(StringType()), True),
    StructField("configurations", ArrayType(StringType()), True),
    StructField("connection_type", StringType(), True),
    StructField("device", StringType(), True),
    StructField("engine_version", StringType(), True),
    StructField("google_aid", StringType(), True),
    StructField("google_aid_src", StringType(), True),
    StructField("ios_app_build", StringType(), True),
    StructField("ios_app_version", StringType(), True),
    StructField("ios_att", StringType(), True),
    StructField("ios_bundle_id", StringType(), True),
    StructField("ios_idfa", StringType(), True),
    StructField("ios_idfv", StringType(), True),
    StructField("ios_testflight", BooleanType(), True),
    StructField("jailbroken", BooleanType(), True),
    StructField("limited_ad_tracking", BooleanType(), True),
    StructField("manufacturer", StringType(), True),
    StructField("os_version", StringType(), True),
    StructField("platform", StringType(), True),
    StructField("sdk_version", StringType(), True),
    StructField("session_id", StringType(), True),
    StructField("session_num", LongType(), True),
    StructField("user_id", StringType(), True),
    StructField("user_id_ext", StringType(), True),
    StructField("v", LongType(), True),
    StructField("custom_01", StringType(), True),
    StructField("custom_02", StringType(), True),
    StructField("custom_03", StringType(), True),
])

ads_schema = StructType([
    StructField("ip", StringType(), True, {"comment": "IP address of the player"}),
    StructField("game_id", LongType(), True, {"comment": "Game's unique identifier"}),
    StructField("first_in_batch", BooleanType(), True, {"comment": ""}),
    StructField("country_code", StringType(), True, {"comment": "Country code for the player's country based on events (please note this may change day on day if the player is travelling)"}),
    StructField("arrival_ts", TimestampType(), True, {"comment": "Timestamp for which the event arrived at GA (discrepancy might be for users being offline, for example)"}),
    StructField("install_campaign", StringType(), True, {"comment": ""}),
    StructField("install_publisher", StringType(), True, {"comment": ""}),
    StructField("install_site", StringType(), True, {"comment": ""}),
    StructField("is_paying", BooleanType(), True, {"comment": ""}),
    StructField("origin", StringType(), True, {"comment": ""}),
    StructField("pay_ft", LongType(), True, {"comment": ""}),
    StructField("attribution_partner", StringType(), True, {"comment": ""}),
    StructField("revenue", DoubleType(), True, {"comment": ""}),
    StructField("cohort_month", StringType(), True, {"comment": "First day of the month the player installed the game	"}),
    StructField("is_converting", BooleanType(), True, {"comment": "Flag indicating whether it's the first time the player is making a payment (since we have history of it)	"}),
    StructField("cohort_week", StringType(), True, {"comment": "First day of the week the player installed the game	"}),
    StructField("first_build", StringType(), True, {"comment": ""}),
    StructField("install_ts", TimestampType(), True, {"comment": "Date the player installed the game"}),
    StructField("install_hour", TimestampType(), True, {"comment": ""}),
    StructField("session_id", StringType(), True, {"comment": "Session's unique identifier"}),
    StructField("os_version", StringType(), True, {"comment": "Device's OS version"}),
    StructField("client_ts", TimestampType(), True, {"comment": "Timestamp for which the event occurred"}),
    StructField("session_num", LongType(), True, {"comment": "Session number for that player"}),
    StructField("build", StringType(), True, {"comment": "Game build"}),
    StructField("user_id", StringType(), True, {"comment": "Device identifier of the player (note the same user_id might be linked to multiple game_ids"}),
    StructField("v", LongType(), True, {"comment": "Reflects the version of events coming in to the collectors."}),
    StructField("category", StringType(), True, {"comment": ""}),
    StructField("sdk_version", StringType(), True, {"comment": "SDK version"}),
    StructField("manufacturer", StringType(), True, {"comment": "Device's manufacturer"}),
    StructField("platform", StringType(), True, {"comment": "Platform e.g. ios, android	"}),
    StructField("device", StringType(), True, {"comment": "Device model"}),
    StructField("ab_id", StringType(), True, {"comment": "A/B Testing experiment identifier in case the player is participating in an A/B Test"}),
    StructField("ab_variant_id", StringType(), True, {"comment": "A/B Testing variant identifier in case the player is participating in an A/B Test"}),
    StructField("android_app_build", StringType(), True, {"comment": ""}),
    StructField("android_app_signature", StringType(), True, {"comment": ""}),
    StructField("android_app_version", StringType(), True, {"comment": ""}),
    StructField("android_bundle_id", StringType(), True, {"comment": ""}),
    StructField("android_channel_id", StringType(), True, {"comment": ""}),
    StructField("android_id", StringType(), True, {"comment": "Android id	"}), #find a better defintion
    StructField("android_mac_md5", StringType(), True, {"comment": ""}),
    StructField("android_mac_sha1", StringType(), True, {"comment": ""}),
    StructField("configuration_keys", ArrayType(StringType()), True, {"comment": ""}),
    StructField("configurations", ArrayType(StringType()), True, {"comment": ""}),
    StructField("connection_type", StringType(), True, {"comment": "connection, e.g. lan, wwan, wifi, offline	"}),
    StructField("engine_version", StringType(), True, {"comment": "engine version"}),
    StructField("google_aid", StringType(), True, {"comment": "Android advertising id	"}),
    StructField("google_aid_src", StringType(), True, {"comment": ""}),
    StructField("ios_app_build", StringType(), True, {"comment": ""}),
    StructField("ios_app_version", StringType(), True, {"comment": ""}),
    StructField("ios_att", StringType(), True, {"comment": ""}),
    StructField("ios_bundle_id", StringType(), True, {"comment": ""}),
    StructField("ios_idfa", StringType(), True, {"comment": "IOS identifier for advertisers"}),
    StructField("ios_idfv", StringType(), True, {"comment": "IOS identifier for vendor"}),
    StructField("ios_testflight", BooleanType(), True, {"comment": ""}),
    StructField("jailbroken", BooleanType(), True, {"comment": "whether the player has jailbreaking (process of removing all restrictions imposed on an IOS device) enabled"}),
    StructField("limited_ad_tracking", BooleanType(), True, {"comment": "if True, it means the player does not want to be targeted, preventing attribution of installs to any advertising source"}),
    StructField("user_id_ext", StringType(), True, {"comment": ""}),
    StructField("ad_action", StringType(), True, {"comment": "clicked | show | failed_show | reward_received"}),
    StructField("ad_placement", StringType(), True, {"comment": "end_of_game, after_level,[any string] Max 64 characters"}),
    StructField("ad_sdk_name", StringType(), True, {"comment": "admob, fyber, applovin, ironsource,[any string] Lowercase with no spaces or underscores"}),
    StructField("ad_type", StringType(), True, {"comment": "video | rewarded_video | playable | interstitial | offer_wall | banner"}),
    StructField("custom_01", StringType(), True, {"comment": "Custom field 1"}),
    StructField("custom_02", StringType(), True, {"comment": "Custom field 2"}),
    StructField("custom_03", StringType(), True, {"comment": "Custom field 3"}),
])

ads_comment = """
Ads shown and clicked, fill rate.

Refer to GameAnalytics [documentation](https://docs.gameanalytics.com/event-types/ad-events) for more details.
"""  

@dlt.table(
  name="ads",
  comment=ads_comment,
  schema=ads_schema
  )
@dlt.expect("valid_ad_type", "ad_type IN ('video', 'rewarded_video', 'playable', 'interstitial', 'offer_wall', 'banner')")
@dlt.expect("valid_ad_action", "ad_action IN ('clicked', 'show', 'failed_show', 'reward_received')")
def ads():
  return (
    dlt.read_stream("events")
    .filter(F.col("data").contains('category":"ads"'))
    .withColumn("user_meta", F.from_json("user_meta", ads_user_meta_schema))
    .withColumn("data", F.from_json("data", ads_data_schema))
    .withColumn("arrival_ts", F.to_timestamp(F.from_unixtime(F.col("arrival_ts"))))
    .withColumn("install_campaign", F.col("user_meta.install_campaign"))
    .withColumn("install_site", F.col("user_meta.install_site"))
    .withColumn("is_paying", F.col("user_meta.is_paying"))
    .withColumn("origin", F.col("user_meta.origin"))
    .withColumn("pay_ft", F.col("user_meta.pay_ft"))
    .withColumn("attribution_partner", F.col("user_meta.attribution_partner"))
    .withColumn("revenue", F.col("user_meta.revenue"))
    .withColumn("cohort_month", F.from_unixtime(F.col("user_meta.cohort_month"), "yyyy-MM"))
    .withColumn("is_converting", F.col("user_meta.is_converting"))
    .withColumn("cohort_week", F.from_unixtime(F.col("user_meta.cohort_week")))
    .withColumn("first_build", F.col("user_meta.first_build"))
    .withColumn("install_ts", F.to_timestamp(F.from_unixtime(F.col("user_meta.install_ts"))))
    .withColumn("install_hour", F.to_timestamp(F.from_unixtime(F.col("user_meta.install_hour"))))
    .withColumn("session_id", F.col("data.session_id"))
    .withColumn("os_version", F.col("data.os_version"))
    .withColumn("client_ts", F.to_timestamp(F.from_unixtime(F.col("data.client_ts"))))
    .withColumn("session_num", F.col("data.session_num"))
    .withColumn("build", F.col("data.build"))
    .withColumn("user_id", F.col("data.user_id"))
    .withColumn("v", F.col("data.v"))
    .withColumn("category", F.col("data.category"))
    .withColumn("sdk_version", F.col("data.sdk_version"))
    .withColumn("manufacturer", F.col("data.manufacturer"))
    .withColumn("platform", F.col("data.platform"))
    .withColumn("device", F.col("data.device"))
    .withColumn("ab_id", F.col("data.ab_id"))
    .withColumn("ab_variant_id", F.col("data.ab_variant_id"))
    .withColumn("android_app_build", F.col("data.android_app_build"))
    .withColumn("android_app_signature", F.col("data.android_app_signature"))
    .withColumn("android_app_version", F.col("data.android_app_version"))
    .withColumn("android_bundle_id", F.col("data.android_bundle_id"))
    .withColumn("android_channel_id", F.col("data.android_channel_id"))
    .withColumn("android_id", F.col("data.android_id"))
    .withColumn("android_mac_md5", F.col("data.android_mac_md5"))
    .withColumn("android_mac_sha1", F.col("data.android_mac_sha1"))
    .withColumn("configuration_keys", F.col("data.configuration_keys"))
    .withColumn("configurations", F.col("data.configurations"))
    .withColumn("connection_type", F.col("data.connection_type"))
    .withColumn("engine_version", F.col("data.engine_version"))
    .withColumn("google_aid", F.col("data.google_aid"))
    .withColumn("google_aid_src", F.col("data.google_aid_src"))
    .withColumn("ios_app_build", F.col("data.ios_app_build"))
    .withColumn("ios_app_version", F.col("data.ios_app_version"))
    .withColumn("ios_att", F.col("data.ios_att"))
    .withColumn("ios_bundle_id", F.col("data.ios_bundle_id"))
    .withColumn("ios_idfa", F.col("data.ios_idfa"))
    .withColumn("ios_idfv", F.col("data.ios_idfv"))
    .withColumn("ios_testflight", F.col("data.ios_testflight"))
    .withColumn("jailbroken", F.col("data.jailbroken"))
    .withColumn("limited_ad_tracking", F.col("data.limited_ad_tracking"))
    .withColumn("user_id_ext", F.col("data.user_id_ext"))
    .withColumn("ad_action", F.col("data.ad_action"))
    .withColumn("ad_placement", F.col("data.ad_placement"))
    .withColumn("ad_sdk_name", F.col("data.ad_sdk_name"))
    .withColumn("ad_type", F.col("data.ad_type"))
    .withColumn("custom_01", F.col("data.custom_01"))
    .withColumn("custom_02", F.col("data.custom_02"))
    .withColumn("custom_03", F.col("data.custom_03"))
    .drop("file_path","file_modification_time","user_meta","data")
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Resource

# COMMAND ----------

resource_user_meta_schema = StructType([
    StructField("attribution_partner", StringType(), True),
    StructField("cohort_month", LongType(), True),
    StructField("cohort_week", LongType(), True),
    StructField("first_build", StringType(), True),
    StructField("install_campaign", StringType(), True),
    StructField("install_hour", LongType(), True),
    StructField("install_adgroup", StringType(), True),
    StructField("install_publisher", StringType(), True),
    StructField("install_site", StringType(), True),
    StructField("install_ts", LongType(), True),
    StructField("is_converting", BooleanType(), True),
    StructField("is_paying", BooleanType(), True),
    StructField("origin", StringType(), True),
    StructField("pay_ft", LongType(), True),
    StructField("revenue", DoubleType(), True)
])

resource_data_schema = StructType([
    StructField("amount", LongType(), True),
    StructField("ab_id", StringType(), True),
    StructField("ab_variant_id", StringType(), True),
    StructField("build", StringType(), True),
    StructField("category", StringType(), True),
    StructField("client_ts", LongType(), True),
    StructField("configuration_keys", ArrayType(StringType()), True),
    StructField("element", StringType(), True),
    StructField("configurations", ArrayType(StringType()), True),
    StructField("connection_type", StringType(), True),
    StructField("device", StringType(), True),
    StructField("engine_version", StringType(), True),
    StructField("event_id", StringType(), True),
    StructField("google_aid", StringType(), True),
    StructField("manufacturer", StringType(), True),
    StructField("os_version", StringType(), True),
    StructField("platform", StringType(), True),
    StructField("sdk_version", StringType(), True),
    StructField("session_id", StringType(), True),
    StructField("session_num", LongType(), True),
    StructField("user_id", StringType(), True),
    StructField("user_id_ext", StringType(), True),
    StructField("v", LongType(), True),
    StructField("custom_01", StringType(), True),
    StructField("custom_02", StringType(), True),
    StructField("custom_03", StringType(), True),
])

resource_schema = StructType([
    StructField("ip", StringType(), True, {"comment": "IP address of the player"}),
    StructField("game_id", LongType(), True, {"comment": "Game's unique identifier"}),
    StructField("first_in_batch", BooleanType(), True, {"comment": ""}),
    StructField("country_code", StringType(), True, {"comment": "Country code for the player's country based on events (please note this may change day on day if the player is travelling)"}),
    StructField("arrival_ts", TimestampType(), True, {"comment": "Timestamp for which the event arrived at GA (discrepancy might be for users being offline, for example)"}),
    StructField("install_campaign", StringType(), True, {"comment": ""}),
    StructField("install_adgroup", StringType(), True, {"comment": ""}),
    StructField("install_publisher", StringType(), True, {"comment": ""}),
    StructField("install_site", StringType(), True, {"comment": ""}),
    StructField("is_paying", BooleanType(), True, {"comment": ""}),
    StructField("origin", StringType(), True, {"comment": ""}),
    StructField("pay_ft", LongType(), True, {"comment": ""}),
    StructField("site_id", StringType(), True, {"comment": ""}),
    StructField("attribution_partner", StringType(), True, {"comment": ""}),
    StructField("revenue", DoubleType(), True, {"comment": ""}),
    StructField("cohort_month", StringType(), True, {"comment": "First day of the month the player installed the game	"}),
    StructField("is_converting", BooleanType(), True, {"comment": "Flag indicating whether it's the first time the player is making a payment (since we have history of it)	"}),
    StructField("cohort_week", StringType(), True, {"comment": "First day of the week the player installed the game	"}),
    StructField("first_build", StringType(), True, {"comment": ""}),
    StructField("install_ts", TimestampType(), True, {"comment": "Date the player installed the game"}),
    StructField("install_hour", TimestampType(), True, {"comment": ""}),
    StructField("session_id", StringType(), True, {"comment": "Session's unique identifier"}),
    StructField("os_version", StringType(), True, {"comment": "Device's OS version"}),
    StructField("client_ts", TimestampType(), True, {"comment": "Timestamp for which the event occurred"}),
    StructField("session_num", LongType(), True, {"comment": "Session number for that player"}),
    StructField("build", StringType(), True, {"comment": "Game build"}),
    StructField("user_id", StringType(), True, {"comment": "Device identifier of the player (note the same user_id might be linked to multiple game_ids)	"}),
    StructField("v", LongType(), True, {"comment": "Reflects the version of events coming in to the collectors."}),
    StructField("category", StringType(), True, {"comment": ""}),
    StructField("sdk_version", StringType(), True, {"comment": "SDK version"}),
    StructField("manufacturer", StringType(), True, {"comment": "Device's manufacturer"}),
    StructField("platform", StringType(), True, {"comment": "Platform e.g. ios, android	"}),
    StructField("device", StringType(), True, {"comment": "Device model"}),
    StructField("amount", LongType(), True, {"comment": ""}),
    StructField("ab_id", StringType(), True, {"comment": "A/B Testing experiment identifier in case the player is participating in an A/B Test"}),
    StructField("ab_variant_id", StringType(), True, {"comment": "A/B Testing variant identifier in case the player is participating in an A/B Test"}),
    StructField("configuration_keys", ArrayType(StringType()), True, {"comment": ""}),
    StructField("configurations", ArrayType(StringType()), True, {"comment": ""}),
    StructField("connection_type", StringType(), True, {"comment": "connection, e.g. lan, wwan, wifi, offline	"}),
    StructField("engine_version", StringType(), True, {"comment": "engine version"}),
    StructField("google_aid", StringType(), True, {"comment": "Android advertising id	"}),
    StructField("limited_ad_tracking", BooleanType(), True, {"comment": "if True, it means the player does not want to be targeted, preventing attribution of installs to any advertising source"}),
    StructField("user_id_ext", StringType(), True, {"comment": ""}),
    StructField("custom_01", StringType(), True, {"comment": "Custom field 1"}),
    StructField("custom_02", StringType(), True, {"comment": "Custom field 2"}),
    StructField("custom_03", StringType(), True, {"comment": "Custom field 3"}),
    StructField("event_id", StringType(), True, {"comment": "A 4 part event id string. [flowType]:[virtualCurrency]:[itemType]:[itemId]"}),
    StructField("flow_type", StringType(), True, {"comment": "Flow type is an enum with only 2 possible string values. Sink means spending virtual currency on something. Source means receiving virtual currency from some action."}),
    StructField("virtual_currency", StringType(), True, {"comment": ""}),
    StructField("item_type", StringType(), True, {"comment": ""}),
    StructField("item_id", StringType(), True, {"comment": ""}),
    StructField("element", StringType(), True, {"comment": ""}),
])

resource_comment = """
Managing the flow of virtual currencies - like gems or lives	

Refer to GameAnalytics [documentation](https://docs.gameanalytics.com/event-types/resource-events) for more details.
"""  

@dlt.table(
  name="resource",
  comment=resource_comment,
  schema=resource_schema
  )
@dlt.expect("valid_flow_type", "flow_type IN ('Sink', 'Source')")
def resource():
  return (
    dlt.read_stream("events")
    .filter(F.col("data").contains('"category":"resource"'))
    .withColumn("user_meta", F.from_json("user_meta", resource_user_meta_schema))
    .withColumn("data", F.from_json("data", resource_data_schema))
    .withColumn("arrival_ts", F.to_timestamp(F.from_unixtime(F.col("arrival_ts"))))
    .withColumn("install_campaign", F.col("user_meta.install_campaign"))
    .withColumn("install_site", F.col("user_meta.install_site"))
    .withColumn("is_paying", F.col("user_meta.is_paying"))
    .withColumn("origin", F.col("user_meta.origin"))
    .withColumn("pay_ft", F.col("user_meta.pay_ft"))
    .withColumn("attribution_partner", F.col("user_meta.attribution_partner"))
    .withColumn("revenue", F.col("user_meta.revenue"))
    .withColumn("cohort_month", F.from_unixtime(F.col("user_meta.cohort_month"), "yyyy-MM"))
    .withColumn("is_converting", F.col("user_meta.is_converting"))
    .withColumn("cohort_week", F.from_unixtime(F.col("user_meta.cohort_week")))
    .withColumn("first_build", F.col("user_meta.first_build"))
    .withColumn("install_ts", F.to_timestamp(F.from_unixtime(F.col("user_meta.install_ts"))))
    .withColumn("install_hour", F.to_timestamp(F.from_unixtime(F.col("user_meta.install_hour"))))
    .withColumn("amount", F.col("data.amount"))
    .withColumn("ab_id", F.col("data.ab_id"))
    .withColumn("ab_variant_id", F.col("data.ab_variant_id"))
    .withColumn("build", F.col("data.build"))
    .withColumn("category", F.col("data.category"))
    .withColumn("configuration_keys", F.col("data.configuration_keys"))
    .withColumn("configurations", F.col("data.configurations"))
    .withColumn("connection_type", F.col("data.connection_type"))
    .withColumn("custom_01", F.col("data.custom_01"))
    .withColumn("custom_02", F.col("data.custom_02"))
    .withColumn("custom_03", F.col("data.custom_03"))
    .withColumn("device", F.col("data.device"))
    .withColumn("engine_version", F.col("data.engine_version"))
    .withColumn("event_id", F.col("data.event_id"))
    .withColumn("google_aid", F.col("data.google_aid"))
    .withColumn("manufacturer", F.col("data.manufacturer"))
    .withColumn("os_version", F.col("data.os_version"))
    .withColumn("platform", F.col("data.platform"))
    .withColumn("sdk_version", F.col("data.sdk_version"))
    .withColumn("session_id", F.col("data.session_id"))
    .withColumn("session_num", F.col("data.session_num"))
    .withColumn("user_id", F.col("data.user_id"))
    .withColumn("user_id_ext", F.col("data.user_id_ext"))
    .withColumn("v", F.col("data.v"))
    .withColumn("event_id_split",F.split(F.col("data.event_id"), ":"))
    .withColumn("flow_type", F.col("event_id_split").getItem(0))
    .withColumn("virtual_currency", F.col("event_id_split").getItem(1))
    .withColumn("item_type", F.col("event_id_split").getItem(2))
    .withColumn("item_id", F.col("event_id_split").getItem(3))
    .withColumn("element", F.col("data.element"))
    .drop("file_path","file_modification_time","user_meta","data","event_id_split")
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Impression

# COMMAND ----------

impression_user_meta_schema = StructType([
    StructField("attribution_partner", StringType(), True),
    StructField("cohort_month", LongType(), True),
    StructField("cohort_week", LongType(), True),
    StructField("first_build", StringType(), True),
    StructField("install_campaign", StringType(), True),
    StructField("install_hour", LongType(), True),
    StructField("install_adgroup", StringType(), True),
    StructField("install_publisher", StringType(), True),
    StructField("install_site", StringType(), True),
    StructField("install_ts", LongType(), True),
    StructField("is_converting", BooleanType(), True),
    StructField("is_paying", BooleanType(), True),
    StructField("origin", StringType(), True),
    StructField("pay_ft", LongType(), True),
    StructField("revenue", DoubleType(), True)
])

impression_data_schema = StructType([
    StructField("ab_id", StringType(), True),
    StructField("ab_variant_id", StringType(), True),
    StructField("ad_network_name", StringType(), True),
    StructField("ad_network_version", StringType(), True),
    StructField("android_app_build", StringType(), True),
    StructField("android_app_signature", StringType(), True),
    StructField("android_app_version", StringType(), True),
    StructField("android_bundle_id", StringType(), True),
    StructField("android_channel_id", StringType(), True),
    StructField("android_id", StringType(), True),
    StructField("android_mac_md5", StringType(), True),
    StructField("android_mac_sha1", StringType(), True),
    StructField("build", StringType(), True),
    StructField("category", StringType(), True),
    StructField("client_ts", LongType(), True),
    StructField("configuration_keys", ArrayType(StringType()), True),
    StructField("element", StringType(), True),
    StructField("configurations", ArrayType(StringType()), True),
    StructField("connection_type", StringType(), True),
    StructField("device", StringType(), True),
    StructField("engine_version", StringType(), True),
    StructField("google_aid", StringType(), True),
    StructField("google_aid_src", StringType(), True),
    StructField("impression_data", StructType([
        StructField("adgroup_id", StringType(), True),
        StructField("adgroup_name", StringType(), True),
        StructField("adgroup_priority", LongType(), True),
        StructField("adgroup_type", StringType(), True),
        StructField("adunit_format", StringType(), True),
        StructField("adunit_id", StringType(), True),
        StructField("adunit_name", StringType(), True),
        StructField("app_version", StringType(), True),
        StructField("country", StringType(), True),
        StructField("creative_id", StringType(), True),
        StructField("currency", StringType(), True),
        StructField("demand_partner_data", StructType([
            StructField("encrypted_cpm", StringType(), True)
        ]), True),
        StructField("id", StringType(), True),
        StructField("network_name", StringType(), True),
        StructField("network_placement_id", StringType(), True),
        StructField("placement", StringType(), True),
        StructField("precision", StringType(), True),
        StructField("publisher_revenue", DoubleType(), True),
        StructField("publisher_revenue_usd_cents", DoubleType(), True),
        StructField("revenue", DoubleType(), True)
    ]), True),
    StructField("ios_app_build", StringType(), True),
    StructField("ios_app_version", StringType(), True),
    StructField("ios_att", StringType(), True),
    StructField("ios_bundle_id", StringType(), True),
    StructField("ios_idfa", StringType(), True),
    StructField("ios_idfv", StringType(), True),
    StructField("jailbroken", BooleanType(), True),
    StructField("limited_ad_tracking", BooleanType(), True),
    StructField("manufacturer", StringType(), True),
    StructField("oaid", StringType(), True),
    StructField("os_version", StringType(), True),
    StructField("platform", StringType(), True),
    StructField("sdk_version", StringType(), True),
    StructField("session_id", StringType(), True),
    StructField("session_num", LongType(), True),
    StructField("user_id", StringType(), True),
    StructField("user_id_ext", StringType(), True),
    StructField("v", LongType(), True),
    StructField("custom_01", StringType(), True),
    StructField("custom_02", StringType(), True),
    StructField("custom_03", StringType(), True),
])

impression_schema = StructType([
    StructField("ip", StringType(), True, {"comment": "IP address of the player"}),
    StructField("game_id", LongType(), True, {"comment": "Game's unique identifier"}),
    StructField("first_in_batch", BooleanType(), True, {"comment": ""}),
    StructField("country_code", StringType(), True, {"comment": "Country code for the player's country based on events (please note this may change day on day if the player is travelling)"}),
    StructField("arrival_ts", TimestampType(), True, {"comment": "Timestamp for which the event arrived at GA (discrepancy might be for users being offline, for example)"}),
    StructField("install_campaign", StringType(), True, {"comment": ""}),
    StructField("install_adgroup", StringType(), True, {"comment": ""}),
    StructField("install_publisher", StringType(), True, {"comment": ""}),
    StructField("install_site", StringType(), True, {"comment": ""}),
    StructField("is_paying", BooleanType(), True, {"comment": ""}),
    StructField("origin", StringType(), True, {"comment": ""}),
    StructField("pay_ft", LongType(), True, {"comment": ""}),
    StructField("site_id", StringType(), True, {"comment": ""}),
    StructField("attribution_partner", StringType(), True, {"comment": ""}),
    StructField("revenue", DoubleType(), True, {"comment": ""}),
    StructField("cohort_month", StringType(), True, {"comment": "First day of the month the player installed the game	"}),
    StructField("is_converting", BooleanType(), True, {"comment": "Flag indicating whether it's the first time the player is making a payment (since we have history of it)	"}),
    StructField("cohort_week", StringType(), True, {"comment": "First day of the week the player installed the game	"}),
    StructField("first_build", StringType(), True, {"comment": ""}),
    StructField("install_ts", TimestampType(), True, {"comment": "Date the player installed the game"}),
    StructField("install_hour", TimestampType(), True, {"comment": ""}),
    StructField("session_id", StringType(), True, {"comment": "Session's unique identifier"}),
    StructField("os_version", StringType(), True, {"comment": "Device's OS version"}),
    StructField("client_ts", TimestampType(), True, {"comment": "Timestamp for which the event occurred"}),
    StructField("session_num", LongType(), True, {"comment": "Session number for that player"}),
    StructField("build", StringType(), True, {"comment": "Game build"}),
    StructField("user_id", StringType(), True, {"comment": "Device identifier of the player (note the same user_id might be linked to multiple game_ids)	"}),
    StructField("v", LongType(), True, {"comment": "Reflects the version of events coming in to the collectors."}),
    StructField("category", StringType(), True, {"comment": ""}),
    StructField("sdk_version", StringType(), True, {"comment": "SDK version"}),
    StructField("manufacturer", StringType(), True, {"comment": "Device's manufacturer"}),
    StructField("platform", StringType(), True, {"comment": "Platform e.g. ios, android	"}),
    StructField("device", StringType(), True, {"comment": "Device model"}),
    StructField("ab_id", StringType(), True, {"comment": "A/B Testing experiment identifier in case the player is participating in an A/B Test"}),
    StructField("ab_variant_id", StringType(), True, {"comment": "A/B Testing variant identifier in case the player is participating in an A/B Test"}),
    StructField("ad_network_name", StringType(), True, {"comment": ""}),
    StructField("ad_network_version", StringType(), True, {"comment": ""}),
    StructField("android_app_build", StringType(), True, {"comment": ""}),
    StructField("android_app_signature", StringType(), True, {"comment": ""}),
    StructField("android_app_version", StringType(), True, {"comment": ""}),
    StructField("android_bundle_id", StringType(), True, {"comment": ""}),
    StructField("android_channel_id", StringType(), True, {"comment": ""}),
    StructField("android_id", StringType(), True, {"comment": "Android id	"}), #find a better defintion
    StructField("android_mac_md5", StringType(), True, {"comment": ""}),
    StructField("android_mac_sha1", StringType(), True, {"comment": ""}),
    StructField("configuration_keys", ArrayType(StringType()), True, {"comment": ""}),
    StructField("configurations", ArrayType(StringType()), True, {"comment": ""}),
    StructField("element", StringType(), True, {"comment": ""}),
    StructField("connection_type", StringType(), True, {"comment": "connection, e.g. lan, wwan, wifi, offline	"}),
    StructField("engine_version", StringType(), True, {"comment": "engine version"}),
    StructField("google_aid", StringType(), True, {"comment": "Android advertising id	"}),
    StructField("google_aid_src", StringType(), True, {"comment": ""}),
    StructField("adgroup_id", StringType(), True, {"comment": ""}),
    StructField("adgroup_name", StringType(), True, {"comment": ""}),
    StructField("adgroup_priority", LongType(), True, {"comment": ""}),
    StructField("adgroup_type", StringType(), True, {"comment": ""}),
    StructField("adunit_format", StringType(), True, {"comment": ""}),
    StructField("adunit_id", StringType(), True, {"comment": ""}),
    StructField("adunit_name", StringType(), True, {"comment": ""}),
    StructField("app_version", StringType(), True, {"comment": ""}),
    StructField("country", StringType(), True, {"comment": ""}),
    StructField("creative_id", StringType(), True, {"comment": ""}),
    StructField("currency", StringType(), True, {"comment": ""}),
    StructField("encrypted_cpm", StringType(), True, {"comment": ""}),
    StructField("id", StringType(), True, {"comment": ""}),
    StructField("network_name", StringType(), True, {"comment": ""}),
    StructField("network_placement_id", StringType(), True, {"comment": ""}),
    StructField("placement", StringType(), True, {"comment": ""}),
    StructField("precision", StringType(), True, {"comment": ""}),
    StructField("publisher_revenue", DoubleType(), True, {"comment": ""}),
    StructField("publisher_revenue_usd_cents", DoubleType(), True, {"comment": ""}),
    StructField("impression_revenue", DoubleType(), True, {"comment": ""}),
    StructField("ios_app_build", StringType(), True, {"comment": ""}),
    StructField("ios_app_version", StringType(), True, {"comment": ""}),
    StructField("ios_att", StringType(), True, {"comment": ""}),
    StructField("ios_bundle_id", StringType(), True, {"comment": ""}),
    StructField("ios_idfa", StringType(), True, {"comment": "IOS identifier for advertisers"}),
    StructField("ios_idfv", StringType(), True, {"comment": "IOS identifier for vendor"}),
    StructField("jailbroken", BooleanType(), True, {"comment": "whether the player has jailbreaking (process of removing all restrictions imposed on an IOS device) enabled"}),
    StructField("limited_ad_tracking", BooleanType(), True, {"comment": "if True, it means the player does not want to be targeted, preventing attribution of installs to any advertising source"}),
    StructField("oaid", StringType(), True, {"comment": ""}),
    StructField("user_id_ext", StringType(), True, {"comment": ""}),
    StructField("custom_01", StringType(), True, {"comment": "Custom field 1"}),
    StructField("custom_02", StringType(), True, {"comment": "Custom field 2"}),
    StructField("custom_03", StringType(), True, {"comment": "Custom field 3"}),

])

impression_comment = """
Impression data from different ad networks		

Refer to GameAnalytics [documentation](https://docs.gameanalytics.com/event-types/impression-events) for more details.
"""  

@dlt.table(
  name="impression",
  comment=impression_comment,
  schema=impression_schema
  )
def impression():
  return (
    dlt.read_stream("events")
    .filter(F.col("data").contains('"category":"impression"'))
    .withColumn("user_meta", F.from_json("user_meta", impression_user_meta_schema))
    .withColumn("data", F.from_json("data", impression_data_schema))
    .withColumn("arrival_ts", F.to_timestamp(F.from_unixtime(F.col("arrival_ts"))))
    .withColumn("install_campaign", F.col("user_meta.install_campaign"))
    .withColumn("install_site", F.col("user_meta.install_site"))
    .withColumn("is_paying", F.col("user_meta.is_paying"))
    .withColumn("origin", F.col("user_meta.origin"))
    .withColumn("pay_ft", F.col("user_meta.pay_ft"))
    .withColumn("attribution_partner", F.col("user_meta.attribution_partner"))
    .withColumn("revenue", F.col("user_meta.revenue"))
    .withColumn("cohort_month", F.from_unixtime(F.col("user_meta.cohort_month"), "yyyy-MM"))
    .withColumn("is_converting", F.col("user_meta.is_converting"))
    .withColumn("cohort_week", F.from_unixtime(F.col("user_meta.cohort_week")))
    .withColumn("first_build", F.col("user_meta.first_build"))
    .withColumn("install_ts", F.to_timestamp(F.from_unixtime(F.col("user_meta.install_ts"))))
    .withColumn("install_hour", F.to_timestamp(F.from_unixtime(F.col("user_meta.install_hour"))))
    .withColumn("session_id", F.col("data.session_id"))
    .withColumn("os_version", F.col("data.os_version"))
    .withColumn("client_ts", F.to_timestamp(F.from_unixtime(F.col("data.client_ts"))))
    .withColumn("session_num", F.col("data.session_num"))
    .withColumn("build", F.col("data.build"))
    .withColumn("user_id", F.col("data.user_id"))
    .withColumn("v", F.col("data.v"))
    .withColumn("category", F.col("data.category"))
    .withColumn("sdk_version", F.col("data.sdk_version"))
    .withColumn("manufacturer", F.col("data.manufacturer"))
    .withColumn("platform", F.col("data.platform"))
    .withColumn("device", F.col("data.device"))
    .withColumn("ab_id", F.col("data.ab_id"))
    .withColumn("ab_variant_id", F.col("data.ab_variant_id"))
    .withColumn("android_app_build", F.col("data.android_app_build"))
    .withColumn("android_app_signature", F.col("data.android_app_signature"))
    .withColumn("android_app_version", F.col("data.android_app_version"))
    .withColumn("android_bundle_id", F.col("data.android_bundle_id"))
    .withColumn("android_channel_id", F.col("data.android_channel_id"))
    .withColumn("android_id", F.col("data.android_id"))
    .withColumn("android_mac_md5", F.col("data.android_mac_md5"))
    .withColumn("android_mac_sha1", F.col("data.android_mac_sha1"))
    .withColumn("configuration_keys", F.col("data.configuration_keys"))
    .withColumn("configurations", F.col("data.configurations"))
    .withColumn("connection_type", F.col("data.connection_type"))
    .withColumn("engine_version", F.col("data.engine_version"))
    .withColumn("google_aid", F.col("data.google_aid"))
    .withColumn("google_aid_src", F.col("data.google_aid_src"))
    .withColumn("ios_app_build", F.col("data.ios_app_build"))
    .withColumn("ios_app_version", F.col("data.ios_app_version"))
    .withColumn("ios_att", F.col("data.ios_att"))
    .withColumn("ios_bundle_id", F.col("data.ios_bundle_id"))
    .withColumn("ios_idfa", F.col("data.ios_idfa"))
    .withColumn("ios_idfv", F.col("data.ios_idfv"))
    .withColumn("jailbroken", F.col("data.jailbroken"))
    .withColumn("limited_ad_tracking", F.col("data.limited_ad_tracking"))
    .withColumn("oaid", F.col("data.oaid"))
    .withColumn("user_id_ext", F.col("data.user_id_ext"))
    .withColumn("ad_network_name", F.col("data.ad_network_name"))
    .withColumn("ad_network_version", F.col("data.ad_network_version"))
    .withColumn("element", F.col("data.element"))
    .withColumn("adgroup_id", F.col("data.impression_data.adgroup_id"))
    .withColumn("adgroup_name", F.col("data.impression_data.adgroup_name"))
    .withColumn("adgroup_priority", F.col("data.impression_data.adgroup_priority"))
    .withColumn("adgroup_type", F.col("data.impression_data.adgroup_type"))
    .withColumn("adunit_format", F.col("data.impression_data.adunit_format"))
    .withColumn("adunit_id", F.col("data.impression_data.adunit_id"))
    .withColumn("adunit_name", F.col("data.impression_data.adunit_name"))
    .withColumn("app_version", F.col("data.impression_data.app_version"))
    .withColumn("country", F.col("data.impression_data.country"))
    .withColumn("creative_id", F.col("data.impression_data.creative_id"))
    .withColumn("currency", F.col("data.impression_data.currency"))
    .withColumn("encrypted_cpm", F.col("data.impression_data.demand_partner_data.encrypted_cpm"))
    .withColumn("id", F.col("data.impression_data.id"))
    .withColumn("network_name", F.col("data.impression_data.network_name"))
    .withColumn("network_placement_id", F.col("data.impression_data.network_placement_id"))
    .withColumn("placement", F.col("data.impression_data.placement"))
    .withColumn("precision", F.col("data.impression_data.precision"))
    .withColumn("publisher_revenue", F.col("data.impression_data.publisher_revenue"))
    .withColumn("publisher_revenue_usd_cents", F.col("data.impression_data.publisher_revenue_usd_cents"))
    .withColumn("impression_revenue", F.col("data.impression_data.revenue"))
    .withColumn("custom_01", F.col("data.custom_01"))
    .withColumn("custom_02", F.col("data.custom_02"))
    .withColumn("custom_03", F.col("data.custom_03"))
    .drop("file_path","file_modification_time","user_meta","data")
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Error

# COMMAND ----------

error_user_meta_schema = StructType([
    StructField("attribution_partner", StringType(), True),
    StructField("cohort_month", LongType(), True),
    StructField("cohort_week", LongType(), True),
    StructField("first_build", StringType(), True),
    StructField("install_campaign", StringType(), True),
    StructField("install_hour", LongType(), True),
    StructField("install_adgroup", StringType(), True),
    StructField("install_publisher", StringType(), True),
    StructField("install_site", StringType(), True),
    StructField("install_ts", LongType(), True),
    StructField("is_converting", BooleanType(), True),
    StructField("is_paying", BooleanType(), True),
    StructField("origin", StringType(), True),
    StructField("pay_ft", LongType(), True),
    StructField("revenue", DoubleType(), True)
])

error_data_schema = StructType([
    StructField("ab_id", StringType(), True),
    StructField("ab_variant_id", StringType(), True),
    StructField("android_app_build", StringType(), True),
    StructField("android_app_signature", StringType(), True),
    StructField("android_app_version", StringType(), True),
    StructField("android_bundle_id", StringType(), True),
    StructField("android_channel_id", StringType(), True),
    StructField("android_id", StringType(), True),
    StructField("android_mac_md5", StringType(), True),
    StructField("android_mac_sha1", StringType(), True),
    StructField("build", StringType(), True),
    StructField("category", StringType(), True),
    StructField("client_ts", LongType(), True),
    StructField("configuration_keys", ArrayType(StringType()), True),
    StructField("element", StringType(), True),
    StructField("configurations", ArrayType(StringType()), True),
    StructField("connection_type", StringType(), True),
    StructField("country_code", StringType(), True),
    StructField("device", StringType(), True),
    StructField("engine_version", StringType(), True),
    StructField("google_aid", StringType(), True),
    StructField("google_aid_src", StringType(), True),
    StructField("ios_app_build", StringType(), True),
    StructField("ios_app_version", StringType(), True),
    StructField("ios_att", StringType(), True),
    StructField("ios_bundle_id", StringType(), True),
    StructField("ios_idfa", StringType(), True),
    StructField("ios_idfv", StringType(), True),
    StructField("ios_testflight", BooleanType(), True),
    StructField("jailbroken", BooleanType(), True),
    StructField("limited_ad_tracking", BooleanType(), True),
    StructField("manufacturer", StringType(), True),
    StructField("message", StringType(), True),
    StructField("oaid", StringType(), True),
    StructField("os_version", StringType(), True),
    StructField("platform", StringType(), True),
    StructField("sdk_version", StringType(), True),
    StructField("session_id", StringType(), True),
    StructField("session_num", LongType(), True),
    StructField("severity", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("user_id_ext", StringType(), True),
    StructField("v", LongType(), True),
    StructField("custom_01", StringType(), True),
    StructField("custom_02", StringType(), True),
    StructField("custom_03", StringType(), True),
])

error_schema = StructType([
    StructField("ip", StringType(), True, {"comment": "IP address of the player"}),
    StructField("game_id", LongType(), True, {"comment": "Game's unique identifier"}),
    StructField("first_in_batch", BooleanType(), True, {"comment": ""}),
    StructField("country_code", StringType(), True, {"comment": "Country code for the player's country based on events (please note this may change day on day if the player is travelling)"}),
    StructField("arrival_ts", TimestampType(), True, {"comment": "Timestamp for which the event arrived at GA (discrepancy might be for users being offline, for example)"}),
    StructField("install_campaign", StringType(), True, {"comment": ""}),
    StructField("install_adgroup", StringType(), True, {"comment": ""}),
    StructField("install_publisher", StringType(), True, {"comment": ""}),
    StructField("install_site", StringType(), True, {"comment": ""}),
    StructField("is_paying", BooleanType(), True, {"comment": ""}),
    StructField("origin", StringType(), True, {"comment": ""}),
    StructField("pay_ft", LongType(), True, {"comment": ""}),
    StructField("site_id", StringType(), True, {"comment": ""}),
    StructField("attribution_partner", StringType(), True, {"comment": ""}),
    StructField("revenue", DoubleType(), True, {"comment": ""}),
    StructField("cohort_month", StringType(), True, {"comment": "First day of the month the player installed the game	"}),
    StructField("is_converting", BooleanType(), True, {"comment": "Flag indicating whether it's the first time the player is making a payment (since we have history of it)	"}),
    StructField("cohort_week", StringType(), True, {"comment": "First day of the week the player installed the game	"}),
    StructField("first_build", StringType(), True, {"comment": ""}),
    StructField("install_ts", TimestampType(), True, {"comment": "Date the player installed the game"}),
    StructField("install_hour", TimestampType(), True, {"comment": ""}),
    StructField("session_id", StringType(), True, {"comment": "Session's unique identifier"}),
    StructField("os_version", StringType(), True, {"comment": "Device's OS version"}),
    StructField("client_ts", TimestampType(), True, {"comment": "Timestamp for which the event occurred"}),
    StructField("session_num", LongType(), True, {"comment": "Session number for that player"}),
    StructField("build", StringType(), True, {"comment": "Game build"}),
    StructField("user_id", StringType(), True, {"comment": "Device identifier of the player (note the same user_id might be linked to multiple game_ids)	"}),
    StructField("v", LongType(), True, {"comment": "Reflects the version of events coming in to the collectors."}),
    StructField("category", StringType(), True, {"comment": ""}),
    StructField("sdk_version", StringType(), True, {"comment": "SDK version"}),
    StructField("manufacturer", StringType(), True, {"comment": "Device's manufacturer"}),
    StructField("platform", StringType(), True, {"comment": "Platform e.g. ios, android	"}),
    StructField("device", StringType(), True, {"comment": "Device model"}),
    StructField("ab_id", StringType(), True, {"comment": "A/B Testing experiment identifier in case the player is participating in an A/B Test"}),
    StructField("ab_variant_id", StringType(), True, {"comment": "A/B Testing variant identifier in case the player is participating in an A/B Test"}),
    StructField("android_app_build", StringType(), True, {"comment": ""}),
    StructField("android_app_signature", StringType(), True, {"comment": ""}),
    StructField("android_app_version", StringType(), True, {"comment": ""}),
    StructField("android_bundle_id", StringType(), True, {"comment": ""}),
    StructField("android_channel_id", StringType(), True, {"comment": ""}),
    StructField("android_id", StringType(), True, {"comment": "Android id	"}), #find a better defintion
    StructField("android_mac_md5", StringType(), True, {"comment": ""}),
    StructField("android_mac_sha1", StringType(), True, {"comment": ""}),
    StructField("configuration_keys", ArrayType(StringType()), True, {"comment": ""}),
    StructField("configurations", ArrayType(StringType()), True, {"comment": ""}),
    StructField("element", StringType(), True, {"comment": ""}),
    StructField("connection_type", StringType(), True, {"comment": "connection, e.g. lan, wwan, wifi, offline	"}),
    StructField("engine_version", StringType(), True, {"comment": "engine version"}),
    StructField("google_aid", StringType(), True, {"comment": "Android advertising id	"}),
    StructField("google_aid_src", StringType(), True, {"comment": ""}),
    StructField("ios_app_build", StringType(), True, {"comment": ""}),
    StructField("ios_app_version", StringType(), True, {"comment": ""}),
    StructField("ios_att", StringType(), True, {"comment": ""}),
    StructField("ios_bundle_id", StringType(), True, {"comment": ""}),
    StructField("ios_idfa", StringType(), True, {"comment": "IOS identifier for advertisers"}),
    StructField("ios_idfv", StringType(), True, {"comment": "IOS identifier for vendor"}),
    StructField("jailbroken", BooleanType(), True, {"comment": "whether the player has jailbreaking (process of removing all restrictions imposed on an IOS device) enabled"}),
    StructField("limited_ad_tracking", BooleanType(), True, {"comment": "if True, it means the player does not want to be targeted, preventing attribution of installs to any advertising source"}),
    StructField("oaid", StringType(), True, {"comment": ""}),
    StructField("user_id_ext", StringType(), True, {"comment": ""}),
    StructField("message", StringType(), True, {"comment": ""}),
    StructField("severity", StringType(), True, {"comment": ""}),
    StructField("custom_01", StringType(), True, {"comment": "Custom field 1"}),
    StructField("custom_02", StringType(), True, {"comment": "Custom field 2"}),
    StructField("custom_03", StringType(), True, {"comment": "Custom field 3"}),

])

error_comment = """
Submit exception stack traces or custom error messages.

Refer to GameAnalytics [documentation](https://docs.gameanalytics.com/event-types/error-events) for more details.
"""  

@dlt.table(
  name="error",
  comment=error_comment,
  schema=error_schema
  )
def error():
  return (
    dlt.read_stream("events")
    .filter(F.col("data").contains('"category":"error"'))
    .withColumn("user_meta", F.from_json("user_meta", error_user_meta_schema))
    .withColumn("data", F.from_json("data", error_data_schema))
    .withColumn("arrival_ts", F.to_timestamp(F.from_unixtime(F.col("arrival_ts"))))
    .withColumn("install_campaign", F.col("user_meta.install_campaign"))
    .withColumn("install_site", F.col("user_meta.install_site"))
    .withColumn("is_paying", F.col("user_meta.is_paying"))
    .withColumn("origin", F.col("user_meta.origin"))
    .withColumn("pay_ft", F.col("user_meta.pay_ft"))
    .withColumn("attribution_partner", F.col("user_meta.attribution_partner"))
    .withColumn("revenue", F.col("user_meta.revenue"))
    .withColumn("cohort_month", F.from_unixtime(F.col("user_meta.cohort_month"), "yyyy-MM"))
    .withColumn("is_converting", F.col("user_meta.is_converting"))
    .withColumn("cohort_week", F.from_unixtime(F.col("user_meta.cohort_week")))
    .withColumn("first_build", F.col("user_meta.first_build"))
    .withColumn("install_ts", F.to_timestamp(F.from_unixtime(F.col("user_meta.install_ts"))))
    .withColumn("install_hour", F.to_timestamp(F.from_unixtime(F.col("user_meta.install_hour"))))
    .withColumn("session_id", F.col("data.session_id"))
    .withColumn("os_version", F.col("data.os_version"))
    .withColumn("client_ts", F.to_timestamp(F.from_unixtime(F.col("data.client_ts"))))
    .withColumn("session_num", F.col("data.session_num"))
    .withColumn("build", F.col("data.build"))
    .withColumn("user_id", F.col("data.user_id"))
    .withColumn("v", F.col("data.v"))
    .withColumn("category", F.col("data.category"))
    .withColumn("sdk_version", F.col("data.sdk_version"))
    .withColumn("manufacturer", F.col("data.manufacturer"))
    .withColumn("platform", F.col("data.platform"))
    .withColumn("device", F.col("data.device"))
    .withColumn("ab_id", F.col("data.ab_id"))
    .withColumn("ab_variant_id", F.col("data.ab_variant_id"))
    .withColumn("android_app_build", F.col("data.android_app_build"))
    .withColumn("android_app_signature", F.col("data.android_app_signature"))
    .withColumn("android_app_version", F.col("data.android_app_version"))
    .withColumn("android_bundle_id", F.col("data.android_bundle_id"))
    .withColumn("android_channel_id", F.col("data.android_channel_id"))
    .withColumn("android_id", F.col("data.android_id"))
    .withColumn("android_mac_md5", F.col("data.android_mac_md5"))
    .withColumn("android_mac_sha1", F.col("data.android_mac_sha1"))
    .withColumn("configuration_keys", F.col("data.configuration_keys"))
    .withColumn("configurations", F.col("data.configurations"))
    .withColumn("connection_type", F.col("data.connection_type"))
    .withColumn("engine_version", F.col("data.engine_version"))
    .withColumn("google_aid", F.col("data.google_aid"))
    .withColumn("google_aid_src", F.col("data.google_aid_src"))
    .withColumn("ios_app_build", F.col("data.ios_app_build"))
    .withColumn("ios_app_version", F.col("data.ios_app_version"))
    .withColumn("ios_att", F.col("data.ios_att"))
    .withColumn("ios_bundle_id", F.col("data.ios_bundle_id"))
    .withColumn("ios_idfa", F.col("data.ios_idfa"))
    .withColumn("ios_idfv", F.col("data.ios_idfv"))
    .withColumn("jailbroken", F.col("data.jailbroken"))
    .withColumn("limited_ad_tracking", F.col("data.limited_ad_tracking"))
    .withColumn("user_id_ext", F.col("data.user_id_ext"))
    .withColumn("message", F.col("data.message"))
    .withColumn("severity", F.col("data.severity"))
    .withColumn("custom_01", F.col("data.custom_01"))
    .withColumn("custom_02", F.col("data.custom_02"))
    .withColumn("custom_03", F.col("data.custom_03"))
    .drop("file_path","file_modification_time","user_meta","data")
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Business

# COMMAND ----------

business_user_meta_schema = StructType([
  StructField("attribution_partner", StringType(), True),
  StructField("cohort_month", LongType(), True),
  StructField("cohort_week", LongType(), True),
  StructField("first_build", StringType(), True),
  StructField("install_campaign", StringType(), True),
  StructField("install_hour", LongType(), True),
  StructField("install_publisher", StringType(), True),
  StructField("install_site", StringType(), True),
  StructField("install_ts", LongType(), True),
  StructField("is_converting", BooleanType(), True),
  StructField("is_paying", BooleanType(), True),
  StructField("origin", StringType(), True),
  StructField("pay_ft", LongType(), True),
  StructField("receipt_status", StringType(), True),
  StructField("revenue", DoubleType(), True)
])

business_data_schema = StructType([
    StructField("ab_id", StringType(), True),
    StructField("ab_variant_id", StringType(), True),
    StructField("amount", LongType(), True),
    StructField("amount_usd", DoubleType(), True),
    StructField("android_app_build", StringType(), True),
    StructField("android_app_signature", StringType(), True),
    StructField("android_app_version", StringType(), True),
    StructField("android_bundle_id", StringType(), True),
    StructField("android_channel_id", StringType(), True),
    StructField("android_id", StringType(), True),
    StructField("android_mac_md5", StringType(), True),
    StructField("android_mac_sha1", StringType(), True),
    StructField("build", StringType(), True),
    StructField("cart_type", StringType(), True),
    StructField("category", StringType(), True),
    StructField("client_ts", LongType(), True),
    StructField("configuration_keys", ArrayType(StringType()), True),
    StructField("element", StringType(), True),
    StructField("configurations", ArrayType(StringType()), True),
    StructField("connection_type", StringType(), True),
    StructField("country_code", StringType(), True),
    StructField("currency", StringType(), True),
    StructField("device", StringType(), True),
    StructField("engine_version", StringType(), True),
    StructField("event_id", StringType(), True),
    StructField("google_aid", StringType(), True),
    StructField("google_aid_src", StringType(), True),
    StructField("ios_app_build", StringType(), True),
    StructField("ios_app_version", StringType(), True),
    StructField("ios_att", StringType(), True),
    StructField("ios_bundle_id", StringType(), True),
    StructField("ios_idfa", StringType(), True),
    StructField("ios_idfv", StringType(), True),
    StructField("jailbroken", BooleanType(), True),
    StructField("limited_ad_tracking", BooleanType(), True),
    StructField("manufacturer", StringType(), True),
    StructField("oaid", StringType(), True),
    StructField("os_version", StringType(), True),
    StructField("platform", StringType(), True),
    StructField("receipt_info", StructType([
      StructField("receipt", StringType(), True),
      StructField("receipt_id", StringType(), True),
      StructField("signature", StringType(), True),
      StructField("store", StringType(), True)
      ]), True),
    StructField("sdk_version", StringType(), True),
    StructField("session_id", StringType(), True),
    StructField("session_num", LongType(), True),
    StructField("transaction_num", LongType(), True),
    StructField("user_id", StringType(), True),
    StructField("user_id_ext", StringType(), True),
    StructField("v", LongType(), True),
    StructField("custom_01", StringType(), True),
    StructField("custom_02", StringType(), True),
    StructField("custom_03", StringType(), True),
])

business_schema = StructType([
    StructField("ip", StringType(), True, {"comment": "IP address of the player"}),
    StructField("game_id", LongType(), True, {"comment": "Game's unique identifier"}),
    StructField("first_in_batch", BooleanType(), True, {"comment": ""}),
    StructField("country_code", StringType(), True, {"comment": "Country code for the player's country based on events (please note this may change day on day if the player is travelling)"}),
    StructField("arrival_ts", TimestampType(), True, {"comment": "Timestamp for which the event arrived at GA (discrepancy might be for users being offline, for example)"}),
    StructField("install_campaign", StringType(), True, {"comment": ""}),
    StructField("install_adgroup", StringType(), True, {"comment": ""}),
    StructField("install_publisher", StringType(), True, {"comment": ""}),
    StructField("install_site", StringType(), True, {"comment": ""}),
    StructField("is_paying", BooleanType(), True, {"comment": ""}),
    StructField("origin", StringType(), True, {"comment": ""}),
    StructField("pay_ft", LongType(), True, {"comment": ""}),
    StructField("site_id", StringType(), True, {"comment": ""}),
    StructField("attribution_partner", StringType(), True, {"comment": ""}),
    StructField("revenue", DoubleType(), True, {"comment": ""}),
    StructField("cohort_month", StringType(), True, {"comment": "First day of the month the player installed the game	"}),
    StructField("is_converting", BooleanType(), True, {"comment": "Flag indicating whether it's the first time the player is making a payment (since we have history of it)	"}),
    StructField("cohort_week", StringType(), True, {"comment": "First day of the week the player installed the game	"}),
    StructField("first_build", StringType(), True, {"comment": ""}),
    StructField("install_ts", TimestampType(), True, {"comment": "Date the player installed the game"}),
    StructField("install_hour", TimestampType(), True, {"comment": ""}),
    StructField("receipt_status", StringType(), True, {"comment": ""}),
    StructField("session_id", StringType(), True, {"comment": "Session's unique identifier"}),
    StructField("os_version", StringType(), True, {"comment": "Device's OS version"}),
    StructField("client_ts", TimestampType(), True, {"comment": "Timestamp for which the event occurred"}),
    StructField("session_num", LongType(), True, {"comment": "Session number for that player"}),
    StructField("build", StringType(), True, {"comment": "Game build"}),
    StructField("user_id", StringType(), True, {"comment": "Device identifier of the player (note the same user_id might be linked to multiple game_ids)	"}),
    StructField("v", LongType(), True, {"comment": "Reflects the version of events coming in to the collectors."}),
    StructField("category", StringType(), True, {"comment": ""}),
    StructField("sdk_version", StringType(), True, {"comment": "SDK version"}),
    StructField("manufacturer", StringType(), True, {"comment": "Device's manufacturer"}),
    StructField("platform", StringType(), True, {"comment": "Platform e.g. ios, android	"}),
    StructField("device", StringType(), True, {"comment": "Device model"}),
    StructField("ab_id", StringType(), True, {"comment": "A/B Testing experiment identifier in case the player is participating in an A/B Test"}),
    StructField("ab_variant_id", StringType(), True, {"comment": "A/B Testing variant identifier in case the player is participating in an A/B Test"}),
    StructField("android_app_build", StringType(), True, {"comment": ""}),
    StructField("android_app_signature", StringType(), True, {"comment": ""}),
    StructField("android_app_version", StringType(), True, {"comment": ""}),
    StructField("android_bundle_id", StringType(), True, {"comment": ""}),
    StructField("android_channel_id", StringType(), True, {"comment": ""}),
    StructField("android_id", StringType(), True, {"comment": "Android id	"}), #find a better defintion
    StructField("android_mac_md5", StringType(), True, {"comment": ""}),
    StructField("android_mac_sha1", StringType(), True, {"comment": ""}),
    StructField("configuration_keys", ArrayType(StringType()), True, {"comment": ""}),
    StructField("configurations", ArrayType(StringType()), True, {"comment": ""}),
    StructField("element", StringType(), True, {"comment": ""}),
    StructField("connection_type", StringType(), True, {"comment": "connection, e.g. lan, wwan, wifi, offline	"}),
    StructField("engine_version", StringType(), True, {"comment": "engine version"}),
    StructField("google_aid", StringType(), True, {"comment": "Android advertising id	"}),
    StructField("google_aid_src", StringType(), True, {"comment": ""}),
    StructField("ios_app_build", StringType(), True, {"comment": ""}),
    StructField("ios_app_version", StringType(), True, {"comment": ""}),
    StructField("ios_att", StringType(), True, {"comment": ""}),
    StructField("ios_bundle_id", StringType(), True, {"comment": ""}),
    StructField("ios_idfa", StringType(), True, {"comment": "IOS identifier for advertisers"}),
    StructField("ios_idfv", StringType(), True, {"comment": "IOS identifier for vendor"}),
    StructField("jailbroken", BooleanType(), True, {"comment": "whether the player has jailbreaking (process of removing all restrictions imposed on an IOS device) enabled"}),
    StructField("limited_ad_tracking", BooleanType(), True, {"comment": "if True, it means the player does not want to be targeted, preventing attribution of installs to any advertising source"}),
    StructField("oaid", StringType(), True, {"comment": ""}),
    StructField("user_id_ext", StringType(), True, {"comment": ""}),
    StructField("transaction_num", LongType(), True, {"comment": ""}),
    StructField("amount", LongType(), True, {"comment": ""}),
    StructField("amount_usd", DoubleType(), True, {"comment": ""}),
    StructField("currency", StringType(), True, {"comment": ""}),
    StructField("cart_type", StringType(), True, {"comment": ""}),
    StructField("event_id", StringType(), True, {"comment": ""}),
    StructField("receipt", StringType(), True, {"comment": ""}),
    StructField("receipt_id", StringType(), True, {"comment": ""}),
    StructField("signature", StringType(), True, {"comment": ""}),
    StructField("store", StringType(), True, {"comment": ""}),
    StructField("custom_01", StringType(), True, {"comment": "Custom field 1"}),
    StructField("custom_02", StringType(), True, {"comment": "Custom field 2"}),
    StructField("custom_03", StringType(), True, {"comment": "Custom field 3"}),

])

business_comment = """
In-App Purchases supporting receipt validation on GA servers.

Refer to GameAnalytics [documentation](https://docs.gameanalytics.com/event-types/business-events) for more details.
"""  

@dlt.table(
  name="business",
  comment=business_comment,
  schema=business_schema
  )
def business_():
  return (
    dlt.read_stream("events")
    .filter(F.col("data").contains('category":"business"'))
    .withColumn("user_meta", F.from_json("user_meta", business_user_meta_schema))
    .withColumn("data", F.from_json("data", business_data_schema))
    .withColumn("arrival_ts", F.to_timestamp(F.from_unixtime(F.col("arrival_ts"))))
    .withColumn("install_campaign", F.col("user_meta.install_campaign"))
    .withColumn("install_site", F.col("user_meta.install_site"))
    .withColumn("is_paying", F.col("user_meta.is_paying"))
    .withColumn("origin", F.col("user_meta.origin"))
    .withColumn("pay_ft", F.col("user_meta.pay_ft"))
    .withColumn("attribution_partner", F.col("user_meta.attribution_partner"))
    .withColumn("revenue", F.col("user_meta.revenue"))
    .withColumn("cohort_month", F.from_unixtime(F.col("user_meta.cohort_month"), "yyyy-MM"))
    .withColumn("is_converting", F.col("user_meta.is_converting"))
    .withColumn("cohort_week", F.from_unixtime(F.col("user_meta.cohort_week")))
    .withColumn("first_build", F.col("user_meta.first_build"))
    .withColumn("install_ts", F.to_timestamp(F.from_unixtime(F.col("user_meta.install_ts"))))
    .withColumn("install_hour", F.to_timestamp(F.from_unixtime(F.col("user_meta.install_hour"))))
    .withColumn("session_id", F.col("data.session_id"))
    .withColumn("os_version", F.col("data.os_version"))
    .withColumn("client_ts", F.to_timestamp(F.from_unixtime(F.col("data.client_ts"))))
    .withColumn("session_num", F.col("data.session_num"))
    .withColumn("build", F.col("data.build"))
    .withColumn("user_id", F.col("data.user_id"))
    .withColumn("v", F.col("data.v"))
    .withColumn("category", F.col("data.category"))
    .withColumn("sdk_version", F.col("data.sdk_version"))
    .withColumn("manufacturer", F.col("data.manufacturer"))
    .withColumn("platform", F.col("data.platform"))
    .withColumn("device", F.col("data.device"))
    .withColumn("ab_id", F.col("data.ab_id"))
    .withColumn("ab_variant_id", F.col("data.ab_variant_id"))
    .withColumn("android_app_build", F.col("data.android_app_build"))
    .withColumn("android_app_signature", F.col("data.android_app_signature"))
    .withColumn("android_app_version", F.col("data.android_app_version"))
    .withColumn("android_bundle_id", F.col("data.android_bundle_id"))
    .withColumn("android_channel_id", F.col("data.android_channel_id"))
    .withColumn("android_id", F.col("data.android_id"))
    .withColumn("android_mac_md5", F.col("data.android_mac_md5"))
    .withColumn("android_mac_sha1", F.col("data.android_mac_sha1"))
    .withColumn("configuration_keys", F.col("data.configuration_keys"))
    .withColumn("configurations", F.col("data.configurations"))
    .withColumn("connection_type", F.col("data.connection_type"))
    .withColumn("engine_version", F.col("data.engine_version"))
    .withColumn("google_aid", F.col("data.google_aid"))
    .withColumn("google_aid_src", F.col("data.google_aid_src"))
    .withColumn("ios_app_build", F.col("data.ios_app_build"))
    .withColumn("ios_app_version", F.col("data.ios_app_version"))
    .withColumn("ios_att", F.col("data.ios_att"))
    .withColumn("ios_bundle_id", F.col("data.ios_bundle_id"))
    .withColumn("ios_idfa", F.col("data.ios_idfa"))
    .withColumn("ios_idfv", F.col("data.ios_idfv"))
    .withColumn("jailbroken", F.col("data.jailbroken"))
    .withColumn("limited_ad_tracking", F.col("data.limited_ad_tracking"))
    .withColumn("user_id_ext", F.col("data.user_id_ext"))
    .withColumn("receipt_status", F.col("user_meta.receipt_status"))
    .withColumn("transaction_num", F.col("data.transaction_num"))
    .withColumn("amount", F.col("data.amount"))
    .withColumn("amount_usd", F.col("data.amount_usd"))
    .withColumn("currency", F.col("data.currency"))
    .withColumn("cart_type", F.col("data.cart_type"))
    .withColumn("event_id", F.col("data.event_id"))
    .withColumn("receipt", F.col("data.receipt_info.receipt"))
    .withColumn("receipt_id", F.col("data.receipt_info.receipt_id"))
    .withColumn("signature", F.col("data.receipt_info.signature"))
    .withColumn("store", F.col("data.receipt_info.store"))
    .withColumn("custom_01", F.col("data.custom_01"))
    .withColumn("custom_02", F.col("data.custom_02"))
    .withColumn("custom_03", F.col("data.custom_03"))
    .drop("file_path","file_modification_time","user_meta","data")
  )
