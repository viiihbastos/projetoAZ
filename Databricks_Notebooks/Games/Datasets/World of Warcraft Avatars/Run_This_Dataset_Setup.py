# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Simplify ETL with Delta Live Table
# MAGIC
# MAGIC DLT makes Data Engineering accessible for all. Just declare your transformations in SQL or Python, and DLT will handle the Data Engineering complexity for you.
# MAGIC
# MAGIC <img style="float:right" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/dlt-golden-demo-1.png" width="700"/>
# MAGIC
# MAGIC **Accelerate ETL development** <br/>
# MAGIC Enable analysts and data engineers to innovate rapidly with simple pipeline development and maintenance 
# MAGIC
# MAGIC **Remove operational complexity** <br/>
# MAGIC By automating complex administrative tasks and gaining broader visibility into pipeline operations
# MAGIC
# MAGIC **Trust your data** <br/>
# MAGIC With built-in quality controls and quality monitoring to ensure accurate and useful BI, Data Science, and ML 
# MAGIC
# MAGIC **Simplify batch and streaming** <br/>
# MAGIC With self-optimization and auto-scaling data pipelines for batch or streaming processing 
# MAGIC
# MAGIC ## Our Delta Live Table pipeline
# MAGIC
# MAGIC We'll be using as input a raw dataset containing information on our customers Loan and historical transactions. 
# MAGIC
# MAGIC Our goal is to ingest this data in near real time and build table for our Analyst team while ensuring data quality.
# MAGIC
# MAGIC <!-- do not remove -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Ffeatures%2Fdlt%2Fnotebook_dlt_sql&dt=DLT">
# MAGIC <!-- [metadata={"description":"Full DLT demo, going into details. Use loan dataset",
# MAGIC  "authors":["dillon.bostwick@databricks.com"],
# MAGIC  "db_resources":{},
# MAGIC   "search_tags":{"vertical": "retail", "step": "Data Engineering", "components": ["autoloader", "dlt"]}}] -->

# COMMAND ----------

# MAGIC %md-sandbox 
# MAGIC
# MAGIC ## Bronze layer: incrementally ingest data leveraging Databricks Autoloader
# MAGIC
# MAGIC <img style="float: right; padding-left: 10px" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/dlt-golden-demo-2.png" width="500"/>
# MAGIC
# MAGIC Our raw data is being sent to a blob storage. 
# MAGIC
# MAGIC Autoloader simplify this ingestion, including schema inference, schema evolution while being able to scale to millions of incoming files. 
# MAGIC
# MAGIC Autoloader is available in Python using the `cloud_files` format and can be used with a variety of format (json, csv, avro...):
# MAGIC
# MAGIC
# MAGIC #### STREAMING LIVE TABLE 
# MAGIC Defining tables as `STREAMING` will guarantee that you only consume new incoming data. Without `STREAMING`, you will scan and ingest all the data available at once. See the [documentation](https://docs.databricks.com/data-engineering/delta-live-tables/delta-live-tables-incremental-data.html) for more details

# COMMAND ----------

# MAGIC %md
# MAGIC Run this cell only and input your Kaggle username and key

# COMMAND ----------

# MAGIC %run "./setup"

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *

# COMMAND ----------

def create_table(comment, landing_location, table_name):
    df = (
        spark.read.format("csv")
        .option("inferSchema", "true")
        .option("header", "true")
        .load(landing_location)
    )
    df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(table_name)

# COMMAND ----------

create_table("Player game session data with start and end data", f"{tmpdir}/sessions_landing/", "BZ_wow_player_sessions")
create_table("Character level events with timestamp", f"{tmpdir}/level_events_landing/", "BZ_wow_character_level_events")
create_table("Event when characters change zones", f"{tmpdir}/zone_events_landing/", "BZ_wow_character_zone_events")
create_table("Raw player data", f"{tmpdir}/player_data_landing/", "BZ_wow_player_data")
create_table("Character Guild change events", f"{tmpdir}/guild_events_landing/", "BZ_wow_character_guild_events")
create_table("Zone list with custom map coordinates, and zone names", f"{tmpdir}/location_coords_landing/", "BZ_wow_zone_coordinates")
create_table("List of zones", f"{tmpdir}/zones_landing/", "BZ_wow_zones")
create_table("Zone locations & game version", f"{tmpdir}/locations_landing/", "BZ_wow_locations")

# COMMAND ----------

# MAGIC %md-sandbox 
# MAGIC
# MAGIC ## Silver layer: joining tables while ensuring data quality
# MAGIC
# MAGIC <img style="float: right; padding-left: 10px" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/dlt-golden-demo-3.png" width="500"/>
# MAGIC
# MAGIC Once the bronze layer is defined, we'll create the sliver layers by Joining data. Note that bronze tables are referenced using the `LIVE` spacename. 
# MAGIC
# MAGIC To consume only increment from the Bronze layer like `BZ_raw_txs`, we'll be using the `read_stream` function: `dlt.read_stream("BZ_raw_txs")`
# MAGIC
# MAGIC Note that we don't have to worry about compactions, DLT handles that for us.
# MAGIC
# MAGIC #### Expectations
# MAGIC By defining expectations (`@dlt.expect`), you can enforce and track your data quality. See the [documentation](https://docs.databricks.com/data-engineering/delta-live-tables/delta-live-tables-expectations.html) for more details

# COMMAND ----------

df = spark.read.table("games_solutions.world_of_warcraft_avatars.BZ_wow_player_sessions") \
  .withColumn("minutes_since_last_session", (
    F.round((F.unix_timestamp(F.col("start_timestamp")) - F.unix_timestamp(F.lag(F.col("end_timestamp"), 1).over(Window.partitionBy("char").orderBy("start_timestamp"))))/60))) \
  .na.fill(value=0, subset=["minutes_since_last_session"]) \
  .withColumn("session_length", F.when(F.round((F.unix_timestamp(F.col('end_timestamp')) - F.unix_timestamp(F.col('start_timestamp')))/60) == 0, 9).otherwise(F.round((F.unix_timestamp(F.col('end_timestamp')) - F.unix_timestamp(F.col('start_timestamp')))/60))) \
  .withColumn("changed", F.when(F.col("minutes_since_last_session") >= 30000, 1).otherwise(0))

df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable("SLV_wow_subscription_session_changes")
display(df)

# COMMAND ----------

# Read the source data
source_df = spark.read.format("delta").table("BZ_wow_character_level_events")

# Get the latest record for each character based on the sequence column (Timestamp)
latest_char_level_df = (source_df
    .withColumn("rank", F.row_number().over(Window.partitionBy("char").orderBy(F.desc("Timestamp"))))
    .filter(F.col("rank") == 1)  # Keep only the latest records
    .select("char", "level")  # Select only relevant columns
)

# Write the result to the target Delta table
latest_char_level_df.write.format("delta").mode("overwrite").saveAsTable("SLV_wow_latest_char_level")

# COMMAND ----------

# Read the source data
source_df = spark.read.format("delta").table("BZ_wow_character_zone_events")

# Get the latest record for each character based on the sequence column (Timestamp)
latest_char_zone_df = (source_df
    .withColumn("rank", F.row_number().over(Window.partitionBy("char").orderBy(F.desc("Timestamp"))))
    .filter(F.col("rank") == 1)  # Keep only the latest records
    .select("char", "zone")  # Select only relevant columns
)

# Write the result to the target Delta table
latest_char_zone_df.write.format("delta").mode("overwrite").saveAsTable("SLV_wow_latest_char_zone")

# COMMAND ----------

# Define the cumulative window
cumulative_window = Window.partitionBy('char').orderBy('sessionid').rangeBetween(Window.unboundedPreceding, 0)

# Read the source data
session_df = spark.read.format("delta").table("BZ_wow_player_sessions")

# Calculate session length and other aggregations
session_agg_df = (session_df
    .withColumn("session_length", 
                F.when(F.round((F.unix_timestamp(F.col('end_timestamp')) - F.unix_timestamp(F.col('start_timestamp')))/60) == 0, 9)
                .otherwise(F.round((F.unix_timestamp(F.col('end_timestamp')) - F.unix_timestamp(F.col('start_timestamp')))/60))
    )
    .withColumn("session_gap", 
                F.round((F.unix_timestamp(F.col("start_timestamp")) - F.unix_timestamp(F.lag(F.col("end_timestamp"), 1)
                .over(Window.partitionBy("char").orderBy("start_timestamp"))))/60)
    )
    .na.fill(value=0, subset=["session_gap"])
    .withColumn("total_playtime", F.sum(F.col("session_length")).over(cumulative_window))
    .withColumn("avg_session_length", F.avg(F.col("session_length")).over(cumulative_window))
)

# Write the result to the target Delta table
session_agg_df.write.format("delta").mode("overwrite").saveAsTable("SLV_wow_session_agg")

# COMMAND ----------

# Read the source data
zonedf = spark.read.format("delta").table("BZ_wow_character_zone_events")
sessions = spark.read.format("delta").table("BZ_wow_player_sessions")

# Join sessions with zone data and aggregate
zone_change_agg_df = (
    sessions.join(zonedf,
                  (sessions.char == zonedf.char) & 
                  (zonedf.Timestamp.between(sessions.start_timestamp, sessions.end_timestamp)),
                  'left')
    .groupBy(sessions.char, sessions.sessionid)
    .agg(F.count(zonedf.zone).alias('num_of_zone_changes'))
)

# Write the result to the target Delta table
zone_change_agg_df.write.format("delta").mode("overwrite").saveAsTable("SLV_wow_zone_change_agg")

# COMMAND ----------

# Read the source data
sessions = spark.read.format("delta").table("BZ_wow_player_sessions")
leveldf = spark.read.format("delta").table("BZ_wow_character_level_events")

# Join sessions with level data and aggregate
level_change_agg_df = (
    sessions.join(leveldf,
                  (sessions.char == leveldf.char) & 
                  (leveldf.Timestamp.between(sessions.start_timestamp, sessions.end_timestamp)),
                  'left')
    .groupBy(sessions.char, sessions.sessionid)
    .agg(
        (F.count("Level")+1).alias('num_of_level_changes'),
        F.max("Level").alias('end_level'))
    .na.fill(value=1, subset=["end_level"])
)

# Write the result to the target Delta table
level_change_agg_df.write.format("delta").mode("overwrite").saveAsTable("SLV_wow_level_change_agg")

# COMMAND ----------

# MAGIC %md-sandbox 
# MAGIC
# MAGIC ## Gold layer
# MAGIC
# MAGIC <img style="float: right; padding-left: 10px" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/dlt-golden-demo-4.png" width="500"/>
# MAGIC
# MAGIC Our last step is to materialize the Gold Layer.
# MAGIC
# MAGIC Because these tables will be requested at scale using a SQL Endpoint, we'll can add Zorder at the table level to ensure faster queries using `pipelines.autoOptimize.zOrderCols`, and DLT will handle the rest.

# COMMAND ----------

# Load the required tables
latest_char_level_df = spark.read.format("delta").table("SLV_wow_latest_char_level")
player_data_df = spark.read.format("delta").table("BZ_wow_player_data")
latest_char_zone_df = spark.read.format("delta").table("SLV_wow_latest_char_zone")

# Perform the joins
player_status_df = latest_char_level_df.join(player_data_df, ["char"]).join(latest_char_zone_df, ["char"])

# Write the result to the target Delta table
player_status_df.write.format("delta").mode("overwrite").saveAsTable("GLD_wow_player_status")

# Display the result
display(player_status_df)

# COMMAND ----------

# Load the required tables
sessions = spark.read.table("SLV_wow_session_agg")
zone_changes = spark.read.table("SLV_wow_zone_change_agg")
level_changes = spark.read.table("SLV_wow_level_change_agg")

# Perform the joins
session_agg_df = sessions.join(zone_changes, ["char", "sessionid"], "left") \
    .join(level_changes, ["char", "sessionid"], "left") \
    .select(
        sessions.char,
        sessions.sessionid,
        "start_timestamp",
        "end_timestamp",
        "session_length",
        "session_gap",
        "total_playtime",
        "avg_session_length",
        F.coalesce(zone_changes.num_of_zone_changes, F.lit(0)).alias("num_of_zone_changes"),
        F.coalesce(level_changes.num_of_level_changes, F.lit(0)).alias("num_of_level_changes"),
        "end_level"
    )

# Write the result to the target Delta table
session_agg_df.write.format("delta").mode("overwrite").saveAsTable("GLD_wow_player_session_agg")

# Display the result
display(session_agg_df)

# COMMAND ----------

# Load the table
subscription_session_changes_df = spark.read.table("SLV_wow_subscription_session_changes")

# Process the data
player_subscriptions_agg_df = subscription_session_changes_df \
    .withColumn("group_id", F.sum("changed").over(Window.partitionBy("char").orderBy("end_timestamp").rowsBetween(Window.unboundedPreceding, 0))).drop("changed") \
    .groupBy('char', 'group_id').agg(
        F.min('start_timestamp').alias('start_timestamp'),
        F.max('end_timestamp').alias('end_timestamp'),
        F.count('sessionid').alias('num_of_sessions'),
        F.round(F.avg("minutes_since_last_session"), 0).alias("avg_session_gap"),
        F.round(F.avg("session_length"), 0).alias("avg_session_time"),
        F.sum('session_length').alias('total_subscription_length')
    ) \
    .drop('group_id') \
    .withColumn("subscriptionid", F.monotonically_increasing_id())

# Write the result to the target Delta table
player_subscriptions_agg_df.write.format("delta").mode("overwrite").saveAsTable("GLD_wow_player_subscriptions_agg")

# Display the result
display(player_subscriptions_agg_df)
