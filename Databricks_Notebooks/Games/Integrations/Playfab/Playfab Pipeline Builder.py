# Databricks notebook source
dbutils.widgets.text("EventTable", "TablePathHere")

# COMMAND ----------

import json
from pyspark.sql.functions import first, max, col
from pyspark.sql.types import StringType

def generate_columns(json_data, parent_key=None):
    columns = []

    for key, value in json_data.items():
        if parent_key:
            column_key = f"{parent_key}.{key}"
            column_alias = column_key.replace('.', '_')
        else:
            column_key = key
            column_alias = key

        if isinstance(value, dict):
            columns.extend(generate_columns(value, column_key))
        else:
            columns.append(f'get_json_object(EventData, "$.{column_key}") AS {column_alias}')

    return columns

def generate_delta_live_table_code(eventtype, json_str):
    # Parse the JSON string into a dictionary
    json_data = json.loads(json_str)

    # Generate the SELECT statement to extract JSON data into columns
    select_columns = generate_columns(json_data)
    select_columns_str = ", ".join(select_columns)

    # Create the Delta Live Table notebook code
    delta_live_table_code = f'''
    CREATE OR REFRESH STREAMING LIVE TABLE bz_{eventtype} AS
    SELECT {select_columns_str}
    FROM STREAM(live.playfab_bronze)
    WHERE FullName_Name = '{eventtype}'
    '''

    return delta_live_table_code
  
generate_delta_udf = udf(generate_delta_live_table_code, StringType())
  

eventsDF = spark.read.table(dbutils.widgets.get("EventTable"))

# Assuming you have a DataFrame named 'df' with the given schema
latest_rows_agg = (eventsDF.groupBy("FullName_Name")
                   .agg(first("EventId", True).alias("First_EventId"),
                        first("Timestamp", True).alias("Latest_Timestamp")))

latest_rows = (eventsDF.alias("original")
                 .join(latest_rows_agg.alias("latest"), 
                       (col("original.FullName_Name") == col("latest.FullName_Name")) & 
                       (col("original.EventId") == col("latest.First_EventId")), "inner")
                 .select("original.*"))


eventsDF = latest_rows.withColumn("DeltaLiveTableCode", generate_delta_udf(eventsDF["FullName_Name"], eventsDF["EventData"]))


# Display the result
display(eventsDF)
