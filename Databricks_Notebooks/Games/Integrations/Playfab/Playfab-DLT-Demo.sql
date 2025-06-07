-- Databricks notebook source
-- MAGIC %md-sandbox
-- MAGIC ### 1/ Ingesting data with Autoloader
-- MAGIC
-- MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/cdc_dlt/cdc_dlt_pipeline_1.png" width="700" style="float: right" />
-- MAGIC
-- MAGIC Our first step is to ingest the data from the cloud storage. Again, this could be from any other source like (message queue etc).
-- MAGIC
-- MAGIC This can be challenging for multiple reason. We have to:
-- MAGIC
-- MAGIC - operate at scale, potentially ingesting millions of small files
-- MAGIC - infer schema and json type
-- MAGIC - handle bad record with incorrect json schema
-- MAGIC - take care of schema evolution (ex: new column in the customer table)
-- MAGIC
-- MAGIC Databricks Autoloader solves all these challenges out of the box.

-- COMMAND ----------

CREATE OR REFRESH STREAMING LIVE TABLE raw_events
--TBLPROPERTIES(pipelines.reset.allowed = false)
AS SELECT *, current_timestamp() as ingest_timestamp
  FROM cloud_files(
    "wasbs://sink@duncanplayfabdata.blob.core.windows.net/data",
    "parquet",
    map(
    "schema", "SchemaVersion string,
    EventId string,
    FullName_Name string,
    FullName_Namespace string,
    Timestamp timestamp,
    Entity_Id string,
    Entity_Type string,
    EntityLineage_namespace string,
    EntityLineage_master_player_account string,
    EntityLineage_title string,
    EntityLineage_title_player_account string,
    EventData string",
    "recursiveFileLookup", "true"
    )
  )

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Streaming Updates in the Lakehouse
-- MAGIC
-- MAGIC Delta Lake allows users to easily combine streaming and batch workloads in a unified multi-hop pipeline. Each stage represents a state of our data valuable to driving core use cases within the business. Because all data and metadata lives in object storage in the cloud, multiple users and applications can access data in near-real time, allowing analysts to access the freshest data as it's being processed.
-- MAGIC
-- MAGIC ![](https://files.training.databricks.com/images/sslh/multi-hop-simple.png)
-- MAGIC
-- MAGIC - **Bronze** tables contain raw data ingested from various sources (JSON files, RDBMS data,  IoT data, etc.).
-- MAGIC
-- MAGIC - **Silver** tables will provide a more refined view of our data. We can join fields from various bronze tables to enrich streaming records, or update account statuses based on recent activity.
-- MAGIC
-- MAGIC - **Gold** tables provide business level aggregates often used for reporting and dashboarding. This would include aggregations such as daily active website users, weekly sales per store, or gross revenue per quarter by department. 
-- MAGIC
-- MAGIC The end outputs are actionable insights, dashboards and reports of business metrics.
-- MAGIC
-- MAGIC By considering our business logic at all steps of the ETL pipeline, we can ensure that storage and compute costs are optimized by reducing unnecessary duplication of data and limiting ad hoc querying against full historic data.
-- MAGIC
-- MAGIC Each stage can be configured as a batch or streaming job, and ACID transactions ensure that we succeed or fail completely.

-- COMMAND ----------

CREATE
OR REFRESH STREAMING LIVE TABLE bz_PlayerSpawn AS
SELECT
  get_json_object(EventData, "$.EventName") AS EventName,
  get_json_object(EventData, "$.EventNamespace") AS EventNamespace,
  get_json_object(EventData, "$.Source") AS Source,
  get_json_object(EventData, "$.EntityType") AS EntityType,
  get_json_object(EventData, "$.TitleId") AS TitleId,
  get_json_object(EventData, "$.EntityId") AS EntityId,
  get_json_object(EventData, "$.EventId") AS EventId,
  get_json_object(EventData, "$.SourceType") AS SourceType,
  get_json_object(EventData, "$.Timestamp") AS Timestamp,
  get_json_object(EventData, "$.MatchId") AS MatchId,
  get_json_object(EventData, "$.TeamId") AS TeamId
FROM
  STREAM(live.raw_events)
WHERE
  FullName_Name = 'PlayerSpawn'

-- COMMAND ----------

CREATE
OR REFRESH STREAMING LIVE TABLE bz_ItemSpawned AS
SELECT
  get_json_object(EventData, "$.EventName") AS EventName,
  get_json_object(EventData, "$.EventNamespace") AS EventNamespace,
  get_json_object(EventData, "$.EntityType") AS EntityType,
  get_json_object(EventData, "$.Source") AS Source,
  get_json_object(EventData, "$.EventId") AS EventId,
  get_json_object(EventData, "$.EntityId") AS EntityId,
  get_json_object(EventData, "$.SourceType") AS SourceType,
  get_json_object(EventData, "$.Timestamp") AS Timestamp,
  get_json_object(EventData, "$.PlayFabEnvironment.Vertical") AS PlayFabEnvironment_Vertical,
  get_json_object(EventData, "$.PlayFabEnvironment.Cloud") AS PlayFabEnvironment_Cloud,
  get_json_object(EventData, "$.PlayFabEnvironment.Application") AS PlayFabEnvironment_Application,
  get_json_object(EventData, "$.PlayFabEnvironment.Commit") AS PlayFabEnvironment_Commit,
  get_json_object(EventData, "$.MatchId") AS MatchId,
  get_json_object(EventData, "$.ItemName") AS ItemName
FROM
  STREAM(live.raw_events)
WHERE
  FullName_Name = 'ItemSpawned'

-- COMMAND ----------

CREATE
OR REFRESH STREAMING LIVE TABLE bz_ItemPickup AS
SELECT
  get_json_object(EventData, "$.EventName") AS EventName,
  get_json_object(EventData, "$.EventNamespace") AS EventNamespace,
  get_json_object(EventData, "$.EntityType") AS EntityType,
  get_json_object(EventData, "$.Source") AS Source,
  get_json_object(EventData, "$.EventId") AS EventId,
  get_json_object(EventData, "$.EntityId") AS EntityId,
  get_json_object(EventData, "$.SourceType") AS SourceType,
  get_json_object(EventData, "$.Timestamp") AS Timestamp,
  get_json_object(EventData, "$.PlayFabEnvironment.Vertical") AS PlayFabEnvironment_Vertical,
  get_json_object(EventData, "$.PlayFabEnvironment.Cloud") AS PlayFabEnvironment_Cloud,
  get_json_object(EventData, "$.PlayFabEnvironment.Application") AS PlayFabEnvironment_Application,
  get_json_object(EventData, "$.PlayFabEnvironment.Commit") AS PlayFabEnvironment_Commit,
  get_json_object(EventData, "$.MatchId") AS MatchId,
  get_json_object(EventData, "$.ItemName") AS ItemName
FROM
  STREAM(live.raw_events)
WHERE
  FullName_Name = 'ItemPickup'

-- COMMAND ----------

CREATE
OR REFRESH STREAMING LIVE TABLE bz_Elimination AS
SELECT
  get_json_object(EventData, "$.EventName") AS EventName,
  get_json_object(EventData, "$.EventNamespace") AS EventNamespace,
  get_json_object(EventData, "$.EntityType") AS EntityType,
  get_json_object(EventData, "$.Source") AS Source,
  get_json_object(EventData, "$.EventId") AS EventId,
  get_json_object(EventData, "$.EntityId") AS EntityId,
  get_json_object(EventData, "$.SourceType") AS SourceType,
  get_json_object(EventData, "$.Timestamp") AS Timestamp,
  get_json_object(EventData, "$.PlayFabEnvironment.Vertical") AS PlayFabEnvironment_Vertical,
  get_json_object(EventData, "$.PlayFabEnvironment.Cloud") AS PlayFabEnvironment_Cloud,
  get_json_object(EventData, "$.PlayFabEnvironment.Application") AS PlayFabEnvironment_Application,
  get_json_object(EventData, "$.PlayFabEnvironment.Commit") AS PlayFabEnvironment_Commit,
  get_json_object(EventData, "$.TargetId") AS TargetId,
  get_json_object(EventData, "$.TargetTeamId") AS TargetTeamId,
  get_json_object(EventData, "$.MatchId") AS MatchId,
  get_json_object(EventData, "$.InstigatorId") AS InstigatorId,
  get_json_object(EventData, "$.InstigatorTeamId") AS InstigatorTeamId
FROM
  STREAM(live.raw_events)
WHERE
  FullName_Name = 'Elimination'

-- COMMAND ----------

CREATE
OR REFRESH STREAMING LIVE TABLE bz_MatchStart AS
SELECT
  get_json_object(EventData, "$.EventName") AS EventName,
  get_json_object(EventData, "$.EventNamespace") AS EventNamespace,
  get_json_object(EventData, "$.EntityType") AS EntityType,
  get_json_object(EventData, "$.Source") AS Source,
  get_json_object(EventData, "$.EventId") AS EventId,
  get_json_object(EventData, "$.EntityId") AS EntityId,
  get_json_object(EventData, "$.SourceType") AS SourceType,
  get_json_object(EventData, "$.Timestamp") AS Timestamp,
  get_json_object(EventData, "$.PlayFabEnvironment.Vertical") AS PlayFabEnvironment_Vertical,
  get_json_object(EventData, "$.PlayFabEnvironment.Cloud") AS PlayFabEnvironment_Cloud,
  get_json_object(EventData, "$.PlayFabEnvironment.Application") AS PlayFabEnvironment_Application,
  get_json_object(EventData, "$.PlayFabEnvironment.Commit") AS PlayFabEnvironment_Commit,
  get_json_object(EventData, "$.MatchId") AS MatchId
FROM
  STREAM(live.raw_events)
WHERE
  FullName_Name = 'MatchStart'

-- COMMAND ----------

CREATE
OR REFRESH STREAMING LIVE TABLE bz_player_created AS
SELECT
  get_json_object(EventData, "$.Created") AS Created,
  get_json_object(EventData, "$.EventName") AS EventName,
  get_json_object(EventData, "$.PublisherId") AS PublisherId,
  get_json_object(EventData, "$.EventNamespace") AS EventNamespace,
  get_json_object(EventData, "$.EntityType") AS EntityType,
  get_json_object(EventData, "$.Source") AS Source,
  get_json_object(EventData, "$.TitleId") AS TitleId,
  get_json_object(EventData, "$.EntityId") AS EntityId,
  get_json_object(EventData, "$.EventId") AS EventId,
  get_json_object(EventData, "$.SourceType") AS SourceType,
  get_json_object(EventData, "$.Timestamp") AS Timestamp,
  get_json_object(EventData, "$.PlayFabEnvironment.Vertical") AS PlayFabEnvironment_Vertical,
  get_json_object(EventData, "$.PlayFabEnvironment.Cloud") AS PlayFabEnvironment_Cloud,
  get_json_object(EventData, "$.PlayFabEnvironment.Application") AS PlayFabEnvironment_Application,
  get_json_object(EventData, "$.PlayFabEnvironment.Commit") AS PlayFabEnvironment_Commit
FROM
  STREAM(live.raw_events)
WHERE
  FullName_Name = 'player_created'

-- COMMAND ----------

CREATE
OR REFRESH STREAMING LIVE TABLE bz_player_virtual_currency_balance_changed AS
SELECT
  get_json_object(EventData, "$.EventName") AS EventName,
  get_json_object(EventData, "$.VirtualCurrencyName") AS VirtualCurrencyName,
  get_json_object(EventData, "$.VirtualCurrencyBalance") AS VirtualCurrencyBalance,
  get_json_object(EventData, "$.VirtualCurrencyPreviousBalance") AS VirtualCurrencyPreviousBalance,
  get_json_object(EventData, "$.EventNamespace") AS EventNamespace,
  get_json_object(EventData, "$.EntityType") AS EntityType,
  get_json_object(EventData, "$.Source") AS Source,
  get_json_object(EventData, "$.TitleId") AS TitleId,
  get_json_object(EventData, "$.EntityId") AS EntityId,
  get_json_object(EventData, "$.EventId") AS EventId,
  get_json_object(EventData, "$.SourceType") AS SourceType,
  get_json_object(EventData, "$.Timestamp") AS Timestamp,
  get_json_object(EventData, "$.PlayFabEnvironment.Vertical") AS PlayFabEnvironment_Vertical,
  get_json_object(EventData, "$.PlayFabEnvironment.Cloud") AS PlayFabEnvironment_Cloud,
  get_json_object(EventData, "$.PlayFabEnvironment.Application") AS PlayFabEnvironment_Application,
  get_json_object(EventData, "$.PlayFabEnvironment.Commit") AS PlayFabEnvironment_Commit
FROM
  STREAM(live.raw_events)
WHERE
  FullName_Name = 'player_virtual_currency_balance_changed'

-- COMMAND ----------

CREATE
OR REFRESH STREAMING LIVE TABLE bz_player_added_title AS
SELECT
  get_json_object(EventData, "$.EventName") AS EventName,
  get_json_object(EventData, "$.Platform") AS Platform,
  get_json_object(EventData, "$.PlatformUserId") AS PlatformUserId,
  get_json_object(EventData, "$.EventNamespace") AS EventNamespace,
  get_json_object(EventData, "$.EntityType") AS EntityType,
  get_json_object(EventData, "$.Source") AS Source,
  get_json_object(EventData, "$.TitleId") AS TitleId,
  get_json_object(EventData, "$.EntityId") AS EntityId,
  get_json_object(EventData, "$.EventId") AS EventId,
  get_json_object(EventData, "$.SourceType") AS SourceType,
  get_json_object(EventData, "$.Timestamp") AS Timestamp,
  get_json_object(EventData, "$.PlayFabEnvironment.Vertical") AS PlayFabEnvironment_Vertical,
  get_json_object(EventData, "$.PlayFabEnvironment.Cloud") AS PlayFabEnvironment_Cloud,
  get_json_object(EventData, "$.PlayFabEnvironment.Application") AS PlayFabEnvironment_Application,
  get_json_object(EventData, "$.PlayFabEnvironment.Commit") AS PlayFabEnvironment_Commit
FROM
  STREAM(live.raw_events)
WHERE
  FullName_Name = 'player_added_title'

-- COMMAND ----------

CREATE
OR REFRESH STREAMING LIVE TABLE bz_player_linked_account AS
SELECT
  get_json_object(EventData, "$.EventName") AS EventName,
  get_json_object(EventData, "$.Origination") AS Origination,
  get_json_object(EventData, "$.OriginationUserId") AS OriginationUserId,
  get_json_object(EventData, "$.EventNamespace") AS EventNamespace,
  get_json_object(EventData, "$.EntityType") AS EntityType,
  get_json_object(EventData, "$.Source") AS Source,
  get_json_object(EventData, "$.TitleId") AS TitleId,
  get_json_object(EventData, "$.EntityId") AS EntityId,
  get_json_object(EventData, "$.EventId") AS EventId,
  get_json_object(EventData, "$.SourceType") AS SourceType,
  get_json_object(EventData, "$.Timestamp") AS Timestamp,
  get_json_object(EventData, "$.PlayFabEnvironment.Vertical") AS PlayFabEnvironment_Vertical,
  get_json_object(EventData, "$.PlayFabEnvironment.Cloud") AS PlayFabEnvironment_Cloud,
  get_json_object(EventData, "$.PlayFabEnvironment.Application") AS PlayFabEnvironment_Application,
  get_json_object(EventData, "$.PlayFabEnvironment.Commit") AS PlayFabEnvironment_Commit
FROM
  STREAM(live.raw_events)
WHERE
  FullName_Name = 'player_linked_account'

-- COMMAND ----------

CREATE
OR REFRESH STREAMING LIVE TABLE bz_player_logged_in AS
SELECT
  get_json_object(EventData, "$.EventName") AS EventName,
  get_json_object(EventData, "$.Platform") AS Platform,
  get_json_object(EventData, "$.PlatformUserId") AS PlatformUserId,
  get_json_object(EventData, "$.Location.ContinentCode") AS Location_ContinentCode,
  get_json_object(EventData, "$.Location.CountryCode") AS Location_CountryCode,
  get_json_object(EventData, "$.Location.City") AS Location_City,
  get_json_object(EventData, "$.Location.Latitude") AS Location_Latitude,
  get_json_object(EventData, "$.Location.Longitude") AS Location_Longitude,
  get_json_object(EventData, "$.IPV4Address") AS IPV4Address,
  get_json_object(EventData, "$.ExperimentVariants") AS ExperimentVariants,
  get_json_object(EventData, "$.EventNamespace") AS EventNamespace,
  get_json_object(EventData, "$.EntityType") AS EntityType,
  get_json_object(EventData, "$.Source") AS Source,
  get_json_object(EventData, "$.TitleId") AS TitleId,
  get_json_object(EventData, "$.EntityId") AS EntityId,
  get_json_object(EventData, "$.EventId") AS EventId,
  get_json_object(EventData, "$.SourceType") AS SourceType,
  get_json_object(EventData, "$.Timestamp") AS Timestamp,
  get_json_object(EventData, "$.PlayFabEnvironment.Vertical") AS PlayFabEnvironment_Vertical,
  get_json_object(EventData, "$.PlayFabEnvironment.Cloud") AS PlayFabEnvironment_Cloud,
  get_json_object(EventData, "$.PlayFabEnvironment.Application") AS PlayFabEnvironment_Application,
  get_json_object(EventData, "$.PlayFabEnvironment.Commit") AS PlayFabEnvironment_Commit
FROM
  STREAM(live.raw_events)
WHERE
  FullName_Name = 'player_logged_in'

-- COMMAND ----------

CREATE
OR REFRESH STREAMING LIVE TABLE bz_entity_created AS
SELECT
  get_json_object(EventData, "$.SchemaVersion") AS SchemaVersion,
  get_json_object(EventData, "$.FullName.Namespace") AS FullName_Namespace,
  get_json_object(EventData, "$.FullName.Name") AS FullName_Name,
  get_json_object(EventData, "$.Id") AS Id,
  get_json_object(EventData, "$.Timestamp") AS Timestamp,
  get_json_object(EventData, "$.Entity.Id") AS Entity_Id,
  get_json_object(EventData, "$.Entity.Type") AS Entity_Type,
  get_json_object(EventData, "$.Originator.Id") AS Originator_Id,
  get_json_object(EventData, "$.Originator.Type") AS Originator_Type,
  get_json_object(EventData, "$.EntityLineage.namespace") AS EntityLineage_namespace,
  get_json_object(EventData, "$.EntityLineage.title") AS EntityLineage_title,
  get_json_object(
    EventData,
    "$.EntityLineage.master_player_account"
  ) AS EntityLineage_master_player_account,
  get_json_object(
    EventData,
    "$.EntityLineage.title_player_account"
  ) AS EntityLineage_title_player_account,
  get_json_object(EventData, "$.PayloadContentType") AS PayloadContentType
FROM
  STREAM(live.raw_events)
WHERE
  FullName_Name = 'entity_created'

-- COMMAND ----------

CREATE
OR REFRESH STREAMING LIVE TABLE bz_entity_logged_in AS
SELECT
  get_json_object(EventData, "$.EventName") AS EventName,
  get_json_object(EventData, "$.Source") AS Source,
  get_json_object(EventData, "$.EntityChain") AS EntityChain,
  get_json_object(EventData, "$.EntityId") AS EntityId,
  get_json_object(EventData, "$.EntityLineage.NamespaceId") AS EntityLineage_NamespaceId,
  get_json_object(EventData, "$.EntityLineage.TitleId") AS EntityLineage_TitleId,
  get_json_object(
    EventData,
    "$.EntityLineage.MasterPlayerAccountId"
  ) AS EntityLineage_MasterPlayerAccountId,
  get_json_object(
    EventData,
    "$.EntityLineage.TitlePlayerAccountId"
  ) AS EntityLineage_TitlePlayerAccountId,
  get_json_object(EventData, "$.EventNamespace") AS EventNamespace,
  get_json_object(EventData, "$.EventId") AS EventId,
  get_json_object(EventData, "$.EntityType") AS EntityType,
  get_json_object(EventData, "$.SourceType") AS SourceType,
  get_json_object(EventData, "$.Timestamp") AS Timestamp,
  get_json_object(EventData, "$.PlayFabEnvironment.Vertical") AS PlayFabEnvironment_Vertical,
  get_json_object(EventData, "$.PlayFabEnvironment.Cloud") AS PlayFabEnvironment_Cloud,
  get_json_object(EventData, "$.PlayFabEnvironment.Application") AS PlayFabEnvironment_Application,
  get_json_object(EventData, "$.PlayFabEnvironment.Commit") AS PlayFabEnvironment_Commit
FROM
  STREAM(live.raw_events)
WHERE
  FullName_Name = 'entity_logged_in'

-- COMMAND ----------

CREATE
OR REFRESH STREAMING LIVE TABLE bz_MatchEnd AS
SELECT
  get_json_object(EventData, "$.EventName") AS EventName,
  get_json_object(EventData, "$.EventNamespace") AS EventNamespace,
  get_json_object(EventData, "$.EntityType") AS EntityType,
  get_json_object(EventData, "$.Source") AS Source,
  get_json_object(EventData, "$.EventId") AS EventId,
  get_json_object(EventData, "$.EntityId") AS EntityId,
  get_json_object(EventData, "$.SourceType") AS SourceType,
  get_json_object(EventData, "$.Timestamp") AS Timestamp,
  get_json_object(EventData, "$.PlayFabEnvironment.Vertical") AS PlayFabEnvironment_Vertical,
  get_json_object(EventData, "$.PlayFabEnvironment.Cloud") AS PlayFabEnvironment_Cloud,
  get_json_object(EventData, "$.PlayFabEnvironment.Application") AS PlayFabEnvironment_Application,
  get_json_object(EventData, "$.PlayFabEnvironment.Commit") AS PlayFabEnvironment_Commit,
  get_json_object(EventData, "$.Team0Score") AS Team0Score,
  get_json_object(EventData, "$.TargetScore") AS TargetScore,
  get_json_object(EventData, "$.MatchId") AS MatchId,
  get_json_object(EventData, "$.Team1Score") AS Team1Score,
  get_json_object(EventData, "$.WinningTeamId") AS WinningTeamId
FROM
  STREAM(live.raw_events)
WHERE
  FullName_Name = 'MatchEnd'

-- COMMAND ----------

CREATE
OR REFRESH STREAMING LIVE TABLE bz_api_operation AS
SELECT
  get_json_object(EventData, "$.SchemaVersion") AS SchemaVersion,
  get_json_object(EventData, "$.FullName.Namespace") AS FullName_Namespace,
  get_json_object(EventData, "$.FullName.Name") AS FullName_Name,
  get_json_object(EventData, "$.Id") AS Id,
  get_json_object(EventData, "$.Timestamp") AS Timestamp,
  get_json_object(EventData, "$.Entity.Id") AS Entity_Id,
  get_json_object(EventData, "$.Entity.Type") AS Entity_Type,
  get_json_object(EventData, "$.Originator.Id") AS Originator_Id,
  get_json_object(EventData, "$.Originator.Type") AS Originator_Type,
  get_json_object(EventData, "$.Payload.Created") AS Payload_Created,
  get_json_object(EventData, "$.Payload.Completed") AS Payload_Completed,
  get_json_object(EventData, "$.Payload.Operation") AS Payload_Operation,
  get_json_object(EventData, "$.Payload.OperationStatus") AS Payload_OperationStatus,
  get_json_object(EventData, "$.Payload.SegmentId") AS Payload_SegmentId,
  get_json_object(EventData, "$.Payload.SegmentName") AS Payload_SegmentName,
  get_json_object(EventData, "$.Payload.SegmentDefinition") AS Payload_SegmentDefinition,
  get_json_object(EventData, "$.EntityLineage.namespace") AS EntityLineage_namespace,
  get_json_object(EventData, "$.EntityLineage.title") AS EntityLineage_title,
  get_json_object(EventData, "$.PayloadContentType") AS PayloadContentType
FROM
  STREAM(live.raw_events)
WHERE
  FullName_Name = 'api_operation'
