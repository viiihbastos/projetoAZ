# Databricks notebook source
# MAGIC %md The purpose of this notebook is to process Salesforce Marketing Cloud (email) tracking extracts in preparation for publication of summary metrics to the Amperity platform. This notebook was developed on a **Databricks 11.3 LTS** cluster.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC 
# MAGIC In this notebook, we will process daily extracts of tracking data originating from email-based customer marketing campaigns orchestrated through the Salesforce Marketing Cloud (SFMC).  Data from these campaigns are very useful for managing future marketing engagement efforts, making them a valuable input to customer data platform (CDP) systems through which these efforts are managed.
# MAGIC 
# MAGIC The challenge with these data are that they are organized around the scheduled unit of work executed in the SFMC system and not necessarily the customer to whom the email messages are being sent.  The data can be quite voluminous, depending on how jobs are configured, so that organizations will typically want to  restructure and aggregate information from the larger email tracking dataset to align it with the needs of CDP and the marketers that use it.
# MAGIC 
# MAGIC With this in mind, we will demonstrate how SFMC email tracking extracts can be processed within the Databricks lakehouse, making the complete set available for detailed process analysis, and then condensed for consumption by a CDP.  We will make use of Amperity's [Databricks data source](https://docs.amperity.com/datagrid/source_databricks.html) capability to demonstrate how these data can be integrated into a CDP. 

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pyspark.sql.functions as fn
from pyspark.sql.types import *

from delta.tables import *

from zipfile import ZipFile
from copy import deepcopy

# COMMAND ----------

# DBTITLE 1,Setup Notebook Configurations
# instantiate config variable
config = {}

# identify database
config['database'] = 'sfmc'

# identify moint point
config['mount point'] = '/mnt/tracking'

# identify key folder locations
config['incoming dir'] = f"{config['mount point']}/incoming"
config['temp dir'] = f"{config['incoming dir']}/temp"
config['inprocess dir']  = f"{config['incoming dir']}/inprocess"
config['archive dir'] = f"{config['mount point']}/archive"
config['raw dir'] = f"{config['mount point']}/raw"

# COMMAND ----------

# DBTITLE 1,Configure Target Database
# create database if not exists
_ = spark.sql('create database if not exists {0}'.format(config['database']))

# set current datebase context
_ = spark.catalog.setCurrentDatabase(config['database'])

# COMMAND ----------

# MAGIC %md ##Step 1: Access Tracking Data
# MAGIC 
# MAGIC Salesforce Marketing Cloud (SFMC) records information about various events related to email generated through its system through scheduled jobs, *i.e.* *send jobs*.  Periodic extracts of the event data can be [configured](https://help.salesforce.com/s/articleView?id=sf.mc_as_tracking_extract_config.htm) to be deposited in various containers including cloud object stores through the SFMC Automation Studio UI. (Alternative modes of extract including direct interactions with the Salesforce API or use of third-party tools that engage the API can also be used.) Here, we've used the Automation Studio to configure an extract to run daily:
# MAGIC </p>
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/sfmc_extract_1.png' width=750>
# MAGIC </p>
# MAGIC 
# MAGIC And that daily extract is sending files to the *incoming* folder of an Azure Storage account accessible to Databricks as a [mount point](https://learn.microsoft.com/en-us/azure/databricks/dbfs/mounts) that Databricks will recognize as */mnt/tracking* within its filesystem (not shown here):
# MAGIC </p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/sfmc_extract_23.png' width=750>
# MAGIC 
# MAGIC The data are extracted as a compressed ZIP file named **SMFC_Tracking_Extract_*MM-DD-YYYY*.zip** where the *MM*, *DD* and *YYYY* parts of the name represent the month, day and year of the extract, respectively.  Within each zip file are CSV files containing data about the sending jobs and events associated with these.  The CSV files, one for each event type associated with the extract and one for the sending jobs, are named as follows:</p>
# MAGIC 
# MAGIC * **SendJobs.csv** - represents all sends with activity logged against them for the timeframe of the extract
# MAGIC * **Sent.csv** - shows instances of an email send occurring during a specified date range
# MAGIC * **Opens.csv** - shows open instances of an HTML email during the date range
# MAGIC * **Clicks.csv** - shows all instances of a clicked link in an email occurring within the specified date range
# MAGIC * **Bounces.csv** - shows all the hard and soft bounces occurring within a specified date range
# MAGIC * **Unsubs.csv** -  includes information on unsubscribe requests that occurred during a specified data range
# MAGIC 
# MAGIC Many other tracking outputs are [available](https://help.salesforce.com/s/articleView?id=sf.mc_as_tracking_extract_output.htm) depending on the configuration of tracking extracts.

# COMMAND ----------

# MAGIC %md In order to make these data accessible to Databricks, we need to unzip each extract file delivered to the *incoming* folder.  Because each file extracted from the compressed file is named as shown in the last cell (without inclusion of date information or other unique identifiers in the file name), we must rename each extracted file to include the date part.  To facilitate this, we will need to extract each incoming ZIP file to a temporary folder, rename each file using the date information embedded in the original ZIP file's filename, and then move the renamed files to a staging folder for further processing. Should the process get backlogged, we can accumulate multiple days-worth of extracted files in the staging folder (which we will name *inprocess*) before proceeding.  To support this, we will implement a folder structure as follows:
# MAGIC </p>
# MAGIC 
# MAGIC ```
# MAGIC /incoming
# MAGIC    /temp
# MAGIC    /inprocess
# MAGIC /archive
# MAGIC /raw
# MAGIC ```
# MAGIC 
# MAGIC In this folder structure, the *raw* folder is where processed data files will reside.  Subfolders under it will identify various event types and the processed files for those event types, *i.e.* *sent*, *open*, *click*, *etc.*, will be placed in its associated subfolder. In the standard [medallion architecture](https://www.databricks.com/glossary/medallion-architecture#:~:text=A%20medallion%20architecture%20is%20a,Silver%20%E2%87%92%20Gold%20layer%20tables) employed in many lakehouse implementations, this will represent our Bronze layer. The *archive* folder is where processed ZIP files will be placed once *inprocess* data has been fully processed and files are transferred to *raw*.
# MAGIC 
# MAGIC The logic for the unzipping, renaming, and staging of files from the *incoming* folder to the *inprocess* subfolder is provided here:

# COMMAND ----------

# DBTITLE 1,Display Starting Folder Structure
def list_folder_contents(path, level=0):
  
  for file in dbutils.fs.ls(path):
    
    # print long file name
    print(file.path)

    # if directory
    if file.size==0:
      list_folder_contents(file.path, level+1)
      
list_folder_contents(config['mount point'])

# COMMAND ----------

# DBTITLE 1,Initialize Archive List
zips_to_archive = []

# COMMAND ----------

# DBTITLE 1,Process Incoming Extract Data
# for each incoming file
for i in dbutils.fs.ls(config['incoming dir']):
  
  # if zip file
  if i.name[-4:].lower() == '.zip':
    
    # get date string in yyyymmdd format from filename 
    dt = i.name[:-4].split('_')[-1] # split file name (exclude extension) on underscores
    dt = dt.split('-') # split date string on dash
    dt = dt[2]+dt[0]+dt[1] # rearrage date elements in yyyymmdd order
    
    # create clean temp folder for unzipped files
    dbutils.fs.rm(config['temp dir'], recurse=True) # drop folder if exists
    dbutils.fs.mkdirs(config['temp dir']) # create folder
    
    # unzip files to temp folder
    local_file_path = '/'+i.path.replace(':','') # get local file path dbfs:/... ==> /dbfs/
    with ZipFile(local_file_path, 'r') as z:
      z.extractall(f"/dbfs/{config['temp dir']}")
    
    # rename files in temp folder and move to inprocess folder
    for t in dbutils.fs.ls(config['temp dir']+'/'+i.name[:-4]):
      
      # determine where to land renamed file
      file_type = t.name[:-4].lower()
      target_dir = f"{config['inprocess dir']}/{file_type}"
      
      # move temp file to target location
      dbutils.fs.mv(
        t.path, 
        f'{target_dir}/{dt}.csv'
        )
      
    # add zip file to list of files to archive (later)
    zips_to_archive += [i.path]

# COMMAND ----------

# DBTITLE 1,Display Ending Folder Structure
list_folder_contents('/mnt/tracking')

# COMMAND ----------

# MAGIC %md ##Step 2: Process *SendJobs* Data
# MAGIC 
# MAGIC The structure of the send job extract files are documented [here](https://help.salesforce.com/s/articleView?id=sf.mc_as_sendjobs.htm).  We can define a schema aligned with the documented extract file structure as follows:

# COMMAND ----------

# DBTITLE 1,Define SendJobs Schema
# https://help.salesforce.com/s/articleView?id=sf.mc_as_sendjobs.htm
sendjobs_schema = StructType([
    StructField('ClientID', LongType()),
    StructField('SendID', LongType()),
    StructField('FromName', StringType()),
    StructField('FromEmail', StringType()),
    StructField('SchedTime', TimestampType()),
    StructField('SentTime', TimestampType()),
    StructField('Subject', StringType()),
    StructField('EmailName', StringType()),
    StructField('TriggeredSendExternalKey', StringType()),
    StructField('SendDefinitionExternalKey', StringType()),
    StructField('JobStatus', StringType()),
    StructField('PreviewURL', StringType()),
    StructField('IsMultipart', BooleanType()),
    StructField('Additional', StringType())
    ])

# COMMAND ----------

# MAGIC %md To read these data to a dataframe, we can simply point to the folder within which our newly arrived send job data resides.  In addition to reading the data using the expected schema structure information, we can add a few metadata fields such as source file name and processing times to assist with tracking of our ETL processes:

# COMMAND ----------

# DBTITLE 1,Read SendJobs Data
sendjobs_df = (
  spark
    .read
      .csv(
        path=f"{config['inprocess dir']}/sendjobs",
        sep=',',
        header=True,
        schema=sendjobs_schema,
        timestampFormat='M/d/yyyy h:mm:ss a' # datetimes format
        )
      .select(
        '*', # all incoming fields
        fn.input_file_name().alias('source_file'), # plus source file name 
        fn.current_timestamp().alias('processing_time') # plus processing time
        )
  )

display(sendjobs_df)

# COMMAND ----------

# MAGIC %md We can now write these data to tables defined within the Databricks catalog. Because these data may reflect updates to previously processed data, we may wish to perform this operation using a [merge](https://docs.databricks.com/delta/merge.html):

# COMMAND ----------

# DBTITLE 1,Merge SendJobs Data
# create table when not exists
_ = (
  DeltaTable.createIfNotExists(spark)
  .tableName(f"{config['database']}.sendjobs")
  .addColumns(sendjobs_df.schema)
  .execute()
  )  
  
# get reference to target table
target_table = DeltaTable.forName(spark, f"{config['database']}.sendjobs")

# perform merge
_ = (
  target_table.alias('target')
    .merge(
      sendjobs_df.alias('source'),
      '''
      target.ClientID=source.ClientID AND
      target.SendID=source.SendID
      ''' 
      )
    .whenMatchedUpdateAll()
    .whenNotMatchedInsertAll()
    .execute()
  )

# COMMAND ----------

# MAGIC %md We now have updated data in our *sendjobs* table:

# COMMAND ----------

# DBTITLE 1,Display Contents of SendJobs Table
# MAGIC %sql SELECT * FROM sendjobs;

# COMMAND ----------

# MAGIC %md ##Step 3: Process Remaining Events Data
# MAGIC 
# MAGIC We can now repeat this process for each of our event data files. Because each file has a similar base structure and requires pretty much the same processing, we can define our logic using a more generalizable approach.  
# MAGIC 
# MAGIC To support this, we will begin by defining the base schema shared by all event data files and then add the additional fields associated with each specific event type.  Please note, a Python *[deepcopy](https://docs.python.org/3/library/copy.html)* operation is used to ensure the addition of fields to the base schema does not affect subsequent steps:

# COMMAND ----------

# DBTITLE 1,Declare EventType Schemas
# base schema shared by all event type files
base_schema = StructType([
  StructField('ClientID', LongType()),
  StructField('SendID', LongType()),
  StructField('SubscriberKey', StringType()),
  StructField('EmailAddress', StringType()),
  StructField('SubscriberID', LongType()),
  StructField('ListID', LongType()),
  StructField('EventDate', TimestampType()),
  StructField('EventType', StringType())
  ])

# https://help.salesforce.com/s/articleView?id=sf.mc_as_bounces.htm
bounces_schema = deepcopy(base_schema)
_ = ( 
  bounces_schema
    .add(StructField('BounceCategory', StringType()))
    .add(StructField('SMTPCode', IntegerType()))
    .add(StructField('BounceReason', StringType()))
    .add(StructField('BatchID', StringType()))
    .add(StructField('TriggeredSendExternalKey', StringType()))
  )

# https://help.salesforce.com/s/articleView?id=sf.mc_as_clicks.htm
clicks_schema = deepcopy(base_schema) 
_ = ( 
  clicks_schema 
    .add(StructField('SendURLID', LongType()))
    .add(StructField('URLID', LongType()))
    .add(StructField('URL', StringType()))
    .add(StructField('Alias', StringType()))
    .add(StructField('BatchID', LongType()))
    .add(StructField('TriggeredSendExternalKey', StringType()))
    .add(StructField('Browser', StringType()))
    .add(StructField('EmailClient', StringType()))
    .add(StructField('OperatingSystem', StringType()))
    .add(StructField('Device', StringType()))
    )

# https://help.salesforce.com/s/articleView?id=sf.mc_as_opens.htm
opens_schema = deepcopy(base_schema)
_ = (
  opens_schema
    .add(StructField('BatchId', LongType()))
    .add(StructField('TriggeredSendExternalKey', StringType()))   
    )

# https://help.salesforce.com/s/articleView?id=sf.mc_as_sent
sent_schema = deepcopy(base_schema)
_ = (
  sent_schema
    .add(StructField('BatchId', LongType()))
    .add(StructField('TriggeredSendExternalKey', StringType()))   
    )

# https://help.salesforce.com/s/articleView?id=sf.mc_as_unsubs.htm
unsubs_schema = deepcopy(base_schema)
_ = (
  unsubs_schema
    .add(StructField('BatchId', LongType()))
    .add(StructField('TriggeredSendExternalKey', StringType()))   
    )

# COMMAND ----------

# MAGIC %md Now we can write a function to capture the generic logic used to read event type data.  Please note that this function has logic similar to what was used with *SendJobs* but does employ *dropDuplicates* to remove the occasional duplicate record in the dataset:

# COMMAND ----------

# DBTITLE 1,Function to Read Incoming Event Data
def read_events(schema, path, path_glob='*.csv'):
  ''' 
  This function is used to read newly arrived events  
  data in the processing folder to a Spark dataframe. 
  '''  
  
  df = (
    spark
      .read
      .csv( # instructions for reading each file
        path=path,
        sep=',',
        header=True,
        schema=schema,
        timestampFormat='M/d/yyyy h:mm:ss a'
        )
      .dropDuplicates(['ClientID','SendID','SubscriberKey','EmailAddress','SubscriberID','ListID','EventDate','EventType'])
      .select(
        '*', # all incoming fields
        fn.input_file_name().alias('source_file'), # plus source file name 
        fn.current_timestamp().alias('processing_time') # plus processing time
        )
      )

  return df

# COMMAND ----------

# MAGIC %md Similarly, we can write a function to capture our generic data writing (merging) operation.  This function is similar to the code used with the *SendJobs* data but differs in how source data are matched to target data in the merge join, reflecting the different keys used to identify unique event type records:

# COMMAND ----------

# DBTITLE 1,Function to Write Incoming Event Data
def write_events(source_df, table_name):
  ''' 
  This function is used to write events data to a
  table using a merge operation. 
  '''

  # create table when not exists
  _ = (
    DeltaTable.createIfNotExists(spark)
    .tableName(f"{config['database']}.{table_name}")
    .addColumns(source_df.schema)
    .execute()
    )

  # get reference to target table
  target_table = DeltaTable.forName(spark, f"{config['database']}.{table_name}")

  # perform merge
  _ = (
    target_table.alias('target')
      .merge(
        source_df.alias('source'),
        '''
        target.ClientID=source.ClientID AND 
        target.SendID=source.SendID AND 
        target.SubscriberKey=source.SubscriberKey AND
        target.EmailAddress=source.EmailAddress AND
        target.SubscriberID=source.SubscriberID AND
        target.ListID=source.ListID AND
        target.EventDate=source.EventDate AND
        target.EventType=source.EventType
        '''
        )
      .whenMatchedUpdateAll()
      .whenNotMatchedInsertAll()
      .execute()
    )

# COMMAND ----------

# MAGIC %md And now we can process each of our expected event type data files.  Please note that we are using the event type name to retrieve the schema associated with this event type using the Python vars function.  For this to work, we need to make sure our schema variables are all named *\<event_type\>_schema* in the exact case used by the items in our event type list:

# COMMAND ----------

# DBTITLE 1,Process Event Type Data
# for each event type
for event_type in ['bounces','clicks','opens','sent','unsubs']:
  
  # print event type so we can track progress
  print(event_type)

  # read inprocess event data to a data frame
  df = read_events(vars()[f"{event_type}_schema"], path=f"{config['inprocess dir']}/{event_type}")

  # merge the inprocess event data with the target table
  _ = write_events(df, event_type)
  

# COMMAND ----------

# MAGIC %md With event data processing complete, we can now examine the data in the various events table:

# COMMAND ----------

# DBTITLE 1,Display Bounces Data
# MAGIC %sql SELECT * FROM bounces;

# COMMAND ----------

# DBTITLE 1,Display Clicks Data
# MAGIC %sql SELECT * FROM clicks;

# COMMAND ----------

# DBTITLE 1,Display Opens Data
# MAGIC %sql SELECT * FROM opens;

# COMMAND ----------

# DBTITLE 1,Display Sent Data
# MAGIC %sql SELECT * FROM sent;

# COMMAND ----------

# DBTITLE 1,Display Unsubs Data
# MAGIC %sql SELECT * FROM unsubs;

# COMMAND ----------

# MAGIC %md ##Step 4: Move Processed Data Files
# MAGIC 
# MAGIC With our incoming data processed, we can now move the processed data files to the appropriate folder locations in our */raw* directory and archive the newly processed ZIP files:

# COMMAND ----------

# DBTITLE 1,Display Raw Dir Content Before Move
# create raw dir if not exists
dbutils.fs.mkdirs(config['raw dir'])

# list raw dir contents
list_folder_contents(config['raw dir'])

# COMMAND ----------

# DBTITLE 1,Move Processed Data Files to Raw Dir
dbutils.fs.mv(config['inprocess dir'], config['raw dir'], recurse=True)

# COMMAND ----------

# MAGIC %md **NOTE** Because this notebook is a demonstration, we will leave the ZIP files in place for the next run.

# COMMAND ----------

# DBTITLE 1,Archive Zip Files
## for each zip file
#for zf in zips_to_archive:
#  # move it to archive folder
#  dbutils.fs.mv(zf, f"{config['archive dir']}/{zf.split('/')[-1]}")

# COMMAND ----------

# DBTITLE 1,Display Raw Dir Content After Move
list_folder_contents(config['raw dir']) 

# COMMAND ----------

# MAGIC %md ##Step 5: Publish Summary Info
# MAGIC 
# MAGIC With data processing behind us, we now have a database with multiple tables capturing our Salesforce Marketing Cloud (SFMC) tracking data at the most granular level provided by that system:</p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/sfmc_schema.png' width=200>
# MAGIC 
# MAGIC While useful for a variety of purposes, our marketing analysts may only need summary metrics for each customer engaged in the last 7 days.  And they may not want all the details of that engagement but instead might want something like a simple count of event occurrences across this period.  With that need in mind, we might define a routine to capture such summary metrics in anticipation of them being read into the CDP used by these analysts:
# MAGIC 
# MAGIC **NOTE** We've broken this logic up to make it more readable, but it could be written much more efficiently.  It also could be expressed as a SQL statement which might be more easily employed in some environments.

# COMMAND ----------

# DBTITLE 1,Get Dates Over Which to Calculate Metrics
dates = (
  spark
    .table('sendjobs')
    .selectExpr('to_date(schedtime) as date')
    .groupBy()
      .agg(fn.max('date').alias('date')) # get max date in send jobs dataset
    .crossJoin(spark.range(7).withColumn('incr', fn.expr("-1 * cast(id as int)"))) # generate number of days back to examine
    .selectExpr('dateadd(date, incr) as eventdate') # calculate by increment back in time
    .orderBy('eventdate',ascending=False)
)
display(dates)

# COMMAND ----------

# DBTITLE 1,Calculate Event Metrics for Each Date
# assemble relevant event info
events = spark.createDataFrame([], schema='emailaddress string, eventtype string, eventdate date') # empty dataframe to initialize union loop
for event_type in ['bounces','clicks','opens','sent','unsubs']:
  tmp = (
    spark
      .table(event_type) # get event data
      .withColumn('eventdate',fn.expr('to_date(eventdate)')) # convert event datetimes to dates
      .join(dates, on='eventdate', how='leftsemi') # filter on event dates in dates dataset
      .select('emailaddress','eventtype','eventdate')
    )
  events = events.unionAll(tmp)

# count events by email and date (with a pivot on event type)
events = (
  events
    .groupBy('emailaddress','eventdate')
      .pivot('eventtype')
      .agg(fn.count('*'))
)


display(events)

# COMMAND ----------

# DBTITLE 1,Ensure Each Email Has One Record for Each Date
# full set of emails and dates for this period
base_set = (
    events
      .select('emailaddress')
      .distinct()
      .crossJoin(dates)
  )

# email-date metrics (one record for each email-date combination)
metrics = (
  base_set
    .join(events, on=['emailaddress','eventdate'],how='left')
    .fillna(0) # use 0 if no metric value for a given date
  )

display(
  metrics
    .orderBy('emailaddress','eventdate')
    )

# COMMAND ----------

# DBTITLE 1,Calculate 3- & 7-Day Metrics
summary = metrics

# calculate windowed values
for event_type in ['sent','open','click','unsub','bounce']:
  summary = (
    summary
      .withColumn(
        f"{event_type}_last3days", 
        fn.expr(f'sum({event_type}) over(partition by emailaddress order by eventdate rows between 2 preceding and current row)')
        )
      .withColumn(
        f"{event_type}_last7days", 
        fn.expr(f'sum({event_type}) over(partition by emailaddress order by eventdate rows between 6 preceding and current row)')
        )
      .drop(event_type)
    )

# write values for last date in range
summary = (
  summary
    .join( 
      dates.groupBy().agg(fn.max('eventdate').alias('eventdate')), # last date in set
      on='eventdate'
      )
    .fillna(0) # if no occurences, use a 0 for the metric
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('email_tracking_metrics')
  )

# display results
display(
  spark.table('email_tracking_metrics')
  )

# COMMAND ----------

# MAGIC %md ##Step 6: Read Metrics into the CDP
# MAGIC 
# MAGIC The previous data processing steps would typically be set on a fixed schedule to process data periodically after it arrives from the Salesforce Marketing Cloud. The final step in that scheduled process would update the metrics associated with customers in preparation for consumption by the Amperity CDP.
# MAGIC 
# MAGIC Leveraging knowledge of the data processing times, the Amperity CDP would schedule data retrieval from Databricks using it's [Databricks plugin](https://docs.amperity.com/datagrid/source_databricks.html):
# MAGIC </p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/sfmc_read_data2.png' width=850>
# MAGIC </p>
# MAGIC 
# MAGIC Using these data, the marketers can now define new customer segments based on receptiveness to email marketing and use that to drive additional customer engagement:
# MAGIC </p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/sfmc_segmentation.png' width=850>

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
