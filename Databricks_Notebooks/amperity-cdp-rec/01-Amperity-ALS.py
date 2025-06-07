# Databricks notebook source
# MAGIC %md The purpose of this notebook is to generate recommendations for users in the Amperity platform.  This notebook was developed on a **Databricks ML 11.3 LTS** cluster. This solution accelerator notebook is also available at https://github.com/databricks-industry-solutions/amperity-cdp-rec

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC The data in the Amperity customer data platform (CDP) is useful for enabling a wide range of customer experiences. Customer identity resolution, data cleansing, and data unification provided by Amperity provides a very accurate, first-party customer data foundation which can be used to drive a wide range of customer insights and customer engagements. 
# MAGIC
# MAGIC Using these data, we may wish to estimate customer lifetime value, derive behavioral segments and estimate product propensities, all capabilities we can tap into with the Amperity CDP. For some capabilities, such as the generation of per-user product recommendations, we need to develop specialized models leveraging capabilities such as those found in Databricks. 
# MAGIC
# MAGIC In this notebook, we will demonstrate how to publish customer purchase history data from the Amperity CDP to Databricks to enable the training of a simple matrix factorization model. Recommendations produced by the model will then be published back to the CDP to enable any number of personalized interactions with our customers.  This notebook will borrow heavily from the previously published notebooks on matrix factorization available [here](https://www.databricks.com/blog/2023/01/06/products-we-think-you-might-generating-personalized-recommendations.html). Those interested into diving in the details of building such a model should review those notebooks and the accompanying blog.

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.ml.evaluation import RankingEvaluator
from pyspark.ml.recommendation import ALS

import mlflow

import numpy as np

import pyspark.sql.window as w
import pyspark.sql.functions as fn
from pyspark.sql.types import *

import random

# COMMAND ----------

# DBTITLE 1,Get Config Info
# instantiate config variable
config = {}

# identify database
config['database'] = 'amperity'

# name of model to register 
config['model name'] = 'amperity-als'

# set number of product recommendations to return for each user 
config['num_products_to_recommend'] = 25

# COMMAND ----------

# DBTITLE 1,Set Database and MLFlow Experiment
# create database if not exists
_ = spark.sql('create database if not exists {0}'.format(config['database']))

# set current datebase context
_ = spark.catalog.setCurrentDatabase(config['database'])

# set mlflow experiment in the user's folder
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment('/Users/{}/amperity_als'.format(username))

# COMMAND ----------

# MAGIC %md ##Step 1: Import Purchase History from Amperity
# MAGIC
# MAGIC Our first step starts in the Amperity environment. Here, we will configure Amperity to publish data to a Databricks Delta table.  Details on how to do this work are found [here](https://docs.amperity.com/datagrid/destination_databricks_delta_table.html). </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/amp-db-destination.png'>

# COMMAND ----------

# MAGIC %md With the Databricks Delta table destination configured, we can then build a SQL statement to access data from within the Amperity CDP:
# MAGIC </p>
# MAGIC
# MAGIC ```
# MAGIC SELECT
# MAGIC   amperity_id AS "amperity_id",
# MAGIC   order_datetime AS "order_datetime",
# MAGIC   product_id AS "product_id",
# MAGIC   item_quantity AS "item_quantity",
# MAGIC   item_subtotal AS "item_subtotal",
# MAGIC   item_revenue AS "item_revenue",
# MAGIC   order_id AS "order_id"
# MAGIC FROM
# MAGIC   Unified_Itemized_Transactions
# MAGIC WHERE order_datetime >= CAST(CURRENT_DATE - interval '1' year AS date)
# MAGIC ```
# MAGIC
# MAGIC With this query, we have retrieved some fields from the Amperity [*Unified_Itemized_Transactions (UIT)* table](https://docs.amperity.com/datagrid/table_unified_itemized_transactions.html). The UIT table contains every row of transactional data summarized to the item level.  The *amperity_id* field represents the resolved customer identity associated with each order.
# MAGIC
# MAGIC **NOTE** Please note that when setting up this query, you can setup an automatic push to the Databricks Delta table destination as part of the orchestration configuration settings.
# MAGIC
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/amp-select-uit-data.png'>

# COMMAND ----------

# MAGIC %md With our Amperity UIT data now published to Databricks as a table named *unified_itemized_transactions*, we can examine the table structure and the data within it:

# COMMAND ----------

# DBTITLE 1,Examine Structure of Published Table
# MAGIC %sql DESCRIBE EXTENDED unified_itemized_transactions;

# COMMAND ----------

# DBTITLE 1,Review Transaction Data
# MAGIC %sql SELECT * FROM unified_itemized_transactions;

# COMMAND ----------

# MAGIC %md ##Step 2: Assemble Ratings
# MAGIC
# MAGIC Our matrix factorization recommender requires three elements:
# MAGIC </p>
# MAGIC
# MAGIC * user identifier
# MAGIC * product identifier
# MAGIC * user-product rating
# MAGIC
# MAGIC These elements will form a matrix of known *ratings* from which we can extract latent factors.  Those latent factors can then be used to estimate ratings for user-product combinations not yet observed, providing the basis for our recommendations.
# MAGIC
# MAGIC The Spark implementation of this recommender is referred to as [*ALS*](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.recommendation.ALS.html), a reference to the *alternating least squares* algorithm used to perform the matrix factorization work. This algorithm requires user identities and product identities to be supplied as integer values.  To address this, we will quickly build as user and product lookup mapping the string-based *amperity_id* and *product_id* fields, representing our user and product identifiers respectively, to integer values.  Please note we are using an [identity column](https://www.databricks.com/blog/2022/08/08/identity-columns-to-generate-surrogate-keys-are-now-available-in-a-lakehouse-near-you.html) in our lookup table definition to auto-generate those integer values:

# COMMAND ----------

# DBTITLE 1,Build User Lookup
# MAGIC %sql
# MAGIC SET spark.databricks.delta.commitValidation.enabled=false; -- this setting is required for the table creation and insert below
# MAGIC
# MAGIC CREATE OR REPLACE TABLE user_lookup (
# MAGIC   user_id BIGINT GENERATED ALWAYS AS IDENTITY,
# MAGIC   amperity_id STRING
# MAGIC   );
# MAGIC   
# MAGIC INSERT INTO user_lookup (amperity_id)
# MAGIC SELECT DISTINCT 
# MAGIC   amperity_id
# MAGIC FROM unified_itemized_transactions
# MAGIC WHERE 
# MAGIC   amperity_id IS NOT NULL; -- identities which cannot be resolved will have an amperity_id of null

# COMMAND ----------

# DBTITLE 1,Build Product Lookup
# MAGIC %sql
# MAGIC
# MAGIC CREATE OR REPLACE TABLE item_lookup (
# MAGIC   item_id BIGINT GENERATED ALWAYS AS IDENTITY,
# MAGIC   product_id STRING
# MAGIC   );
# MAGIC   
# MAGIC INSERT INTO item_lookup (product_id)
# MAGIC SELECT DISTINCT 
# MAGIC 	product_id
# MAGIC FROM unified_itemized_transactions;

# COMMAND ----------

# MAGIC %md We can now retrieve our transaction data, substituting these integer-based identities for the original string identities:

# COMMAND ----------

# DBTITLE 1,Get Transaction History with Alt Identities
uit = (
   spark
    .table(f"{config['database']}.unified_itemized_transactions")
    .join(spark.table('user_lookup'), on='amperity_id', how='inner')
    .join(spark.table('item_lookup'), on='product_id', how='inner')
  )

display(uit)

# COMMAND ----------

# MAGIC %md We can now focus on our *ratings*.  As discussed in the solution accelerator referenced at the top of this notebook, we do not often have explicit ratings for each product a customer has purchased.  Using other transactional clues such as purchase frequencies or proportion of customer spend, we can derive implicit ratings which we can take as an indication of a customer's preference for a product.  Here we will use proportion of spend as our rating:

# COMMAND ----------

# DBTITLE 1,Calculate Ratings
ratings = (
  uit
    .filter( fn.expr("item_revenue > 0.0") ) # remove any records with $0.00 spend
    .selectExpr( # get just required fields
      'user_id', 
      'item_id', 
      'item_revenue as spend'
      ) 
    .groupBy( # summarize spending on items by user
      'user_id',
      'item_id'
      ) 
      .agg(fn.sum('spend').alias('spend'))
    .withColumn( # use spending ratio as indicator of preference (rating)
      'rating', 
      fn.expr('spend / SUM(spend) OVER(PARTITION BY user_id)') # item spend / total spend by customer
      ) 
    .select(
      'user_id',
      'item_id',
      'rating'
      )
    .orderBy('user_id','item_id')
  )

display(ratings)

# COMMAND ----------

# MAGIC %md ##Step 3: Train & Persist Model
# MAGIC
# MAGIC With our dataset prepared, we typically would perform a hyperparameter tuning exercise in order to identify an ideal set of hyperparameter values for our model. The identified values could then be used for multiple training/re-training cycles making this an occasional part of most recommender training workflows. You can review an example of what this would look like by reviewing the solution accelerator referenced earlier in this notebook.  But to keep our focus on our integration with Amperity, we will simply report the results of a hyperparameter tuning exercise performed outside of this notebook:

# COMMAND ----------

# DBTITLE 1,Get Hyperparameter Values
params =  {
  'alpha': 1.6934701426621088, 
  'regParam': 0.13875024435268196
  }

# COMMAND ----------

# MAGIC %md Using our hyperparameter values, we can train our model using code ported from the solution accelerator. Please note, this following step trains our model using the full set of ratings data available to us.  Increasing the size of your Databricks cluster will reduce its processing time:

# COMMAND ----------

# DBTITLE 1,Train Model
# number of product recommendations to use during evaluation
k = 10

# train model as part of mlflow experiment
with mlflow.start_run(run_name='als_full_model'):

  # instantiate & configure model
  als = ALS(
    rank=100,
    maxIter=50,
    userCol='user_id',  
    itemCol='item_id', 
    ratingCol='rating', 
    implicitPrefs=True,
    numItemBlocks=sc.defaultParallelism,
    numUserBlocks=sc.defaultParallelism,
    **params
    )

  # train model
  model = als.fit(ratings)

  # generate recommendations
  predicted = (
    model
      .recommendForAllUsers(k)
      .select( 
        'user_id',
        fn.posexplode('recommendations').alias('pos', 'rec') 
        )
      .withColumn('recs', fn.expr("collect_list(rec.item_id) over(partition by user_id order by pos)"))
      .groupBy('user_id')
        .agg( fn.max('recs').alias('recs'))
      .withColumn('prediction', fn.col('recs').cast('array<double>'))
    )
  
  # actual recommendations for a customer based on their expressed ratings
  actuals = (
    ratings
      .withColumn('selections', fn.expr("collect_list(item_id) over(partition by user_id order by rating desc)"))
      .filter(fn.expr(f"size(selections)<={k}"))
      .groupBy('user_id')
        .agg(
          fn.max('selections').alias('selections')
          )
      .withColumn('label', fn.col('selections').cast('array<double>'))
    )
  
  # perform evaluation
  eval = RankingEvaluator( 
    predictionCol='prediction',
    labelCol='label',
    metricName='precisionAtK',
    k=k
    )
  mapk = eval.evaluate( predicted.join(actuals, on='user_id') )

  # log model details
  mlflow.log_params(params)
  mlflow.log_metrics( {'map@k':mapk} )
  mlflow.spark.log_model(model, artifact_path='model', registered_model_name=config['model name'])

# COMMAND ----------

# MAGIC %md ##Step 5: Generate Recommendations 
# MAGIC
# MAGIC Using our trained model, we can now generate some number of product recommendations for each user in our dataset. Please note that these recommendations are returned as part of an array.  We are using the *posexplode* function to move each element in the array to a new record and then cleaning up the names of columns in the exploded data:

# COMMAND ----------

# DBTITLE 1,Generate Recommendations
raw_recommendations = (
  model
    .recommendForAllUsers(config['num_products_to_recommend'])
    .select(
      'user_id',
      fn.posexplode('recommendations').alias('rank','rec')
      )
    .withColumn('item_id', fn.col('rec.item_id'))
    .withColumn('rating', fn.col('rec.rating'))
    .select('user_id','item_id','rank','rating')
    ).cache()

display(raw_recommendations)

# COMMAND ----------

# MAGIC %md Before we can send this data back to Amperity, we need to replace our integer-based user and product identifiers with the original identifiers expected by the CDP:

# COMMAND ----------

# DBTITLE 1,Convert Integer IDs to Original Values
recommendations = (
  raw_recommendations
    .join( spark.table('user_lookup'), on='user_id', how='inner')
    .join( spark.table('item_lookup'), on='item_id', how='inner')
    .select('amperity_id','product_id','rank','rating')
  )

display(recommendations)

# COMMAND ----------

# MAGIC %md Lastly, we might take advantage of the processing power of the Databricks environment to enhance our data.  Here we have flagged each user-product recommendation with the last date any recommended product was purchased by the user.  This might be used to control which specific recommendations are sent to customers when:

# COMMAND ----------

# DBTITLE 1,Enhance Recommendations
# get user's last purchase for each product
last_purchase = (
  spark
    .table('unified_itemized_transactions')
    .groupBy('amperity_id','product_id')
      .agg( fn.max('order_datetime').alias('last_order_datetime') )
  )

# enhance recommendations with last purchase date info
enhanced_recommendations = (
  recommendations
    .join( last_purchase, on=['amperity_id','product_id'], how='left')
    .select('amperity_id','product_id','rank','rating','last_order_datetime')
  )

display(
  enhanced_recommendations
  )

# COMMAND ----------

# MAGIC %md We can now persist our recommendations ahead of publication to Amperity:

# COMMAND ----------

# DBTITLE 1,Persist Enhanced Recommendations
_ = (
  enhanced_recommendations
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .saveAsTable('uit_recommendations')
  )

# COMMAND ----------

# MAGIC %md ##Step 6: Publish Recommendations to Amperity
# MAGIC
# MAGIC Once the recommendations data has been generated you can publish the recommendation data available to Amperity for a variety of activation use cases. Configure the [Pull from Databricks](https://docs.amperity.com/datagrid/source_databricks.html) integration to retrieve the data from the set of recommendations persisted in the last step.
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/db-feed-ingest-config.png'>

# COMMAND ----------

# MAGIC %md With the recommendations now residing in Amperity and flagged with users' *amperity_id* values, our recommendations can now serve as an extension of our 360-degree customer view and be used to enable new modes of targeted engagement.

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
