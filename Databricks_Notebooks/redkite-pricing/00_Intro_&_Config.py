# Databricks notebook source
# MAGIC %md The purpose of this notebook is to to set the configuration values used by the [Redkite](https://www.redkite.com/accelerators/pricing) Price Elasticity solution accelerator.  This notebook was developed using a Databricks 13.3 ML LTS cluster.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC In the CPG industry, price optimization plays a pivotal role in driving revenue growth and maximizing profitability. Pricing analysts scrutinize historical data to determine how changes in price as well as the prior state of the market ahead of a pricing change affect consumer response.  This requires a careful balancing of exploratory, predictive and what-if analysis as well as good judgement about other factors at play before organizations are able to make effective changes to their prices.
# MAGIC
# MAGIC Databricks provides an ideal environment for this work with its ability to not only process and prepare the large volumes of historical data that often go into these analysis, but the breadth of the analysis work that takes place on it as well.  Coupled with [RedKite's pricing solution](https://www.redkite.com/accelerators/pricing), analysts can more easily pour over this data to formulate a pricing direction for their products.  The purpose of the following notebooks is to examine some of the key forms of analysis at the heart of this solution.

# COMMAND ----------

# MAGIC %md ##Config

# COMMAND ----------

# DBTITLE 1,Initialize Config
if 'config' not in locals():
  config = {}

# COMMAND ----------

# DBTITLE 1,Database
# set database name
config['database name'] = 'redkite'

# create database to house data
_ = spark.sql('CREATE DATABASE IF NOT EXISTS {0}'.format(config['database name']))

# set database as default for queries
_ = spark.catalog.setCurrentDatabase(config['database name'] )

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
