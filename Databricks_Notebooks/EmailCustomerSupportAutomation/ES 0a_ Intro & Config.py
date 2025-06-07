# Databricks notebook source
# MAGIC %md 
# MAGIC #Reducing time-to-resolution for email customer support using LLMs

# COMMAND ----------

# MAGIC %md
# MAGIC The purpose of this notebook is to introduce the automation of customer support thorugh email. We use Large Langauge models (LLM) to automation the the email response process. You may find this notebook at https://github.com/sachinpatilb/PersonalRepo/tree/master/EmailSummary

# COMMAND ----------

# MAGIC %md
# MAGIC ### Introduction
# MAGIC Organizations often receive customer support inquiries through email channels. These emails need to adhere to service level agreements (SLAs) set by the organization or regulators, with penalties for failing to meet SLAs and rewards for exceeding them. Providing faster and more effective responses to customer inquiries enhances customer experience. However, many organizations struggle to meet SLAs due to the high volume of emails received and limited staff resources to respond to them.
# MAGIC
# MAGIC This solution accelerator proposes to use Large Language Models (LLMs) to automate the email response process. It involves the following key activities:
# MAGIC
# MAGIC 1. Categorization: The first step is to categorize the emails to understand the customer requests and urgency, including associated SLAs, and determine the appropriate approach for responding. Emails can be categorized as queries about the product, specific job requests, or generic emails that don't require a response.
# MAGIC 2. Sentiment Analysis: The sentiment of the email - positive, neutral, or negative - is analyzed.
# MAGIC 3. Synopsis: A summary of the email is created to help customer support professionals quickly understand its contents without reading the entire email.
# MAGIC 4. Automated Email Response: Based on the email's category, sentiment, and analysis, an automated customer email response is generated. 
# MAGIC
# MAGIC
# MAGIC As part of the solution, we present two approaches to deploy the solution on the Databricks Data Intelligence Platform:
# MAGIC 1. Proprietary SaaS LLMs: One approach is to call proprietary SaaS LLMs APIs from the Databricks Data Intelligence platform. This is convenient as it eliminates the need to train the model from scratch. Instead, pre-trained models can be used to classify and categorize emails.
# MAGIC 2. Open LLMs: The second approach involves deploying the solution within the organization's infrastructure as emails can contain sensitive information that cannot be shared outside the organization. This can be achieved using existing open LLM models such as LLAMA, Mosaic, etc. Fine-tuning of these models can be done using Databricks Mosaic models. This option provides more control over the infrastructure and can also reduce costs.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Solution Design/architecture
# MAGIC
# MAGIC In this section, let's describe the solution design and its

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration Settings
# MAGIC
# MAGIC The following represent configuration settings used across various notebooks in this solution accelerator. You should read through all the notebooks to understand how the configuration settings are used before making any changes to the values below.

# COMMAND ----------

# DBTITLE 1,Instantiate Config Variable
if 'config' not in locals().keys():
  config = {}

# COMMAND ----------

# DBTITLE 1,Database and Volume
config['catalog'] = 'email_summary_llm_solution'
config['schema'] = 'email_llm'
config['volume'] = 'source_data'
config['vol_data_landing'] = f"/Volumes/{config['catalog']}/{config['schema']}/{config['volume']}"
config['table_emails_bronze'] = 'emails_bronze'
config['table_emails_silver_foundationalm'] = 'emails_foundational_silver'
config['table_emails_silver_externalm'] = 'emails_externalm_silver'
config['table_emails_silver'] = 'emails_silver'

# COMMAND ----------

# create catalog if not exists
spark.sql('create catalog if not exists {0}'.format(config['catalog']))

# set current catalog context
spark.sql('USE CATALOG {0}'.format(config['catalog']))

# create database if not exists
spark.sql('create database if not exists {0}'.format(config['schema']))

# set current datebase context
spark.sql('USE {0}'.format(config['schema']))

# COMMAND ----------

# DBTITLE 1,Storage (see notebook 0b for more info)
config['mount_point'] ='/tmp/emails_summary'
 
# file paths
config['checkpoint_path'] = config['mount_point'] + '/checkpoints'
config['schema_path'] = config['mount_point'] + '/schema'

# COMMAND ----------

config['openai_api_key']=dbutils.secrets.get(scope = "email_openai_secret_scope", key = "openai_api_key")
