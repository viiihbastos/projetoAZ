# Databricks notebook source
# MAGIC %md The purpose of this notebook is to provide access to configuration data used by the Product Description Generation solution accelerator.  This notebook was developed on a **Databricks ML 14.3 LTS GPU-enabled** cluster.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC With this solution accelerator, we will show how generative AI can be used to help organizations efficiently generate product descriptions, *i.e.* product copy, for use on an ecommerce site.  The overarching workflow supported across these notebooks can be visualized as follows:
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/product_copy_workflow.png' width=600>
# MAGIC
# MAGIC
# MAGIC
# MAGIC The first step will be to extract a basic description of a product from an image.  This basic description as well as metadata provided by the item supplier will then be used to generate a robust product description using a high-powered LLM.  This work will be performed in three separate notebooks focused on:
# MAGIC </p>
# MAGIC
# MAGIC 1. Data Preparation (02_Data_Prep)
# MAGIC 2. Image Description Extraction (03_Extract_Image_Descriptions)
# MAGIC 3. Description Generation (04_Generate_Product_Descriptions)
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md ##Configuration
# MAGIC
# MAGIC The following parameters are used throughout the notebooks to control the resources being used.  If you modify these variables, please note that markdown in the notebooks may refer to the original values associated with these:

# COMMAND ----------

# DBTITLE 1,Initialize Config Variables
if 'config' not in locals().keys():
  config = {}

# COMMAND ----------

# DBTITLE 1,Database
# set catalog
config['catalog'] = 'tristen'
try:
  _ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {config['catalog']}")
except:
  pass
_ = spark.sql(f"USE CATALOG {config['catalog']}")

# set schema
config['schema'] = 'rtl_fashion_gen'
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config['schema']}")
_ = spark.sql(f"USE SCHEMA {config['schema']}")

# COMMAND ----------

# DBTITLE 1,Storage
config['file_path'] = f"/Volumes/{config['catalog']}/{config['schema']}/input_data"

# COMMAND ----------

# MAGIC %md © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | Transformers | State-of-the-art Machine Learning for JAX, PyTorch and TensorFlow | Apache 2.0 | https://pypi.org/project/transformers/ |
# MAGIC | Salesforce/instructblip-flan-t5-xl  | InstructBLIP model using Flan-T5-xl as language model | MIT | https://huggingface.co/Salesforce/instructblip-flan-t5-xl |
# MAGIC | Llama2-70B-Chat | A pretrained and fine-tuned generative text model with 70 billion parameters |  Meta |https://ai.meta.com/resources/models-and-libraries/llama-downloads                       |
