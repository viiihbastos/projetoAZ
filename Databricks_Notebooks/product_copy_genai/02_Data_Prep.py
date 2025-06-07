# Databricks notebook source
# MAGIC %md The purpose of this notebook is to load the dataset that will be used with the Product Description Generation solution accelerator.  This notebook was developed on a **Databricks ML 14.3 LTS GPU-enabled** cluster.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC In this first notebook, we will read both product metadata and product images into persisted tables.  These tables will provide us access to the data needed for subsequent notebooks in the accelerator.
# MAGIC
# MAGIC The dataset we are using is the [e-commerce product images dataset](https://www.kaggle.com/datasets/vikashrajluhaniwal/fashion-images) available on Kaggle.  The dataset is provided under a public domain Creative Commons 1.0 license and consists of 2,900 product images and associated product metadata.
# MAGIC
# MAGIC Before running this accelerator, the files associated with this dataset should be uploaded to an [external volume](https://learn.microsoft.com/en-us/azure/databricks/connect/unity-catalog/volumes) you have configured within your environment.  The path to this folder is configured in notebook 00. Because the dataset is downloaded as a ZIP file, you may find that uploading the ZIP file to a Databricks accessible path and then unzipping it there might be easiest.  To support this, you may wish to run a BASH script in a new notebook cell similar to what is shown here:
# MAGIC
# MAGIC ```
# MAGIC %sh
# MAGIC
# MAGIC cd /Volumes/tristen/rtl_fashion_gen/input_data # this is the path of the folder where the zip file resides
# MAGIC unzip /Volumes/tristen/rtl_fashion_gen/input_data/archive.zip 
# MAGIC ```
# MAGIC Once this script is run, the assets that we will access will reside in a subfolder named *data*.

# COMMAND ----------

# DBTITLE 1,Get Config
# MAGIC %run "./01_Intro_and_Config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pyspark.sql.functions as fn

# COMMAND ----------

# MAGIC %md ##Step 1: Load Image Data
# MAGIC
# MAGIC To load the images into a persisted table, we will recursively read the contents starting with the base-level folder associated with our dataset.  We will limit the data we upload to those files with a *jpg* extension as this is consistent with the contents of the downloaded dataset:

# COMMAND ----------

# DBTITLE 1,Product Images
# read jpg image files
images = (
  spark
    .read
    .format("binaryFile") # read file contents as binary
    .option("recursiveFileLookup", "true") # recursive navigation of folder structures
    .option("pathGlobFilter", "*.jpg") # read only files with jpg extension
    .load(f"{config['file_path']}/data") # starting point for accessing files
  )

# write images to persisted table
_ = (
  images
    .write
    .mode("overwrite")
    .format("delta")
    .saveAsTable("product_images")
)

# display data in table
display(
  spark
    .read
    .table("product_images")
    )

# COMMAND ----------

# MAGIC %md ##Step 2: Load Product Info
# MAGIC
# MAGIC We will now read the product information found in the *fashion.csv* file associated with this dataset:

# COMMAND ----------

# DBTITLE 1,Read Product Metadata
# read metadata file
info = (
  spark
    .read
    .format("csv")
    .option("header", True)
    .option("delimiter", ",")
    .load(f"{config['file_path']}/data/fashion.csv")
  ) 

# display data
display(info)

# COMMAND ----------

# MAGIC %md Each product is associated with one and only one image.  The path to the specific image in the storage environment is based on the product category, and gender associated with each item.  Each file is named for the product id with which it associated.  This path was automatically captured when we read the images into a dataframe in the prior step.  Here, we will need to construct the path to provide us a key on which to link product information and the images: 

# COMMAND ----------

# DBTITLE 1,Persist with DBFS Path to Image
_= (
  info # add field for dbfs path to image file
    .withColumn('path', 
                fn.concat(
                   fn.lit('dbfs:'),
                  fn.lit(f"{config['file_path']}/data/"), 
                  'category', fn.lit("/"),
                  'gender', fn.lit("/"),
                  fn.lit("Images/images_with_product_ids/"),
                  'productid', fn.lit('.jpg')
                )
      )
    .write # write data to persisted table
      .mode("overwrite")
      .option('overwriteSchema','true')
      .format("delta")
      .saveAsTable("product_info")
)

# review data in table
display(
  spark.table('product_info')
  )

# COMMAND ----------

# MAGIC %md ##Step 3: Examine the Data Set
# MAGIC
# MAGIC A quick review of the data reveals we have 2,906 products in our dataset and one image for each:

# COMMAND ----------

# DBTITLE 1,Count of Products & Images
# MAGIC %sql 
# MAGIC
# MAGIC SELECT 
# MAGIC   COUNT(a.path) as products,
# MAGIC   COUNT(b.path) as images
# MAGIC FROM product_info a
# MAGIC FULL OUTER JOIN product_images b
# MAGIC    ON a.path=b.path

# COMMAND ----------

# MAGIC %md We can break down the data based on category and gender which may be helpful should we wish to subset our data to use different prompts or otherwise limit the scope of later steps:

# COMMAND ----------

# DBTITLE 1,Breakdown of Top-Level Categorizations
# MAGIC %sql
# MAGIC
# MAGIC SELECT 
# MAGIC   a.category,
# MAGIC   a.gender,
# MAGIC   COUNT(*) as instances
# MAGIC FROM product_info a
# MAGIC GROUP BY a.category, a.gender WITH CUBE
# MAGIC ORDER BY 1, 2;

# COMMAND ----------

# MAGIC %md © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | Transformers | State-of-the-art Machine Learning for JAX, PyTorch and TensorFlow | Apache 2.0 | https://pypi.org/project/transformers/ |
# MAGIC | Salesforce/instructblip-flan-t5-xl  | InstructBLIP model using Flan-T5-xl as language model | MIT | https://huggingface.co/Salesforce/instructblip-flan-t5-xl |
# MAGIC | Llama2-70B-Chat | A pretrained and fine-tuned generative text model with 70 billion parameters |  Meta |https://ai.meta.com/resources/models-and-libraries/llama-downloads                       |
