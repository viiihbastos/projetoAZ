# Databricks notebook source
# MAGIC %md The purpose of this notebook is to combine image descriptions and product metadata to create a final product description as part of  with the Product Description Generation solution accelerator.  This notebook was developed on a **Databricks ML 14.3 LTS GPU-enabled** cluster.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC In this final notebook, we will combine product information with our extracted description and use all of this data as inputs to a large language model, instructing it to generate product copy for us. 

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install databricks-genai-inference
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pyspark.sql.functions as fn

from databricks_genai_inference import ChatCompletion

# COMMAND ----------

# DBTITLE 1,Get Configuration Settings
# MAGIC %run "./01_Intro_and_Config"

# COMMAND ----------

# MAGIC %md ##Step 1: Assemble Product Attributes
# MAGIC
# MAGIC Our first step is to assemble the relevant product attributes into a simple to read list.  This information needs to be captured as a string so that we can easily submit it to the large language model:

# COMMAND ----------

# DBTITLE 1,Assemble Product Attributes
attributes = (
  spark
    .table('product_info')
    .join(spark.table('image_descriptions'), on='path')
    .selectExpr( """
          concat_ws(
            ', ',
            concat('gender: ', gender),
            concat('productType: ', productType),
            concat('colour: ', colour),
            concat('usage: ', usage),
            concat('name: ', productTitle),
            concat('description: "', description, '"')
            ) as attributes""", 
          'path'
          )
)

attributes.createOrReplaceTempView('attributes')


display(attributes)

# COMMAND ----------

# MAGIC %md ##Step 2: Generate Descriptions
# MAGIC
# MAGIC Our next step is to use a large language model to produce a description for each of our products.  We will use the Meta Llama2 70B model due to its recent popularity.  This model is hosted within the Databricks environment as part of the experimental [foundation model API](https://docs.databricks.com/en/machine-learning/foundation-models/index.html) infrastructure. Because of rate limiting and the general performance characteristics of this large language model, you will notice that this step takes a bit of time to complete:
# MAGIC
# MAGIC **NOTE** You might consider limiting the data used in this step if running this as part of a demonstration.

# COMMAND ----------

# DBTITLE 1,Create Table to Capture Descriptions
# MAGIC %sql
# MAGIC
# MAGIC CREATE OR REPLACE TABLE product_descriptions(
# MAGIC   path string,
# MAGIC   iteration int,
# MAGIC   description string
# MAGIC );

# COMMAND ----------

# DBTITLE 1,Generate Multiple Descriptions for Each Product
# get products and attributes to generate descriptions for
attributes_pd = attributes.limit(10).toPandas()

# configs for looping
max_n = 3
results = []


for i, row in attributes_pd.iterrows():

  # let the developer know which row we are on
  print(i,end='\r')

  # get data from row
  path = row['path']
  attribs = row['attributes']

  # create n number of variations for this product
  for n in range(max_n):

    # generate description using product attributes
    response = ChatCompletion.create(
      model="llama-2-70b-chat",
      messages=[
        {"role": "system", "content": "You are friendly, playful assistant providing a useful description of this product based on supplied characteristics. Keep your answers to 100 words or less. Do not use emojis in your response."},
        {"role": "user","content": attribs}
        ],
      temperature=0.5 # elevate temperature to get more varied results
      )
    description = response.message

    # capture output for later
    results += [(path, n, description)]

# save data to table for later use
_ = (
  spark
    .createDataFrame(results, schema="path string, iteration int, description string")
    .write
      .format('delta')
      .mode('append')
      .saveAsTable('product_descriptions')
  )

# COMMAND ----------

# DBTITLE 1,Review Descriptions
# MAGIC %sql
# MAGIC
# MAGIC SELECT 
# MAGIC   b.content, 
# MAGIC   a.iteration,
# MAGIC   a.description
# MAGIC FROM product_descriptions a 
# MAGIC INNER JOIN product_images b 
# MAGIC   ON a.path=b.path
# MAGIC ORDER BY b.content, a.iteration

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT 
# MAGIC   b.content, 
# MAGIC   a.iteration,
# MAGIC   c.description as extracted_description,
# MAGIC   a.description as draft_copy
# MAGIC FROM product_descriptions a 
# MAGIC INNER JOIN product_images b 
# MAGIC   ON a.path=b.path
# MAGIC INNER JOIN image_descriptions c 
# MAGIC   ON a.path=c.path
# MAGIC ORDER BY b.content, a.iteration

# COMMAND ----------

# MAGIC %md From a quick review of the decriptions generated, you can see some terms the LLM is gravitating towards in the product copy.  Altering the system prompt and playing with model parameters can help the workflow create different results.  There is also a bit of randomness within the model so that generating multiple descriptions for a single item (as demonstrated in the queries above) and then having a reviewer combine the bits and pieces they like from each can be a great way to arrive at unique descriptions.

# COMMAND ----------

# MAGIC %md © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | Transformers | State-of-the-art Machine Learning for JAX, PyTorch and TensorFlow | Apache 2.0 | https://pypi.org/project/transformers/ |
# MAGIC | Salesforce/instructblip-flan-t5-xl  | InstructBLIP model using Flan-T5-xl as language model | MIT | https://huggingface.co/Salesforce/instructblip-flan-t5-xl |
# MAGIC | Llama2-70B-Chat | A pretrained and fine-tuned generative text model with 70 billion parameters |  Meta |https://ai.meta.com/resources/models-and-libraries/llama-downloads                       |
