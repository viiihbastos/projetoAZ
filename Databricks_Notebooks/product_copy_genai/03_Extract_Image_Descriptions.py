# Databricks notebook source
# MAGIC %md The purpose of this notebook is to extract descriptions from images as part of  with the Product Description Generation solution accelerator.  This notebook was developed on a **Databricks ML 14.3 LTS GPU-enabled** cluster.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC In this notebook, we will generate basic descriptions for each of the images read in the prior notebook.  These descriptions will serve as a critical input to our final noteobook.

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install transformers
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from io import BytesIO
from PIL import Image
import torch

import pandas as pd
from typing import Iterator
import os

import pyspark.sql.functions as fn

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./01_Intro_and_Config"

# COMMAND ----------

# MAGIC %md ##Step 1: Test Description Extraction
# MAGIC
# MAGIC In this first step, we will extract an image from our table of product images:

# COMMAND ----------

# DBTITLE 1,Get a Single Image 
image = (
  spark
    .table('product_images')
    .select('path','content')
    .limit(1)
).collect()[0]

image

# COMMAND ----------

# MAGIC %md We will then install the [Salesforce/instructblip-flan-t5-xl model](https://huggingface.co/Salesforce/instructblip-flan-t5-xl) which has been trained with image and text description data to associate image structures with text:

# COMMAND ----------

# DBTITLE 1,Load the Image-to-Text Model
# Load the appropriate model from transformers into context. We also need to tell it what kind of device to use.
model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl", load_in_8bit=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# COMMAND ----------

# MAGIC %md We can now define a function to accept the binary contents of an image and extract from it a text description.  Please note that the model settings are simply suggestions and that you may want to adjust some values to get a response better aligned with your specific needs:

# COMMAND ----------

# DBTITLE 1,Define Function to Extract Description from Image
def get_description(img):
  "Convert an image binary and generate a description"
  image = Image.open(BytesIO(img)) # This loads the image from the binary type into a format the model understands.

  # Additional prompt engineering represents one of the easiest areas to experiment with the underlying model behavior.
  prompt = "Describe the image using tags from the fashion industry? Mention style and type. Please be concise"
  inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

  # Model parameters can be tuned as desired.
  outputs = model.generate(
          **inputs,
          do_sample=True,
          num_beams=5,
          max_length=256,
          min_length=1,
          top_p=0.9,
          repetition_penalty=1.5,
          length_penalty=1.0,
          temperature=1,
  )
  # We need to decode the outputs from the model back to a string format.
  generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
  return(generated_text)

# COMMAND ----------

# MAGIC %md We can now test passing our single image to our function to see how it behaves:

# COMMAND ----------

# DBTITLE 1,Test the Generation of a Description from an Image
# get description
description = get_description(image['content'])

# print discription and display image
print(
  Image.open(BytesIO(image['content']))
  )
print(description)

# COMMAND ----------

# MAGIC %md ##Step 2: Generate Descriptions for All Images
# MAGIC
# MAGIC With the ability of the model to extract a description from an image demonstrated, let's turn our attention to the definition of a function that will allow us to scale this process.  For this, we'll use a pandas UDF pattern that accepts grouped data.  In a later step, we will divide our data into relatively even groups in order to distribute the work across our Databricks cluster.
# MAGIC
# MAGIC To work in this mode, our function needs to accept a pandas dataframe containing one or more rows of data.  For each row in this dataframe, a function will be applied to extract the corresponding image.  The function will return a set of data with the image's key, *i.e.* path, and its description:

# COMMAND ----------

# DBTITLE 1,Define Function to Extract Descriptions from Images
def get_descriptions(inputs):

  # only get fields that are needed
  df = inputs[['path','content']]

  # Load the appropriate model from transformers into context. We also need to tell it what kind of device to use.
  model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
  processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl", load_in_8bit=True)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)

 # INTERNAL FUNCTION WITH LOGIC FOR DESCRIPTION GENERATION
  def _get_description(img):
    "Convert an image binary and generate a description"
    image = Image.open(BytesIO(img)) # This loads the image from the binary type into a format the model understands.

    # Additional prompt engineering represents one of the easiest areas to experiment with the underlying model behavior.
    prompt = "Describe the image using tags from the fashion industry? Mention style and type. Please be concise"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    # Model parameters can be tuned as desired.
    outputs = model.generate(
            **inputs,
            do_sample=True,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
    )
    # We need to decode the outputs from the model back to a string format.
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return(generated_text)
  
  # get description
  df['description'] = df['content'].apply(_get_description)

  # return only required fields
  return df[['path','description']] 

# COMMAND ----------

# MAGIC %md We will now read our images, assign an incremental id to each one, and then distribute the data across a fixed number of buckets.  To each bucket of data, we will apply our function and then persist the results to a table:
# MAGIC
# MAGIC **NOTE** This steps may take quite some time depending on the number of images you are processing and the size of your cluster.  Consider filtering the data or otherwise limiting the number of rows if you are simply working through a demonstration. 

# COMMAND ----------

# DBTITLE 1,Extract & Persist Descriptions from Images
image_descriptions = (
  spark
    .table('product_images')
    .select('path','content')
    .withColumn('id',fn.expr("ROW_NUMBER() OVER(ORDER BY path)")) # give each image an incremental id
    .withColumn('bucket', fn.expr("id % 100")) # divide ids into 100 buckets
    .drop('id')
    .groupBy('bucket') # group the data by bucket
      .applyInPandas(get_descriptions, schema='path string, description string') # apply the function
    .write # persist the resulting information
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('image_descriptions')
  )

# COMMAND ----------

# MAGIC %md We can now verify the resulting descriptions:

# COMMAND ----------

# DBTITLE 1,Review the Image Descriptions
# MAGIC %sql SELECT * FROM image_descriptions

# COMMAND ----------

# MAGIC %md © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | Transformers | State-of-the-art Machine Learning for JAX, PyTorch and TensorFlow | Apache 2.0 | https://pypi.org/project/transformers/ |
# MAGIC | Salesforce/instructblip-flan-t5-xl  | InstructBLIP model using Flan-T5-xl as language model | MIT | https://huggingface.co/Salesforce/instructblip-flan-t5-xl |
# MAGIC | Llama2-70B-Chat | A pretrained and fine-tuned generative text model with 70 billion parameters |  Meta |https://ai.meta.com/resources/models-and-libraries/llama-downloads                       |
