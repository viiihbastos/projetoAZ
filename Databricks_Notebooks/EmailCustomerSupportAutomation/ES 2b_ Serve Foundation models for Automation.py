# Databricks notebook source
# MAGIC %md
# MAGIC # External Models for Email Response Automation

# COMMAND ----------

# MAGIC %pip install mlflow==2.9.0 langchain==0.0.344 databricks-sdk==0.12.0

# COMMAND ----------

!pip install databricks-genai-inference

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook, we will use Foundation Models for automation of the email response. 

# COMMAND ----------

# MAGIC %run "./ES 0a: Intro & Config"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read the emails from the bronze layer

# COMMAND ----------

# MAGIC %md
# MAGIC Lets read the raw emails persisted in the bronze layer in a Dataframe.
# MAGIC

# COMMAND ----------

emails_silver=spark \
                .read \
                .table(config['table_emails_bronze'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Langchain for the Foundation model
# MAGIC
# MAGIC We will setup Langchain to define the prompt template that will retrieve email Catagory, Sentiment, Synopsis and possible reply.
# MAGIC
# MAGIC The possible reply can be based on the templated and embedding can be used for it. However in this solution, we can not defined it.

# COMMAND ----------

import mlflow
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import Databricks
from langchain.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser

prompt = PromptTemplate(
  input_variables = ["question"],
  template = "You are an assistant. Give a short answer to this question: {question}"
)
chat_model = ChatDatabricks(endpoint="databricks-mixtral-8x7b-instruct", max_tokens = 500)

# Build Prompt Template
template = """
<s>[INST] <<SYS>>
Given the following email text, categorise whether the email is a job request, customer query or generic email where no action required. It should capture sentiment of the email as positive, negative or neutral. Also it should create a short summary of the email. In addition, it should draft possible reply to email. the output of the questions should only be a JSON dictionary of dictionaries

The output should be structured as a JSON dictionary of dictionaries. First attribute name is "Category" which categorises of the email as three possible values - Job, Query or No Action. Second json attribute name is Sentiment with possible values - positive, negative or neutral. Third json attribute name is "Synopsis" which should capture short email summary in 2-3 lines. Forth JSON attribute name "Reply" should be possibly email reply to the original email.
<</SYS>>
Email summary begin here DO NOT give answer except a JSON and No other text : {email_body}  [/INST] """

prompt = PromptTemplate(template=template, input_variables=["email_body"])

# Build LLM Chain
llm_chain = LLMChain(prompt=prompt, llm=chat_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create Email Summarisation UDF

# COMMAND ----------

import mlflow
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import Databricks
from langchain import OpenAI, PromptTemplate, LLMChain
from pyspark.sql import functions as SF
from pyspark.sql.types import StringType
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser
import os


spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch",2)
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Create Summary function
def run_summarisation_pipeline(email_body):
    
    os.environ['DATABRICKS_TOKEN'] = token

    # Chat GPT LLM Model
    chat_model = ChatDatabricks(endpoint="databricks-mixtral-8x7b-instruct", max_tokens = 500)

    # Build Prompt Template
    template = """
    <s>[INST] <<SYS>>
    Given the following email text, categorise whether the email is a job request, customer query or generic email where no action required. It should capture sentiment of the email as positive, negative or neutral. Also it should create a short summary of the email. In addition, it should draft possible reply to email. the output of the questions should only be a JSON dictionary of dictionaries

    The output should be structured as a JSON dictionary of dictionaries. First attribute name is "Category" which categorises of the email as three possible values - Job, Query or No Action. Second json attribute name is Sentiment with possible values - positive, negative or neutral. Third json attribute name is "Synopsis" which should capture short email summary in 2-3 lines. Forth JSON attribute name "Reply" should be possibly email reply to the original email.
    <</SYS>>
    Email summary begin here DO NOT give answer except a JSON and No other text : {email_body}  [/INST] """

    prompt = PromptTemplate(template=template, input_variables=["email_body"])

    # Build LLM Chain
    chain = LLMChain(prompt=prompt, llm=chat_model)

    @retry(wait=wait_exponential(multiplier=10, min=10, max=1800), stop=stop_after_attempt(7))
    def _call_with_retry(email_body):
        return chain.run(email_body)

    try:
        summary_string = _call_with_retry(email_body)
    except Exception as e:
        summary_string = f"FAILED: {e.last_attempt.exception()}"

    return summary_string


# Create UDF
summarisation_udf = SF.udf(
    lambda x: run_summarisation_pipeline(
        email_body=x
    ),
    StringType(),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Process incoming data

# COMMAND ----------

# Make the requests
emails_silver_with_summary = (
    emails_silver
    .withColumn("summary_string", summarisation_udf(SF.col("email_body_clean")))
)


# Save table
(
    emails_silver_with_summary
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(config['table_emails_silver_foundationalm'])
)
