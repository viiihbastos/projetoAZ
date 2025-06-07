# Databricks notebook source
# MAGIC %md
# MAGIC # External Models for Email Response Automation

# COMMAND ----------

# MAGIC %pip install mlflow==2.9.0 langchain==0.0.344 databricks-sdk==0.12.0

# COMMAND ----------

# MAGIC %pip install mlflow[genai]>=2.9.0
# MAGIC %pip install --upgrade mlflow
# MAGIC %pip install --upgrade langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook, we will use External Models for automation of the email response. External models are third-party models hosted outside of Databricks. Supported by Model Serving, external models allow you to streamline the usage and management of various large language model (LLM) providers, such as OpenAI and Anthropic, within an organization. For this specific problem, we have picked OpenAI.
# MAGIC
# MAGIC https://docs.databricks.com/en/generative-ai/external-models/index.html

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
# MAGIC ## Setup Langchain for the external model
# MAGIC
# MAGIC We will setup Langchain to define the prompt template that will retrieve email Catagory, Sentiment, Synopsis and possible reply.
# MAGIC
# MAGIC The possible reply can be based on the templated and embedding can be used for it. However in this solution, we can not defined it.

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
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential
import os

spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch",2)
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
url = spark.conf.get("spark.databricks.workspaceUrl")



# Create Summary function
def run_summarisation_pipeline(email_body):

    os.environ['DATABRICKS_TOKEN'] = token
    
    # Chat GPT LLM Model
    gateway = Databricks(
    host="https://" + url, 
    endpoint_name="Email-OpenAI-Completion-Endpoint"
    )

    # Build Prompt Template
    prompt_template_string = """
    Given the following email text, categorise whether the email is a job request, customer query or generic email where no action required. It should capture sentiment of the email as positive, negative or neutral. Also it should create a short summary of the email. In addition, it should draft possible reply to email.

    The output should be structured as a JSON dictionary of dictionaries. First attribute name is "Category" which categorises of the email as three possible values - Job, Query or No Action. Second json attribute name is Sentiment with possible values - positive, negative or neutral. Third json attribute name is "Synopsis" which should capture short email summary. Forth JSON attribute name "Reply" should be possibly email reply to the original email.

    Email summary begin here: {email_body}"""

    prompt_template = PromptTemplate(template=prompt_template_string, input_variables=["email_body"])

    # Build LLM Chain
    chain = LLMChain(prompt=prompt_template, llm=gateway)

    @retry(wait=wait_exponential(multiplier=10, min=10, max=1800), stop=stop_after_attempt(7))
    def _call_with_retry(email_body):
        return chain.invoke(email_body)

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

# Repartition for concurrency
emails_silver = emails_silver.repartition(10)

# Make the requests
emails_silver_with_summary = (
    emails_silver
    .withColumn("summary_string", summarisation_udf(SF.col("email_body_clean")))
)


# Save table
# (
#     emails_silver_with_summary
#     .write
#     .mode("overwrite")
#     .option("overwriteSchema", "true")
#     .saveAsTable(config['table_emails_silver_externalm'])
# )

# COMMAND ----------

display(emails_silver_with_summary)

# COMMAND ----------

display(emails_silver_with_summary)

# COMMAND ----------


