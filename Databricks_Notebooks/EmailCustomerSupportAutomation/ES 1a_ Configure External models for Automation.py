# Databricks notebook source
# MAGIC %md
# MAGIC # External Models for Email Response Automation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install MLflow with external models support

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
# MAGIC ## Create OpenAI Completion endpoint for our solution

# COMMAND ----------

import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")
client.create_endpoint(
    name="Email-OpenAI-Completion-Endpoint",
    config={
        "served_entities": [{
            "external_model": {
                "name": "gpt-3.5-turbo-instruct",
                "provider": "openai",
                "task": "llm/v1/completions",
                "openai_config": {
                    "openai_api_key": "{{secrets/email_openai_secret_scope/openai_api_key}}",
                }
            }
        }]
    }
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Langchain for the external model
# MAGIC
# MAGIC We will setup Langchain to define the prompt template that will retrieve email Catagory, Sentiment, Synopsis and possible reply.
# MAGIC
# MAGIC The possible reply can be based on the templated and embedding can be used for it. However in this solution, we can not defined it.

# COMMAND ----------

import mlflow
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import Databricks

gateway = Databricks(
    host="https://" + spark.conf.get("spark.databricks.workspaceUrl"), 
    endpoint_name="Email-OpenAI-Completion-Endpoint",
    temperature=0.1,
)

# Build Prompt Template
template = """
Given the following email text, categorise whether the email is a job request, customer query or generic email where no action required. It should capture sentiment of the email as positive, negative or neutral. Also it should create a short summary of the email. In addition, it should draft possible reply to email.

The output should be structured as a JSON dictionary of dictionaries. First attribute name is "Category" which categorises of the email as three possible values - Job, Query or No Action. Second json attribute name is Sentiment with possible values - positive, negative or neutral. Third json attribute name is "Synopsis" which should capture short email summary. Forth JSON attribute name "Reply" should be possibly email reply to the original email.

Email summary begin here: {email_body}"""

prompt = PromptTemplate(template=template, input_variables=["email_body"])

# Build LLM Chain
llm_chain = LLMChain(prompt=prompt, llm=gateway)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test one of the emails with the OpenAI API

# COMMAND ----------

# Retrieve a single review for testing
test_single_review = emails_silver.limit(1).select("email_body_clean").collect()[0][0]

# print(test_single_review)
# Predict on the review
response_string = llm_chain.invoke(test_single_review)

# Print string
print(response_string)
