# Databricks notebook source
# MAGIC %md
# MAGIC # Foundation Models for Email Response Automation

# COMMAND ----------

# MAGIC %md
# MAGIC Databricks Model Serving supports any Foundation Model, be it a fully custom model, a Databricks-managed model, or a third-party Foundation Model. This flexibility allows you to choose the right model for the right job, keeping you ahead of future advances in the range of available models. 
# MAGIC
# MAGIC You can get started with Foundation Model APIs on a pay-per-token basis, which significantly reduces operational costs. Alternatively, for workloads requiring fine-tuned models or performance guarantees, you can switch to Provisioned Throughput.
# MAGIC
# MAGIC Please refer Foundation Model API documentation for more details:
# MAGIC https://docs.databricks.com/en/machine-learning/foundation-models/index.html?_gl=1*licvcu*_gcl_au*ODA1MDc0NDEzLjE3MDMzMjEzNDk

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install the SDK on a Databricks Notebook

# COMMAND ----------

# MAGIC %pip install mlflow==2.9.0 langchain==0.0.344 databricks-sdk==0.12.0 

# COMMAND ----------

!pip install databricks-genai-inference

dbutils.library.restartPython()

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
# MAGIC ## Create Mistral-8X7b-instruct Completion endpoint for our solution
# MAGIC
# MAGIC We analysed multiple foundation models and Mistral is providing better results for the email automation in our case. However any foundation model can be used based on your preference and uniqueness of your data

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
# MAGIC ## Test one of the emails with the Mistral model API

# COMMAND ----------

# Retrieve a single review for testing
test_single_review = emails_silver.limit(2).select("email_body_clean").collect()[0][0]

# print(test_single_review)
# Predict on the review
response = llm_chain.run(test_single_review)

print(f"response.text:{response}")
