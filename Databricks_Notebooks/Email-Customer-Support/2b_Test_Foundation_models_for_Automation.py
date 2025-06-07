# Databricks notebook source
# MAGIC %md
# MAGIC # Test Foundation Models for Email Response Automation

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Prerequisite: Make sure to run 1_Ingest_Emails_Into_Lakehouse before running this notebook.
# MAGIC
# MAGIC Databricks Model Serving supports any Foundation Model, be it a fully custom model, a Databricks-managed model, or a third-party Foundation Model. This flexibility allows you to choose the right model for the right job, keeping you ahead of future advances in the range of available models. 
# MAGIC
# MAGIC You can get started with Foundation Model APIs on a pay-per-token basis, which significantly reduces operational costs. We recommend provisioned throughput for production workloads that need performance guarantees.
# MAGIC
# MAGIC Please refer <a href="https://docs.databricks.com/en/machine-learning/foundation-models/index.html?_gl=1*licvcu*_gcl_au*ODA1MDc0NDEzLjE3MDMzMjEzNDk" target="_blank">Foundation Model API</a> documentation for more details.
# MAGIC
# MAGIC Key highlights for this notebook:
# MAGIC - Note that Foundation model endpoint are already available in Databricks Serving and it is not required to be configured
# MAGIC - We analysed multiple foundation models and Mistral is providing better results for the email automation in our case. However any foundation model can be used based on your preference and uniqueness of your data
# MAGIC
# MAGIC Click on "Serving" in the left navigation to see what foundation models are available out of the box.

# COMMAND ----------

# MAGIC %run ./_resources/00-setup

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
# MAGIC ## Mistral-8X7b-instruct Completion endpoint for our solution
# MAGIC
# MAGIC We analysed multiple foundation models and Mistral is providing better results for the email automation in our case. However any foundation model can be used based on your preference and uniqueness of your data. 
# MAGIC
# MAGIC We recommend using <a href="https://www.databricks.com/blog/announcing-mlflow-28-llm-judge-metrics-and-best-practices-llm-evaluation-rag-applications-part" target="_blank">LLM-as-a-judge</a>  to evaluate models. More documentation on LLM evaluation is available <a href="https://docs.databricks.com/en/mlflow/llm-evaluate.html" target="_blank">here</a> .

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Langchain for the Foundation model
# MAGIC
# MAGIC We will setup Langchain to define the prompt template that will retrieve email Catagory, Sentiment, Synopsis and draft reply.
# MAGIC
# MAGIC The draft reply can be based on the templates and embedding can be used for it. However in this solution, we have not defined it.

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
Given the following email text, categorise whether the email is a job request, customer query or generic email where no action required. It should capture sentiment of the email as positive, negative or neutral. Also it should create a short summary of the email. In addition, it should draft reply to email. the output of the questions should only be a JSON dictionary of dictionaries.

The output should be structured as a JSON dictionary of dictionaries. First attribute name is "Category" which categorises the email as three possible values - Job, Query or No Action. Second json attribute name is Sentiment with possible values - positive, negative or neutral. Third json attribute name is "Synopsis" which should capture short email summary in 2-3 lines. Fourth JSON attribute name "Reply" should be draft email reply to the original email.
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

# COMMAND ----------


