# Databricks notebook source
# MAGIC %md
# MAGIC #Implement a LangChain Agent
# MAGIC
# MAGIC So far we have created all the building blocks of our application as Lang Chain tools but assembled the application as a MLflow Python Function model. This gives us predictability and best latency for user questions.
# MAGIC
# MAGIC But, we can also use the LangChain AgentExecutor that use an LLM as a reasoning engine to determine which actions to take and what the inputs to those actions should be. 

# COMMAND ----------

# MAGIC %md
# MAGIC ###Import all the Tools
# MAGIC
# MAGIC Let us import all the tools that we created so that we can provide that to the Agent Executor

# COMMAND ----------

# MAGIC %run "./05_Create All Tools and Model"

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create Agent

# COMMAND ----------

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage

class CareCostReactAgent:
    
    max_tokens=4096
    temperature=0.01
    invalid_question_category = {
        "PROFANITY": "Content has inappropriate language",
        "RACIAL": "Content has racial slur.",
        "RUDE": "Content has angry tone and has unprofessional language.",
        "IRRELEVANT": "The question is not about a medical procedure cost.",
        "GOOD": "Content is a proper question about a cost of medical procedure."
    }
    agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
            "You are a helpful assistant who can answer questions about medical procedure costs.\
                Use tools to complete the request. Remember that client_id is not same as member_id.\
                    Call all the tools with the correct input arguments.\
                        Especially, Member Cost Calculator takes three input arguments:\
                            the benefit object, procedure_cost and member_deductibles which is dictionary of member deductibles\
                                In case of error, check the error message and correct the error and retry the tool call",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    def __init__(self, model_config:dict):

        #Instantiate tools
        self.model_config = model_config

        self.member_id_retriever = MemberIdRetriever(model_endpoint_name=model_config["member_id_retriever_model_endpoint_name"] ).get()

        self.question_classifier = QuestionClassifier(model_endpoint_name=model_config["question_classifier_model_endpoint_name"],
                                        categories_and_description=self.invalid_question_category
                                    ).get()
        
        self.client_id_lookup = ClientIdLookup(fq_member_table_name=model_config["member_table_name"]).get()
        
        self.benefit_rag = BenefitsRAG(model_endpoint_name=model_config["benefit_retriever_model_endpoint_name"],
                                retriever_config=RetrieverConfig(**model_config["benefit_retriever_config"])
                                ).get()
        
        self.procedure_code_retriever = ProcedureRetriever(retriever_config=
                                                           RetrieverConfig(**model_config["procedure_code_retriever_config"])).get()

        self.procedure_cost_lookup = ProcedureCostLookup(fq_procedure_cost_table_name=model_config["procedure_cost_table_name"]).get()

        self.member_accumulator_lookup = MemberAccumulatorsLookup(fq_member_accumulators_table_name=
                                                                  model_config["member_accumulators_table_name"]).get()

        self.member_cost_calculator = MemberCostCalculator().get()

        self.summarizer = ResponseSummarizer(model_endpoint_name=model_config["summarizer_model_endpoint_name"]).get()

        self.tools = [
            self.member_id_retriever,
            self.question_classifier,
            self.client_id_lookup,
            self.benefit_rag,
            self.procedure_code_retriever,
            self.procedure_cost_lookup,
            self.member_accumulator_lookup,
            self.member_cost_calculator,
            self.summarizer
        ]

        self.chat_model = ChatDatabricks(
            endpoint=model_config["agent_chat_model_endpoint_name"],
            max_tokens = self.max_tokens,
            temperature=self.temperature
        )

        self.agent = create_tool_calling_agent(self.chat_model,
            self.tools,
            prompt = self.agent_prompt
        )
        
        self.agent_executor = AgentExecutor(agent=self.agent, 
                                            tools=self.tools,
                                            handle_parsing_errors=True,
                                            verbose=True,
                                            max_iterations=20)

    def answer(self, member_id:str ,input_question:str) -> str:
        return self.agent_executor.invoke({
            "input": f"My member_id is {member_id}. Question:{input_question}"
        })





# COMMAND ----------

def get_model_config(db_host_url:str,
                       environment:str,
                       catalog:str,
                       schema:str,
                       
                       member_table_name:str,
                       procedure_cost_table_name:str,
                       member_accumulators_table_name:str,
                       
                       vector_search_endpoint_name:str,
                       sbc_details_table_name:str,
                       sbc_details_id_column:str,
                       sbc_details_retrieve_columns:[str],

                       cpt_code_table_name:str,
                       cpt_code_id_column:str,
                       cpt_code_retrieve_columns:[str],

                       agent_chat_model_endpoint_name:str,
                       member_id_retriever_model_endpoint_name:str,
                       question_classifier_model_endpoint_name:str,
                       benefit_retriever_model_endpoint_name:str,
                       summarizer_model_endpoint_name:str,

                       default_parameter_json_string:str) -> dict:
    
    fq_member_table_name = f"{catalog}.{schema}.{member_table_name}"
    fq_procedure_cost_table_name = f"{catalog}.{schema}.{procedure_cost_table_name}"
    fq_member_accumulators_table_name = f"{catalog}.{schema}.{member_accumulators_table_name}"      

    benefit_rag_retriever_config = RetrieverConfig(vector_search_endpoint_name=vector_search_endpoint_name,
                                vector_index_name=f"{catalog}.{schema}.{sbc_details_table_name}_index",
                                vector_index_id_column=sbc_details_id_column, 
                                retrieve_columns=sbc_details_retrieve_columns)

    proc_code_retriever_config = RetrieverConfig(vector_search_endpoint_name=vector_search_endpoint_name,
                                vector_index_name=f"{catalog}.{schema}.{cpt_code_table_name}_index",
                                vector_index_id_column=cpt_code_id_column,
                                retrieve_columns=cpt_code_retrieve_columns)

    return {
        "db_host_url":db_host_url,
        "environment" : "dev",
        "default_parameter_json_string" : default_parameter_json_string, #'{"member_id":"1234"}',
        "agent_chat_model_endpoint_name":agent_chat_model_endpoint_name,
        "member_id_retriever_model_endpoint_name":member_id_retriever_model_endpoint_name,
        "question_classifier_model_endpoint_name":question_classifier_model_endpoint_name,
        "benefit_retriever_model_endpoint_name":benefit_retriever_model_endpoint_name,
        "benefit_retriever_config":benefit_rag_retriever_config.dict(),
        "procedure_code_retriever_config":proc_code_retriever_config.dict(),
        "member_table_name":fq_member_table_name,
        "procedure_cost_table_name":fq_procedure_cost_table_name,
        "member_accumulators_table_name":fq_member_accumulators_table_name,
        "summarizer_model_endpoint_name":summarizer_model_endpoint_name
    }

# COMMAND ----------

# MAGIC %md
# MAGIC Let us create an External Model endpoint to use an Open AI `gpt 4o` model as our Agent Orchestrator.
# MAGIC
# MAGIC External models are third-party models hosted outside of Databricks. Supported by Model Serving, external models allow you to streamline the usage and management of various large language model (LLM) providers, such as OpenAI and Anthropic, within an organization. You can also use Mosaic AI Model Serving as a provider to serve custom models, which offers rate limits for those endpoints. As part of this support, Model Serving offers a high-level interface that simplifies the interaction with these services by providing a unified endpoint to handle specific LLM-related requests.
# MAGIC
# MAGIC See Tutorial: [Create external model endpoints to query OpenAI models](https://docs.databricks.com/en/generative-ai/tutorials/external-models-tutorial.html) for step-by-step guidance on external model endpoint creation and querying supported models served by those endpoints using the MLflow Deployments SDK.
# MAGIC
# MAGIC Let us name the model endpoint `carecost_openai_endpoint`

# COMMAND ----------


care_cst_agent = CareCostReactAgent(model_config=get_model_config(db_host_url=db_host_url,
                                environment="dev",
                                catalog=catalog,
                                schema=schema,
                                member_table_name= member_table_name,
                                procedure_cost_table_name=procedure_cost_table_name,
                                member_accumulators_table_name=member_accumulators_table_name,
                                vector_search_endpoint_name = "care_cost_vs_endpoint",
                                sbc_details_table_name=sbc_details_table_name,
                                sbc_details_id_column="id",
                                sbc_details_retrieve_columns=["id","content"],
                                cpt_code_table_name=cpt_code_table_name,
                                cpt_code_id_column="id",
                                cpt_code_retrieve_columns=["code","description"],
                                agent_chat_model_endpoint_name="carecost_openai_endpoint",  #<< The external open AI endpoint
                                member_id_retriever_model_endpoint_name="databricks-mixtral-8x7b-instruct",
                                question_classifier_model_endpoint_name="databricks-meta-llama-3-1-70b-instruct",
                                benefit_retriever_model_endpoint_name= "databricks-meta-llama-3-1-70b-instruct",
                                summarizer_model_endpoint_name="databricks-dbrx-instruct",                       
                                default_parameter_json_string='{"member_id":"1234"}'))

# COMMAND ----------

agent_response = care_cst_agent.answer(member_id = "1234", input_question="What is the total cost of a shoulder mri")

# COMMAND ----------

agent_response["output"]

# COMMAND ----------


