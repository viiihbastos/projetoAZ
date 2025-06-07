# Databricks notebook source
# MAGIC %md
# MAGIC # Annotating Data & Training Custom NER Model for Social Determinants of Health (SDOH) Prediction
# MAGIC
# MAGIC ## Why SDOH?
# MAGIC
# MAGIC ![SDOH Image](https://www.cdc.gov/publichealthgateway/images/sdoh/SDOH-icons.png?_=88110)
# MAGIC
# MAGIC Social Determinants of Health (SDOH) encapsulate the conditions under which individuals are born, grow, live, work, and age. These factors encompass socioeconomic status, education, neighborhood and physical environment, employment, and social support networks, among others. SDOH wield a significant influence over health outcomes and disparities. Individuals from disadvantaged backgrounds often confront obstacles in accessing quality healthcare, wholesome nourishment, safe housing, and education. This culminates in divergent health outcomes, with marginalized communities experiencing elevated rates of chronic illnesses, morbidity, and mortality.
# MAGIC
# MAGIC Acknowledging the impact of SDOH underscores a more holistic approach to healthcare. It shifts the focus beyond mere medical interventions, ensuring that individuals receive comprehensive care that accounts for their broader life circumstances. Moreover, addressing SDOH can engender effective preventive measures. By targeting the foundational causes of health inequalities, communities can mitigate the prevalence of specific diseases and conditions. Prioritizing SDOH also holds potential to enhance population health on a large scale by instituting policies and programs that tackle the underlying social determinants.
# MAGIC
# MAGIC ## Named Entity Recognition (NER) for SDOH Analysis
# MAGIC
# MAGIC ![NER Image](https://www.johnsnowlabs.com/wp-content/uploads/2023/03/1_7Q7YFyA7tKNgVAKJHlD3Ig.webp)
# MAGIC
# MAGIC Named Entity Recognition (NER) stands as a crucial task in natural language processing, entailing the identification and categorization of specific entities (such as names of individuals, locations, organizations, etc.) within textual data. Within the context of SDOH, NER extraction serves several pivotal purposes. SDOH insights often lie embedded in unstructured clinical notes, a treasure trove of patient information. Employing NER to extract SDOH entities transforms these unstructured notes into structured data. This structured format facilitates seamless processing, analysis, and utilization for research, interventions, and policy-making. NER extraction acts as a cornerstone, unearthing invaluable insights from unstructured clinical narratives pertaining to SDOH, thereby translating textual data into actionable intelligence. This paves the way for more effective interventions and a profound comprehension of the interplay between social determinants and health.
# MAGIC
# MAGIC For more detailed information on SDOH NER models, refer to [this medium article](https://www.johnsnowlabs.com/extract-social-determinants-of-health-entities-from-clinical-text-with-healthcare-nlp/).
# MAGIC
# MAGIC ## Tutorial Objectives
# MAGIC
# MAGIC In this tutorial, we will delve into a straightforward yet potent strategy for pre-annotating data using a predefined list of words or phrases. This method can significantly expedite your workflow when dealing with NLP tasks. After pre-annotation, we will guide you through the process of uploading these preliminary annotations to [NLP Lab](https://www.johnsnowlabs.com/nlp-lab/), an exceptional tool tailored for Natural Language Processing endeavors.
# MAGIC
# MAGIC Our subsequent steps will outline how to conduct annotations within [NLP Lab](https://www.johnsnowlabs.com/nlp-lab/) and subsequently download these annotations for further use. The final section of the tutorial will walk you through the process of training a customized Named Entity Recognition (NER) model using the annotated data.
# MAGIC
# MAGIC Please note: The [NLP Lab](https://www.johnsnowlabs.com/nlp-lab/) module is accessible in Spark NLP for Healthcare version 4.2.2 and beyond.
# MAGIC
# MAGIC ## Key Steps in This Tutorial:
# MAGIC
# MAGIC 1. Employing string matching and existing predefined vocabularies to establish a rudimentary pipeline for initial results.
# MAGIC 2. Uploading the initial outcomes to [NLP Lab](https://www.johnsnowlabs.com/nlp-lab/), conducting annotations, and subsequently downloading these annotations.
# MAGIC 3. Training an NER model using the annotated dataset to achieve enhanced performance.
# MAGIC
# MAGIC ## License
# MAGIC
# MAGIC Copyright / License info of the notebook. Copyright [2022] the Notebook Authors.  The source in this notebook is provided subject to the [Apache 2.0 License](https://spdx.org/licenses/Apache-2.0.html).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC |Library Name|Library License|Library License URL|Library Source URL|
# MAGIC | :-: | :-:| :-: | :-:|
# MAGIC |Pandas |BSD 3-Clause License| https://github.com/pandas-dev/pandas/blob/master/LICENSE | https://github.com/pandas-dev/pandas|
# MAGIC |Numpy |BSD 3-Clause License| https://github.com/numpy/numpy/blob/main/LICENSE.txt | https://github.com/numpy/numpy|
# MAGIC |Apache Spark |Apache License 2.0| https://github.com/apache/spark/blob/master/LICENSE | https://github.com/apache/spark/tree/master/python/pyspark|
# MAGIC |Requests|Apache License 2.0|https://github.com/psf/requests/blob/main/LICENSE|https://github.com/psf/requests|
# MAGIC |Spark NLP Display|Apache License 2.0|https://github.com/JohnSnowLabs/spark-nlp-display/blob/main/LICENSE|https://github.com/JohnSnowLabs/spark-nlp-display|
# MAGIC |Spark NLP |Apache License 2.0| https://github.com/JohnSnowLabs/spark-nlp/blob/master/LICENSE | https://github.com/JohnSnowLabs/spark-nlp|
# MAGIC |Spark NLP for Healthcare|[Proprietary license - John Snow Labs Inc.](https://www.johnsnowlabs.com/spark-nlp-health/) |NA|NA|
# MAGIC |Author|
# MAGIC |-|
# MAGIC |Databricks Inc.|
# MAGIC |John Snow Labs Inc.|
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Disclaimers
# MAGIC
# MAGIC Databricks Inc. (“Databricks”) does not dispense medical, diagnosis, or treatment advice. This Solution Accelerator (“tool”) is for informational purposes only and may not be used as a substitute for professional medical advice, treatment, or diagnosis. This tool may not be used within Databricks to process Protected Health Information (“PHI”) as defined in the Health Insurance Portability and Accountability Act of 1996, unless you have executed with Databricks a contract that allows for processing PHI, an accompanying Business Associate Agreement (BAA), and are running this notebook within a HIPAA Account.  Please note that if you run this notebook within Azure Databricks, your contract with Microsoft applies.
# MAGIC
# MAGIC The job configuration is written in the RUNME notebook in json format. The cost associated with running the accelerator is the user's responsibility.
# MAGIC
# MAGIC ## Instruction
# MAGIC
# MAGIC To run this accelerator, set up JSL Partner Connect [AWS](https://docs.databricks.com/integrations/ml/john-snow-labs.html#connect-to-john-snow-labs-using-partner-connect), [Azure](https://learn.microsoft.com/en-us/azure/databricks/integrations/ml/john-snow-labs#--connect-to-john-snow-labs-using-partner-connect) and navigate to **My Subscriptions** tab. Make sure you have a valid subscription for the workspace you clone this repo into, then **install on cluster** as shown in the screenshot below, with the default options. You will receive an email from JSL when the installation completes.
# MAGIC
# MAGIC <br>
# MAGIC <img src="https://raw.githubusercontent.com/databricks-industry-solutions/oncology/main/images/JSL_partner_connect_install.png" width=65%>
# MAGIC
# MAGIC Once the JSL installation completes successfully, clone this repo into a Databricks workspace. Attach the RUNME notebook to any cluster running a DBR 11.0 or later runtime, and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. Execute the multi-step-job to see how the pipeline runs.
# MAGIC
# MAGIC
