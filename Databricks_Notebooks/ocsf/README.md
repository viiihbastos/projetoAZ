![image](https://github.com/lipyeowlim/public/raw/main/img/logo/databricks_cyber_logo_v1.png)

[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

# Schema Mappings for Open Cybersecurity Schema Framework

Contact author: <cybersecurity@databricks.com>
Co-authors: Databricks cybersecurity SME members

This solution accelerator provides sample code for mapping various types of security logs to the Open Cybersecurity Schema Framework (OCSF) via structured streaming (schema-on-write), Delta Live Tables (schema-on-write), and/or views (schema-on-read). Note that the schema-on-read and schema-on-write approaches are not mutually exclusive: some tables can use schema-on-write and some tables can use schema-on-read.

# Use Cases

Personas: Security Engineers, Data Engineers

* Schema-on-write: How do you transform security logs (in bronze tables) in their source schemas to the OCSF schema and write the results to silver tables?
* Schema-on-read: How do you query security logs stored in their source schemas using queries written against OCSF?

## Reference Architecture

<img src="https://github.com/lipyeowlim/public/raw/main/img/ocsf/ocsf_ref_arch.png" width="600px">

We recommend the use of the [medallion
architecture](https://www.databricks.com/glossary/medallion-architecturer)
where the raw security logs are ingested as bronze tables in their
source schemas. The mappings in this solution accelerator are then
applied to obtain the silver tables (or views) in OCSF. Further
aggregation or detection processing can be applied to obtain gold
tables. The above architecture shows the sample pipelines or data flows
for the [zeek](https://zeek.org/) data source, in particular, the http,
dns, ftp logs.

## Technical Overview

Coming soon.

___

&copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
|                                  |      |      |           |

## Getting started

Although specific solutions can be downloaded as .dbc archives from our websites, we recommend cloning these repositories onto your databricks environment. Not only will you get access to latest code, but you will be part of a community of experts driving industry best practices and re-usable solutions, influencing our respective industries. 

<img width="500" alt="add_repo" src="https://user-images.githubusercontent.com/4445837/177207338-65135b10-8ccc-4d17-be21-09416c861a76.png">

To start using a solution accelerator in Databricks simply follow these steps: 

1. Clone solution accelerator repository in Databricks using [Databricks Repos](https://www.databricks.com/product/repos)
2. Attach the `RUNME` notebook to any cluster and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. The job configuration is written in the RUNME notebook in json format. 
3. Execute the multi-step-job to see how the pipeline runs. 
4. You might want to modify the samples in the solution accelerator to your need, collaborate with other users and run the code samples against your own data. To do so start by changing the Git remote of your repository  to your organization’s repository vs using our samples repository (learn more). You can now commit and push code, collaborate with other user’s via Git and follow your organization’s processes for code development.

The cost associated with running the accelerator is the user's responsibility.


## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support. 
