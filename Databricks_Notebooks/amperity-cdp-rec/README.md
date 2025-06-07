![image](https://user-images.githubusercontent.com/86326159/206014015-a70e3581-e15c-4a10-95ef-36fd5a560717.png)

[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

Data in the Amperity customer data platform (CDP) is useful for enabling a wide range of customer experiences. Customer identity resolution, data cleansing, and data unification provided by Amperity provides a very accurate, first-party customer data foundation which can be used to drive a wide range of customer insights and customer engagements. 

Using these data, we may wish to estimate customer lifetime value, derive behavioral segments and estimate product propensities, all capabilities we can tap into with the Amperity CDP. For some capabilities, such as the generation of per-user product recommendations, we need to develop specialized models leveraging capabilities such as those found in Databricks. 

In this notebook, we will demonstrate how to publish customer purchase history data from the Amperity CDP to Databricks to enable the training of a simple matrix factorization model. Recommendations produced by the model will then be published back to the CDP to enable any number of personalized interactions with our customers.  This notebook will borrow heavily from the previously published notebooks on matrix factorization available [here](https://www.databricks.com/blog/2023/01/06/products-we-think-you-might-generating-personalized-recommendations.html). Those interested into diving in the details of building such a model should review those notebooks and the accompanying blog.

___
<bryan.smith@databricks.com>

___

&copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.


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
