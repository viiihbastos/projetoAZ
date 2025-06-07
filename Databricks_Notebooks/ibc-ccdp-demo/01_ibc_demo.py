# Databricks notebook source
# MAGIC %md
# MAGIC ### Demo Scenario
# MAGIC * We're a global SVOD service that is laser-focused on growing our fanbase to 100M subscribers by the end of 2024. To fuel this growth cost effectively, we've instituted a program to boost retention amongst our fans. As part of this program, we are consistently experimenting with new interventions for those with a high propensity to churn.
# MAGIC
# MAGIC * In this demo, we demonstrate how to use a Composable CDP - powered by AWS, Databricks, and Hightouch - to mitigate churn and grow our user base.
# MAGIC
# MAGIC <img style="margin: auto; display: block" width="1000px" src="https://raw.githubusercontent.com/databricks-industry-solutions/ibc-ccdp-demo/main/images/db_ccdp_ref_arch.png">

# COMMAND ----------

# MAGIC %run ./99_utils

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Load C360 Views
# MAGIC
# MAGIC In this step, we load two tables that are managed by our central data engineering team. 
# MAGIC   * `subscribers:` this table contains account-level information for all of our subscribers. 
# MAGIC   * `svod_usage:` this table contains engagement and quality of service metrics for all of our subscribers.
# MAGIC
# MAGIC We'll use these tables in the next step to craft features for predicting propensity to churn.

# COMMAND ----------

subscribers_df = spark.table('ibc.ccdp_demo.subscribers')
svod_df = spark.table('ibc.ccdp_demo.svod')

# COMMAND ----------

# DBTITLE 1,View subscribers table
# MAGIC %sql
# MAGIC select * from ibc.ccdp_demo.subscribers

# COMMAND ----------

# DBTITLE 1,View SVOD usage table
# MAGIC %sql
# MAGIC select * from ibc.ccdp_demo.svod

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Prepare data for machine learning

# COMMAND ----------

# DBTITLE 1,Create train and test data sets
# Join subscribers and svod usage tables
sub_usage_df = (subscribers_df.join(svod_df, on=subscribers_df.SUBSCRIBER_SK_ID == svod_df.Subscriber_ID, how='inner')
              .drop('Subscriber_ID')
              .withColumn('churn_flag', when(col('SUBSCRIPTION_STATE').like('%Cancelled%'), 1).otherwise(0))
              .withColumnRenamed('Video Start Failure_L30', 'Video_Start_Failure_L30')
              .withColumnRenamed('High Rebuffering_L30', 'High_Rebuffering_L30'))

# Construct train and test data sets
df = sub_usage_df.toPandas()
df = (df[["PSYCHOGRAPHIC_TRAIT","PRIMARY_SUBSCRIPTION_TYPE","PRIMARY_SUBSCRIPTION_PERIOD","AUTORENEW",
          "Engagement_L7","Engagement_L30","Profiles_With_Engagement",'churn_flag']])
df = (pd.get_dummies(df,columns=["PSYCHOGRAPHIC_TRAIT","PRIMARY_SUBSCRIPTION_TYPE","PRIMARY_SUBSCRIPTION_PERIOD","AUTORENEW"],
                     dtype='int64'))
features = [i for i in list(df.columns) if i != 'churn_flag']
X_train, X_test, y_train, y_test = train_test_split(df[features], df["churn_flag"], test_size=0.33, random_state=55)

# View sample of X_train
X_train

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Use Hyperopt to find optimal hyperparameter settings
# MAGIC
# MAGIC What is Hyperopt? 
# MAGIC * [Hyperopt](http://hyperopt.github.io/hyperopt/) is an open source tool that automates the process of model selection and hyperparameter tuning. 
# MAGIC * This popular Python library is pre-installed in [Databricks Runtime for Machine Learning](https://docs.databricks.com/en/runtime/mlruntime.html). 

# COMMAND ----------

# DBTITLE 1,Define model evaluation for hyperopt
def evaluate_model(params):
  #instantiate model
  model = XGBClassifier(use_label_encoder=False,learning_rate=params["learning_rate"],
                            gamma=int(params["gamma"]),
                            reg_alpha=int(params["reg_alpha"]),
                            reg_lambda=int(params["reg_lambda"]),
                            max_depth=int(params["max_depth"]),
                            n_estimators=int(params["n_estimators"]),
                            min_child_weight = params["min_child_weight"], objective='reg:linear', early_stopping_rounds=50)
  
  #train
  model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
  
  #predict
  y_prob = model.predict_proba(X_test)
  y_pred = model.predict(X_test)
  
  #score
  precision = average_precision_score(y_test, y_prob[:,1])
  f1 = f1_score(y_test, y_pred)
    
  mlflow.log_metric('avg_precision', precision)  # record metric with mlflow run
  mlflow.log_metric('avg_f1', f1)  # record metric with mlflow run
  
  # return results (negative precision as we minimize the function)
  return {'loss': -f1, 'status': STATUS_OK, 'model': model}

# COMMAND ----------

# DBTITLE 1,Define search space for hyperopt
# define hyperopt search space
search_space = {'max_depth': scope.int(hp.quniform('max_depth', 2, 8, 1)),
                'learning_rate': hp.loguniform('learning_rate', -3, 0),
                'gamma': hp.uniform('gamma', 0, 5),
                'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
                'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
                'min_child_weight': scope.int(hp.loguniform('min_child_weight', -1, 3)),
                'n_estimators':  scope.int(hp.quniform('n_estimators', 50, 200, 1))}

# COMMAND ----------

# DBTITLE 1,Perform hyperparameter tuning using hyperopt
with mlflow.start_run(run_name='XGBClassifier') as run:
  trials = SparkTrials(parallelism=4)
  
  argmin = fmin(fn=evaluate_model, 
                space=search_space, 
                algo=tpe.suggest, 
                max_evals=20,
                trials=trials)

# COMMAND ----------

# DBTITLE 1,View optimal hyperparameter settings
space_eval(search_space,argmin)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4: Train XGBoost Classifier model using optimal hyperparameter settings

# COMMAND ----------

# perform evaluation
with mlflow.start_run(run_name='XGBClassifier_Churn') as run:
  
  # capture run info for later use
  run_id = run.info.run_id
  
  #configure params
  params = space_eval(search_space, argmin)
 
  # train
  model = XGBClassifier(**params)
  model.fit(X_train, y_train)
  mlflow.sklearn.log_model(model, 'model')  # persist model with mlflow
  
  # predict
  y_prob = model.predict_proba(X_test)
  y_pred = model.predict(X_test)
  
  #score
  precision = average_precision_score(y_test, y_prob[:,1])
  f1 = f1_score(y_test, y_pred)
  
  # score
  model_ap = average_precision_score(y_test, y_prob[:,1])
  mlflow.log_metric('f1', f1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5: Score each subscriber's propensity to churn

# COMMAND ----------

# DBTITLE 1,Prepare data for inference
df = sub_usage_df.toPandas()
df = (df[["SUBSCRIBER_SK_ID", "PSYCHOGRAPHIC_TRAIT","PRIMARY_SUBSCRIPTION_TYPE","PRIMARY_SUBSCRIPTION_PERIOD",
          "AUTORENEW","Engagement_L7","Engagement_L30","Profiles_With_Engagement"]])
df = (pd.get_dummies(df,columns=["PSYCHOGRAPHIC_TRAIT","PRIMARY_SUBSCRIPTION_TYPE","PRIMARY_SUBSCRIPTION_PERIOD","AUTORENEW"],
                     dtype='int64'))

features = [feature_column for feature_column in df.columns if feature_column != 'SUBSCRIBER_SK_ID']

# COMMAND ----------

# DBTITLE 1,Use model to perform inference
df["propensity_to_churn"] = model.predict_proba(df[features])[:,1]

# COMMAND ----------

# DBTITLE 1,View predictions
from pyspark.sql.functions import round
churn_df = spark.createDataFrame(df).select('SUBSCRIBER_SK_ID',round(col('propensity_to_churn'),2).alias('propensity_to_churn'))
display(churn_df)

# COMMAND ----------

# DBTITLE 1,Save Predictions
churn_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("ibc.ccdp_demo.churn") 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6: Activate audience segments in Hightouch
# MAGIC [Link to Hightouch](https://app.hightouch.com/)
