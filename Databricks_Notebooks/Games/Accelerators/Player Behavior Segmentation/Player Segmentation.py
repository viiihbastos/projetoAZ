# Databricks notebook source
# MAGIC %md
# MAGIC # Spark ML Clustering Pipeline for Player Data Analysis
# MAGIC
# MAGIC This documentation outlines the process of loading, processing, and clustering player data using Spark ML's machine learning capabilities. The goal is to segment the playerbase into distinct clusters based on their in-game behavior and purchase patterns.
# MAGIC
# MAGIC ## Pipeline Overview
# MAGIC
# MAGIC 1. **Data Loading**: The player data is loaded from a CSV file, with schema inference enabled to detect data types automatically.
# MAGIC
# MAGIC 2. **Feature Selection**: Non-feature columns such as 'Xuid', 'Year', and 'Month' are removed from the dataset, retaining only the columns relevant for clustering. Leaving only behavior related data points.
# MAGIC
# MAGIC 3. **Feature Engineering**:
# MAGIC     - **Vector Assembly**: The selected features are assembled into a single vector.
# MAGIC     - **Standardization**: The feature vectors are standardized to have a mean of 0 and a standard deviation of 1.
# MAGIC
# MAGIC 4. **Clustering**: The KMeans algorithm is applied to the processed features to segment the data into clusters.
# MAGIC
# MAGIC 5. **Pipeline Execution**: A pipeline is constructed and fitted to the data, encompassing all the above steps in sequence.
# MAGIC
# MAGIC 6. **Transformation**: The fitted pipeline is used to transform the data, adding a prediction column that indicates the cluster assignment for each player.
# MAGIC
# MAGIC 7. **Result Display**: The results, including the original data and the cluster assignments, are displayed.
# MAGIC
# MAGIC ## Usage
# MAGIC
# MAGIC The pipeline is set up to work with a dataset located with this notebook. This path should be adjusted to point to the actual location of the dataset in your environment.
# MAGIC
# MAGIC To execute the pipeline, simply run the provided code in a PySpark environment. The output will include the original data along with a new column named 'prediction' which indicates the cluster each player belongs to.
# MAGIC
# MAGIC ## Important Notes
# MAGIC
# MAGIC - The number of clusters (`k` in KMeans) is currently set to 3. This parameter should be optimized based on the dataset and the desired granularity of clustering.
# MAGIC
# MAGIC ## Output
# MAGIC
# MAGIC The final output will allow the identification of distinct player segments, which can be used for targeted marketing
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Galactic Frontiers Data
# MAGIC
# MAGIC **Galactic Frontiers** is an epic space exploration and conquest game where players navigate a vast universe filled with mysteries, battles, and alliances. Players earn and spend Premium Credits to upgrade their ships, recruit crews, and unlock special abilities through the Conquest Pass. Engage in Battles, make Galactic Discoveries, and complete challenging quests to rise through the ranks.
# MAGIC
# MAGIC ## Features
# MAGIC - **Free Premium Credits Gained**: Earn credits through various in-game activities.
# MAGIC - **Conquest Pass Premium Credits**: Special credits earned by completing conquest missions.
# MAGIC - **Premium Credits Spent on Ship**: Upgrade and customize your spaceship.
# MAGIC - **Premium Credits Spent on Crew**: Recruit and train elite crew members.
# MAGIC - **Total Game Hours**: Track your journey through the galaxy.
# MAGIC - **Captaincy Hours**: Measure your command experience.
# MAGIC - **AI Battles**: Engage in strategic battles against advanced AI opponents.
# MAGIC - **Galactic Discoveries**: Uncover new planets, resources, and secrets.
# MAGIC - **Credits Spent on Base**: Build and enhance your base of operations.
# MAGIC - **Credits Spent on Cosmetics**: Personalize your character and ship with unique cosmetic items.
# MAGIC - **Trade Transactions**: Engage in trade with other players and NPCs.
# MAGIC - **Achievements Earned**: Unlock achievements and showcase your accomplishments.
# MAGIC - **Aggressive Battles**: Participate in intense PvP encounters.

# COMMAND ----------

# MAGIC %pip install openai

# COMMAND ----------

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import DataFrame
import matplotlib.pyplot as plt
from IPython.display import Markdown
import openai
import os
import pandas as pd
import shap
import mlflow
from pyspark.sql.functions import (
    col, skewness, kurtosis, stddev, count, sum, mean, min, max, expr, lit, variance
)
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType

# COMMAND ----------

# MAGIC %md
# MAGIC Replace the api_key with your access token. Generate the token using your profile in the top right.
# MAGIC https://docs.databricks.com/en/dev-tools/auth/pat.html
# MAGIC
# MAGIC Replace the base_url with your workspace url keeping the /serving-endpoints after

# COMMAND ----------

client = openai.OpenAI(
  api_key="",
  base_url="https://insert-workspaceurl-here/serving-endpoints"
)

# COMMAND ----------

def create_and_fit_pipeline(data: DataFrame, num_clusters: int) -> PipelineModel:
    """
    Creates and fits a KMeans clustering pipeline.
    
    Parameters:
    - data: The input DataFrame containing the features for clustering.
    - num_clusters: The number of clusters to use for KMeans.
    
    Returns:
    - A trained PipelineModel object.
    """
    # Assemble features into a single vector
    vec_assembler = VectorAssembler(inputCols=data.columns, outputCol='features')
    
    # Scale features to have zero mean and unit variance
    scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures', withStd=True, withMean=True)

    # Initialize KMeans with the specified number of clusters
    kmeans = KMeans(featuresCol='scaledFeatures', k=num_clusters)

    # Create a pipeline with the stages: vector assembler, scaler, and KMeans
    pipeline = Pipeline(stages=[vec_assembler, scaler, kmeans])

    # Fit the pipeline to the data
    model = pipeline.fit(data)
    return model

def run_kmeans_silhouette(data: DataFrame, max_cluster_size: int) -> dict:
    """
    Runs KMeans clustering for a range of cluster sizes and evaluates each model using silhouette scores.
    
    Parameters:
    - data: The input DataFrame to cluster.
    - max_cluster_size: The maximum number of clusters to evaluate.
    
    Returns:
    - A dictionary with cluster sizes as keys and their corresponding silhouette scores as values.
    """
    evaluator = ClusteringEvaluator(metricName='silhouette')
    silhouette_scores = {}
    
    # Iterate over a range of cluster sizes
    for k in range(2, max_cluster_size + 1):
        # Create and fit the KMeans pipeline for the current number of clusters
        model = create_and_fit_pipeline(data, k)
        # Make predictions (cluster assignments)
        predictions = model.transform(data)
        # Evaluate the clustering using silhouette score
        silhouette = evaluator.evaluate(predictions)
        silhouette_scores[k] = silhouette
        
    return silhouette_scores

def calculate_correlation_matrix(data: DataFrame) -> pd.DataFrame:
    """
    Calculates the correlation matrix for the input DataFrame.
    
    Parameters:
    - data: The input DataFrame.
    
    Returns:
    - A pandas DataFrame representing the correlation matrix.
    """
    pandas_df = data.toPandas()
    correlation_matrix = pandas_df.corr()
    return correlation_matrix

def generate_shap_explanations(model: PipelineModel, data: DataFrame):
    """
    Generates SHAP explanations for the features used in the KMeans clustering.
    
    Parameters:
    - model: The trained KMeans model pipeline.
    - data: The input DataFrame used for clustering.
    """
    # Extract the stages of the pipeline
    vec_assembler = model.stages[0]
    scaler = model.stages[1]
    kmeans_model = model.stages[2]
    
    # Transform the data using the vector assembler and scaler
    feature_data = vec_assembler.transform(data)
    scaled_data = scaler.transform(feature_data)
    
    # Convert to Pandas DataFrame for SHAP
    pandas_df = scaled_data.toPandas()
    feature_names = vec_assembler.getInputCols()
    
    # Use SHAP KernelExplainer for explaining KMeans (since KMeans doesn't support direct SHAP)
    explainer = shap.KernelExplainer(kmeans_model.predict, pandas_df[feature_names])
    shap_values = explainer.shap_values(pandas_df[feature_names])
    
    # Plot the SHAP summary plot
    shap.summary_plot(shap_values, pandas_df[feature_names])

# COMMAND ----------

# Get the current working directory
current_dir = os.getcwd()

# Construct the path to your CSV file
file_path = os.path.join(current_dir, "galactic_frontiers_features.csv")

# Read in the table
full_data = spark.read.csv(f"file:{file_path}", header=True, inferSchema=True)

# Preprocess the data if necessary (e.g., dropping columns, filling missing values)
non_feature_columns = ['Xuid', 'Year', 'Month', ]

full_data = full_data.drop(*non_feature_columns).fillna(0)

display(full_data)

# COMMAND ----------

# Ensure seaborn, pandas, and matplotlib.pyplot are imported
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json

# List of feature columns
feature_columns = [col for col in full_data.columns]

# Create an empty dictionary to store correlation values
correlations = {}

# Calculate correlations between all pairs of features
for col1 in feature_columns:
    correlations[col1] = {}
    for col2 in feature_columns:
        correlations[col1][col2] = full_data.stat.corr(col1, col2)

# Convert the correlations dictionary to a pandas DataFrame for visualization
correlation_matrix = pd.DataFrame(correlations)

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# Convert the correlations dictionary to JSON
correlations_json = json.dumps(correlations)

chat_completion = client.chat.completions.create(
  messages=[
  {
    "role": "system",
    "content": "You are an AI assistant"
  },
  {
    "role": "user",
    "content": f"Explain what features may be correlated both high or low: {correlations_json}"
  }
  ],
  model="databricks-meta-llama-3-1-405b-instruct",
)

display(Markdown(chat_completion.choices[0].message.content))

# COMMAND ----------

# MAGIC %md
# MAGIC # Optimal Cluster Number Determination using the Elbow Method
# MAGIC
# MAGIC This documentation explains the process of determining the optimal number of clusters for KMeans clustering using the Elbow Method in a PySpark environment. The code performs an iterative process to find a balance between the number of clusters and the within-set sum of squared errors (WSSSE).
# MAGIC
# MAGIC ## Rationale
# MAGIC
# MAGIC The purpose of this code is to enhance the earlier clustering approach by adding a method to determine the most appropriate number of clusters (`k`) for KMeans. Instead of arbitrarily choosing a `k` value, the Elbow Method systematically evaluates the performance of the clustering algorithm across a range of `k` values.
# MAGIC
# MAGIC ## Process Overview
# MAGIC
# MAGIC 1. **Feature Preparation**: Just as with the initial clustering pipeline, features are assembled into a vector and standardized.
# MAGIC
# MAGIC 2. **Elbow Method Execution**:
# MAGIC     - Multiple KMeans models are trained over a range of `k` values.
# MAGIC     - For each `k`, the WSSSE is calculated to measure cluster cohesion.
# MAGIC
# MAGIC 3. **Evaluation and Visualization**:
# MAGIC     - The WSSSE for each `k` is stored and then plotted against the `k` values.
# MAGIC     - The plot typically shows a downward trend, where the decline flattens out after a certain `k`—the "elbow" point, indicating the optimal number of clusters.
# MAGIC
# MAGIC 4. **Interpretation**: 
# MAGIC     - A sharp change in the slope of the WSSSE curve indicates the elbow, beyond which increasing `k` brings diminishing returns in terms of error reduction.
# MAGIC     - This point suggests a balance between the complexity of the model (number of clusters) and the fit to the data (WSSSE).
# MAGIC
# MAGIC ## Purpose of this Code
# MAGIC
# MAGIC The code block is designed to replace guesswork with a data-driven approach for cluster number selection. By automating the evaluation across a range of `k` values, we aim to achieve the following:
# MAGIC   
# MAGIC - **Accuracy**: Ensure that the clusters created are meaningful and accurately represent the underlying patterns in the data.
# MAGIC - **Simplicity**: Avoid overcomplicating the model with too many clusters, which may lead to overfitting.
# MAGIC - **Performance**: Prevent under-clustering, where too few clusters might not capture important distinctions between different data points.
# MAGIC
# MAGIC ## Usage and Output
# MAGIC
# MAGIC Run the code in a PySpark environment to produce a plot of WSSSE values. The 'elbow' in the plot will indicate the optimal `k`. This value should then be used for the final KMeans clustering to ensure that the model is neither overfitting nor oversimplifying the data.
# MAGIC
# MAGIC This approach to determining the optimal number of clusters is crucial for downstream tasks such as customer segmentation, anomaly detection, and other data-driven strategies that rely on accurately grouped data.
# MAGIC
# MAGIC ## Additional Notes
# MAGIC
# MAGIC - The range of `k` values is currently set from 2 to 5. This may be adjusted based on the dataset size and diversity.
# MAGIC - It's important to consider the computational cost as increasing the range of `k` values can significantly increase processing time.
# MAGIC - Interpretation of the elbow plot is somewhat subjective and should be supplemented with domain knowledge and additional cluster validation metrics if necessary.
# MAGIC

# COMMAND ----------

# Disable MLflow autologging
mlflow.autolog(disable=True)

# Run KMeans silhouette analysis
data = run_kmeans_silhouette(full_data, max_cluster_size=5)

# Convert dictionary to DataFrame for display
import pandas as pd

df = pd.DataFrame(list(data.items()), columns=['Number of Clusters', 'Silhouette Score'])
display(df)

# Plotting the silhouette analysis
plt.figure(figsize=(10, 6))
plt.plot(df['Number of Clusters'], df['Silhouette Score'], marker='o')
plt.title('Silhouette Analysis for KMeans Clustering')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

chat_completion = client.chat.completions.create(
  messages=[
  {
    "role": "system",
    "content": "You are an AI assistant"
  },
  {
    "role": "user",
    "content": f"This data is around clusters coming from kmeans, tell me which number of clusters is best and why. Return response in markdown: {data}"
  }
  ],
  model="databricks-meta-llama-3-1-405b-instruct",
)

display(Markdown(chat_completion.choices[0].message.content))

# COMMAND ----------

# Enable MLflow autologging
mlflow.autolog()

# Initialize an empty DataFrame to store all predictions
#all_predictions = spark.createDataFrame([], full_data.schema.add("prediction", IntegerType()))

# Create and fit the pipeline
model = create_and_fit_pipeline(full_data, 3)  # Assuming 3 is the optimal number of clusters for the entire dataset

# Generate predictions
predictions = model.transform(full_data)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

predictions = predictions.drop("scaledFeatures", "features").withColumnRenamed("prediction","Cluster")
display(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC # Cluster Characterization Post KMeans Clustering
# MAGIC
# MAGIC After performing KMeans clustering, it is essential to understand the characteristics of each cluster. The following code helps in summarizing the clusters formed by a KMeans model in a Spark DataFrame. It provides insights into the size of each cluster and the average values of features within each cluster.

# COMMAND ----------

# Define the schema for the aggregated DataFrame
agg_schema = StructType([
    StructField("Cluster", StringType(), nullable=False),
    StructField("mean_value", DoubleType()),
    StructField("median_value", DoubleType()),
    StructField("std_value", DoubleType()),
    StructField("min_value", DoubleType()),
    StructField("max_value", DoubleType()),
    StructField("count_value", LongType()),
    StructField("sum_value", DoubleType()),
    StructField("variance_value", DoubleType()),
    StructField("25th_percentile_value", DoubleType()),
    StructField("75th_percentile_value", DoubleType()),
    StructField("skewness_value", DoubleType()),
    StructField("kurtosis_value", DoubleType()),
    StructField("range_value", DoubleType()),
    StructField("coefficient_of_variation_value", DoubleType()),
    StructField("iqr_value", DoubleType()),
    StructField("Feature", StringType(), nullable=False)
])

# Function to aggregate statistics for a single column
def aggregate_column_stats(data, col_name):
    agg_functions = {
        'mean': mean(col(col_name)).alias("mean_value"),
        'median': expr(f'percentile_approx(`{col_name}`, 0.5)').alias("median_value"),
        'std': stddev(col(col_name)).alias("std_value"),
        'min': min(col(col_name)).alias("min_value"),
        'max': max(col(col_name)).alias("max_value"),
        'count': count(col(col_name)).alias("count_value"),
        'sum': sum(col(col_name)).alias("sum_value"),
        'variance': variance(col(col_name)).alias("variance_value"),
        '25th_percentile': expr(f'percentile_approx(`{col_name}`, 0.25)').alias("25th_percentile_value"),
        '75th_percentile': expr(f'percentile_approx(`{col_name}`, 0.75)').alias("75th_percentile_value"),
        'skewness': skewness(col(col_name)).alias("skewness_value"),
        'kurtosis': kurtosis(col(col_name)).alias("kurtosis_value"),
        'range': (max(col(col_name)) - min(col(col_name))).alias("range_value"),
        'coefficient_of_variation': (stddev(col(col_name)) / mean(col(col_name))).alias("coefficient_of_variation_value"),
        'iqr': (expr(f'percentile_approx(`{col_name}`, 0.75) - percentile_approx(`{col_name}`, 0.25)')).alias("iqr_value")
    }
    agg_exprs = list(agg_functions.values())
    return data.groupBy('Cluster').agg(*agg_exprs)

# Initialize an empty DataFrame for union with the specified schema
Cluster_Metrics = spark.createDataFrame([], agg_schema)

# Iterate over the columns and union the aggregation results
for col_name in predictions.columns:
    if col_name != 'Cluster':
        agg_results = aggregate_column_stats(predictions, col_name)
        agg_results = agg_results.withColumn("Feature", lit(col_name))
        Cluster_Metrics = Cluster_Metrics.union(agg_results)

# Rearrange the columns in the desired order
desired_order = ["Feature", "Cluster"] + [c for c in Cluster_Metrics.columns if c not in ["Feature", "Cluster"]]
Cluster_Metrics = Cluster_Metrics.select(desired_order)

Cluster_Metrics.write.format("delta").mode("overwrite").saveAsTable("games_solutions.galactic_frontiers")

display(Cluster_Metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC This next two cell details how to visualize the feature distributions for clusters identified by KMeans using boxplots & scatterplots, why it is done, and how to interpret the results. It guides the user through the process of converting Spark DataFrame data to a format suitable for visualization and underscores the insights that can be gained from such visual analysis.
# MAGIC

# COMMAND ----------

data_json_rdd = Cluster_Metrics.toJSON()  # Convert DataFrame to an RDD of JSON strings
data_json_list = data_json_rdd.collect()  # Collect JSON strings into a Python list
data_json = "[" + ",".join(data_json_list) + "]"  # Join the list into a single JSON array string

chat_completion = client.chat.completions.create(
  messages=[
  {
    "role": "system",
    "content": "You are an AI assistant"
  },
  {
    "role": "user",
    "content": f"You are to help understand our player audience, with the goal to find the core playerbase. Provide a simple summary on the telemetry such as progression, interactions, engagement by cluster and a playerbase summary by cluster here is the data: {data_json}"
  }
  ],
  model="databricks-meta-llama-3-1-405b-instruct",
)

#print(chat_completion.choices[0].message.content)
display(Markdown(chat_completion.choices[0].message.content))
