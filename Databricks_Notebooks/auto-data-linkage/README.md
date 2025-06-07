# ARC has not been under development since early 2022. It is highly recommended you do not use this tool, and is left here purely as a resource for those wishing to explore automatic linking with Splink. It is recommended that you instead use Splink directly. 


# Databricks ARC

Welcome to the Databricks ARC Github page.

## Installation

The package can be installed with pip:

```bash
%pip install databricks-arc
```

## Databricks Runtime Requirements

ARC requires DBR 12.2 LTS ML

## Project Description

Databricks ARC (Automated Record Connector) is a solution accelerator by Databricks that performs highly scalable probabilistic data de-duplication 
and linking without the requirement for any labelled data or subject matter expertise in entity resolution.

De-duplication and linking are 2 sides of the same coin; de-duplication will find records *within* a dataset which represent the same entity, 
whilst linking will find records *across* 2 datasets which represent the same entity. De-deduplication is key requirement for implementing a Master Data Management strategy;
for example, to provide a Single Customer View by consolidating different data silos. Linking is also a key part, by bringing together different fragments of information
to build a holistic representation of an entity. 

To illustrate with an example, this table requires de-duplicating

|**First Name**|**Surname**|**Address Line 1**|**Address Line 2**|**Address Line 3**|**Post Code**|**DoB**|
|--------------|-----------|------------------|------------------|------------------|-------------|-------|
|Jo|Blogs|123 Fake Street|Real Town|Real County|

ARC's linking engine is the UK Ministry of Justice's open-sourced entity resolution package, [Splink](https://github.com/moj-analytical-services/splink). It builds on the technology of Splink by removing the need to manually provide parameters to calibrate an unsupervised de-duplication task, which require both a deep understanding of entity resolution and good knowledge of the dataset itself. The way in which ARC achieves this is detailed in the table below:

| **Parameter**           | **Splink**                                                                                                                                            | **ARC**                                                                                                                                                                                 |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Prior match probability | User to provide SQL-like statements for “deterministic rules” and a recall score for Splink to estimate the prior probability that two records match. | Automatically set prior probability to $$\frac{1}{N}$$.                                                                                                                                                  |
| Training rules          | User to provide SQL-like statements for a series of rules which trains the m probability values for each column.                                      | Automatically generate training rule statements such that each column is trained.                                                                                                       |
| Comparisons             | User to provide distance functions and thresholds for each column to compare.                                                                         | Automatically optimise multi-level parameter space for functions and thresholds.                                                                                                        |
| Blocking rules          | User to provide SQL-like statements to determine the possible comparison space and reduce the number of pairs to compare.                             | User to provide a parameter for the maximum number of pairs they’re willing to compare; Arc identifies all possible blocking rules within that boundary and optimises for the best one. |


### Parameter Optimisation

Arc uses Hyperopt (http://hyperopt.github.io/hyperopt/) to perform a Bayesian search to find the optimal settings, where optimality is defined as minimising the entropy of the data after clustering and standardising cluster record values. The intuition here is that as we are linking different representations of the same entity together (e.g. Facebook == Fakebook), then standardising data values within a cluster will reduce the total number of data values in the dataset.

To achieve this, Arc optimises for a custom information gain metric which it calculates based on the clusters of duplicates that Splink predicts. Intuitively, it is based on the reduction in entropy when the data is split into its clusters. The higher the reduction in entropy in the predicted clusters of duplicates predicted, the better the model is doing. Mathematically, we define the metric as follows:

Let the number of clusters in the matched subset of the data be *c*.

Let the maximum number of unique values in any column in the original dataset be *u*.

Then the "scaled" entropy of column *k*, *N* unique values with probability *P* is

$$E_{s,k} = -\Sigma_{i}^{N} P_{i} \log_{c}(P_{i})$$

Then the "adjusted" entropy of column *k*, *N* unique values with probability *P* is

$$E_{a,k} = -\Sigma_{i}^{N} P_{i} \log_{u}(P_{i})$$

The scaled information gain is

$$I_{s} = \Sigma_{k}^{K} E_{s,k} - E'_{s,k}$$

and the adjusted information gain is

$$I_{a} = \Sigma_{k}^{K} E_{a,k} - E'_{a,k}$$

where *E* is the mean entropy of the individual clusters predicted.

The metric to optimise for is:

$$I_{s}^{I_{a}}$$


## Getting Started

Load a Spark DataFrame of data to be deduplicated:

```python
data = spark.read.table("my_catalog.my_schema.my_duplicated_data")
```

After installation, import and enable ARC:

```python
import arc
from arc.autolinker import AutoLinker

arc.enable_arc()
```

Initialise an instance of the `AutoLinker` class:

```python
autolinker = AutoLinker()
```

Run unsupervised de-duplication:

```python
autolinker.auto_link(
  data=data,                                                         # Spark DataFrame of data to deduplicate
)
```

Access clustered DataFrame - predicted duplicates will share the same `cluster_id`:

```python
clusters = autolinker.best_clusters_at_threshold()
```

Use Splink's built-in visualisers and dashboards:

```python
autolinker.cluster_viewer()
```
For a more in-depth walkthrough please see the included notebooks
