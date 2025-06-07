# User Guide

Please be sure to use an <a href="https://docs.databricks.com/en/machine-learning/index.html#create-a-cluster-using-databricks-runtime-ml">ML Runtime Cluster</a> with a recent Databricks Runtime on Unity Catalog. The dataset is small enough to run on a typical Databricks cluster, but if the demo is running slowly you can change the dgconfig dictionary passed to `generate_iot()` to lower the data volume. Note that the accelerator defaults to using the Catalog "default" and a Unity Catalog Volume for checkpoints - you can alter this to "hive_metastore" and dbfs if you're not on Unity Catalog. Please be sure you've created and have access to the catalog that you pass to `get_config()`, or run `%sql create catalog default` to create a default catalog. It does not reset the tables underneath the schema `iot_distributed_pandas` by default, but you can set `create_schema=False` or `drop_schema=True` in `reset_tables()` or change the catalog and schema names as arguments in `get_config()` if you run into naming conflicts. Be sure to do this for each notebook.

# Issues

For any issue, please raise a ticket to the github project directly.
