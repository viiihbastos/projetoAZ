# Databricks notebook source
# MAGIC %md This notebook sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to illustrate the order of execution. Happy exploring! 
# MAGIC 🎉
# MAGIC 
# MAGIC **Steps**
# MAGIC 1. Simply attach this notebook to a cluster and hit Run-All for this notebook. A multi-step job and the clusters used in the job will be created for you and hyperlinks are printed on the last block of the notebook. 
# MAGIC 
# MAGIC 2. Run the accelerator notebooks: Feel free to explore the multi-step job page and **run the Workflow**, or **run the notebooks interactively** with the cluster to see how this solution accelerator executes. 
# MAGIC 
# MAGIC     2a. **Run the Workflow**: Navigate to the Workflow link and hit the `Run Now` 💥. 
# MAGIC   
# MAGIC     2b. **Run the notebooks interactively**: Attach the notebook with the cluster(s) created and execute as described in the `job_json['tasks']` below.
# MAGIC 
# MAGIC **Prerequisites** 
# MAGIC 1. You need to have cluster creation permissions in this workspace.
# MAGIC 
# MAGIC 2. In case the environment has cluster-policies that interfere with automated deployment, you may need to manually create the cluster in accordance with the workspace cluster policy. The `job_json` definition below still provides valuable information about the configuration these series of notebooks should run with. 
# MAGIC 
# MAGIC **Notes**
# MAGIC 1. The pipelines, workflows and clusters created in this script are not user-specific. Keep in mind that rerunning this script again after modification resets them for other users too.
# MAGIC 
# MAGIC 2. If the job execution fails, please confirm that you have set up other environment dependencies as specified in the accelerator notebooks. Accelerators may require the user to set up additional cloud infra or secrets to manage credentials. 

# COMMAND ----------

# DBTITLE 0,Install util packages
# MAGIC %pip install git+https://github.com/databricks-academy/dbacademy@v1.0.13 git+https://github.com/databricks-industry-solutions/notebook-solution-companion@safe-print-html --quiet --disable-pip-version-check

# COMMAND ----------

from solacc.companion import NotebookSolutionCompanion

# COMMAND ----------

job_json = {
        "timeout_seconds": 28800,
        "max_concurrent_runs": 1,
        "tags": {
            "usage": "solacc_testing",
            "group": "RCG",
            "accelerator": "call-center-analytics"
        },
        "tasks": [
            {
                "job_cluster_key": "driver_only_cluster",
                "notebook_task": {
                    "notebook_path": f"00_Intro"
                },
                "task_key": "call_center_00"
            },
            {
                "job_cluster_key": "call_center_cluster_w_init",
                "notebook_task": {
                    "notebook_path": f"01_Whisper"
                },
                "task_key": "call_center_01",
                "depends_on": [
                    {
                        "task_key": "call_center_00"
                    }
                ]
            }
        ],
        "job_clusters": [
          {
              "job_cluster_key": "driver_only_cluster",
              "new_cluster": {
                  "spark_version": "12.2.x-cpu-ml-scala2.12",
                  "num_workers": 0,
                  "node_type_id": {"AWS": "i3.4xlarge", "MSA": "Standard_E16_v3", "GCP": "n1-highmem-16"}
              }
          },
          {
              "job_cluster_key": "call_center_cluster_w_init",
              "new_cluster": {
                  "spark_version": "12.2.x-gpu-ml-scala2.12",
                  "num_workers": 2,
                  "node_type_id": {"AWS": "g4dn.xlarge", "MSA": "Standard_NC6s_v3", "GCP": "a2-highgpu-1g"}, 
                  "init_scripts": [
                      {
                          "dbfs": {
                              "destination": "dbfs:/tutorials/LibriSpeech/install_ffmpeg.sh"
                          }
                      }
                  ]
              }
          }
      ]
    }

# COMMAND ----------

dbutils.widgets.dropdown("run_job", "False", ["True", "False"])
run_job = dbutils.widgets.get("run_job") == "True"
nsc = NotebookSolutionCompanion()
nsc.deploy_compute(job_json, run_job=run_job)
# _ = nsc.deploy_dbsql("./dashboards/dashboard.dbdash", dbsql_config_table, spark)
