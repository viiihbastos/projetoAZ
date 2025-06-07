# Databricks notebook source
# MAGIC %md
# MAGIC https://github.com/hmi-utwente/video-game-text-corpora/tree/master/Star%20Wars%3A%20Knights%20of%20the%20Old%20Republic

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(id) FROM `duncan`.`workspace`.`NPCDialog`;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM `duncan`.`workspace`.`NPCDialog`;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   id,
# MAGIC   text,
# MAGIC   ai_query(
# MAGIC     "databricks-meta-llama-3-70b-instruct",
# MAGIC     CONCAT("You are an assistant to review the following dialogue for profanity or words that may affect a game rating. Respond with a yes if the text contains it or no: ", text)
# MAGIC   ) AS check
# MAGIC FROM `duncan`.`workspace`.`NPCDialog`
# MAGIC --where id = 77
# MAGIC LIMIT 100

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH npc_dialog AS (
# MAGIC   SELECT 
# MAGIC     id,
# MAGIC     text,
# MAGIC     ai_query(
# MAGIC       "databricks-meta-llama-3-70b-instruct",
# MAGIC       CONCAT("You are an assistant to review the following dialogue for profanity or words that may affect a game rating. Respond with a yes if the text contains it or no: ", text)
# MAGIC     ) AS check
# MAGIC   FROM `duncan`.`workspace`.`NPCDialog`
# MAGIC )
# MAGIC SELECT 
# MAGIC   id,
# MAGIC   text,
# MAGIC   check
# MAGIC FROM npc_dialog
# MAGIC WHERE check LIKE '%Yes%' and id = 77

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC UPDATE `duncan`.`workspace`.`NPCDialog`
# MAGIC SET text = CONCAT(text, " , bastard")
# MAGIC WHERE id = 77;
