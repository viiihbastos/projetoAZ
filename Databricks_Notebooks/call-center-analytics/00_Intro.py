# Databricks notebook source
# MAGIC %md README here

# COMMAND ----------

# MAGIC %md ##Create init script

# COMMAND ----------

# make folder to house init script
dbutils.fs.mkdirs('dbfs:/tutorials/LibriSpeech/')

# write init script
dbutils.fs.put(
  '/tutorials/LibriSpeech/install_ffmpeg.sh',
  '''
#!/bin/bash
apt install -y ffmpeg
''', 
  True
  )

# show script content
print(
  dbutils.fs.head('dbfs:/tutorials/LibriSpeech/install_ffmpeg.sh')
  )
