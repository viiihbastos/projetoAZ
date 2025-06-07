# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC This notebook contains a reference implementation of Tokenization, where our tokens and raw data are stored in Delta Lake tables, and we use Spark as the execution engine.
# MAGIC
# MAGIC This implementation is missing a key part of Tokenization - setup with Unity Catalog.
# MAGIC
# MAGIC Here are a rough set of steps:
# MAGIC 1. Create data - note that this can be data of any type - as long as you can get it into a Spark DataFrame, including non file-based sources.
# MAGIC 2. Create vaults - one per dataset. 
# MAGIC 3. Create tokens for PII.
# MAGIC 4. Anonymize the data by substituting raw data for tokens.
# MAGIC 5. Deanonymize the anyonymized data by substituting tokens for raw data.

# COMMAND ----------

# MAGIC %pip install faker

# COMMAND ----------

# DBTITLE 1,Demonstration setup and reset
BRONZE_DATA_PATH = "dbfs:/tmp/tokenization/bronze/data"
VAULT_BASE_PATH = "dbfs:/tmp/tokenization/vaults/"
SILVER_DATA_PATH = "dbfs:/tmp/tokenization/silver/data"

# Ignore this part - just for this demonstration - for reproducibility
paths_to_reset = [BRONZE_DATA_PATH, VAULT_BASE_PATH, SILVER_DATA_PATH]
for x in paths_to_reset:
  dbutils.fs.rm(x, recurse=True)

# COMMAND ----------

# DBTITLE 1,Create synthetic event data
"""

Synthetic data - we take the option of persisting this

"""

from pyspark.sql.types import StringType
import pyspark.sql.functions as F
from faker import Factory

user_count = 10_000
event_count = 50_000_000

def generate_ip():
    """
    potentially can optimize this by 
      - passing the factory object instead of per-partition instantiation (assuming it is serializable)
      - transitioning to a pandas UDF
    
    """
    faker = Factory.create()
    return faker.ipv4(network=False)
           

schema = StringType()

generate_ip_udf = udf(generate_ip, schema)

user_ip_df = (
  spark.range(user_count)
  .withColumn("ip_address", generate_ip_udf())
  .withColumnRenamed("id", "user_id")
)

event_df = (
  spark.range(event_count)
  .withColumnRenamed("id", "event_id")
  .withColumn("user_id", F.floor(F.rand() * user_count)) # generate a random user id
  .join(user_ip_df, ["user_id"])
)

event_df.write.save(BRONZE_DATA_PATH)

# COMMAND ----------

# DBTITLE 1,Key/field definitions
"""

This cell lets users specify the fields that are to be tokenized, as well as a key field that uniquely identifies a user.

Ideally, these would not need to be manually specified and the system would be able to automatically recognize fields that need to be tokenized and change this specification on the fly, bringing the following benefits: 
  * Users wouldn't have to toil with specifying these fields over a lot of possibly wide datasets
  * With the manual approach, we recommend limiting schema evolution of source datasets so as to avoid new fields leaking private information - if fields leak, you have to reprocess the data, which can be cumbersome
  
For now, we specify these manually.

"""

fields_to_be_tokenized = [
    "user_id",  # user_ids are PII per most privacy legislation
    "ip_address",
]

key_field = "user_id"
non_key_fields = list(set(fields_to_be_tokenized) - set([key_field]))

# COMMAND ----------

# DBTITLE 1,Create vaults
"""
This cell creates the vault tables backed with Delta Lake. 

We use Delta Lake identity columns to generate tokens: https://docs.databricks.com/sql/language-manual/sql-ref-syntax-ddl-create-table-using.html#parameters (scroll down to GENERATED ALWAYS...)

We make sure that those tokens can associate to raw, usually private data. We also make sure that tokens and users are associated to make any privacy compliance (i.e. deletion) easier.

The order of the creation doesn't matter, these could be run in parallel to the same effect. All that matters is that the vaults are completed before heading onto the next step.

"""

for field in non_key_fields:
    vault_path = VAULT_BASE_PATH + field
    print(f"Creating {field} vault at {vault_path}")
    spark.sql(
      f"""
      CREATE TABLE IF NOT EXISTS delta.`{vault_path}`
      (token LONG GENERATED ALWAYS AS IDENTITY (START WITH 1 INCREMENT BY 1), {key_field} LONG NOT NULL, {field} STRING NOT NULL)
      """
    )

vault_path = VAULT_BASE_PATH + key_field
print(f"Creating {key_field} vault at {vault_path}")
spark.sql(
  f"""
  CREATE TABLE IF NOT EXISTS delta.`{vault_path}`
  (token LONG GENERATED ALWAYS AS IDENTITY (START WITH 1 INCREMENT BY 1), {key_field} LONG NOT NULL)
  """
)

# COMMAND ----------

# DBTITLE 1,Tokenization
from delta import DeltaTable

"""

This cell takes raw data and tokenizes it, generating tokens for fields as specified in the "Key/field definitions" cell.

If, for a piece of data, an entry in the vault does not exist, the data is inserted into the table and a unique token is generated alongside it.
If the data already exists in the table, no further action occurs (no data is written out).

"""

# this configuration is for identity column generation. alternatively, a values map can be supplied to the whenNotMatchedInsertAll clause in the below merge statement
spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", True)

bronze = spark.read.load(BRONZE_DATA_PATH)

def distinct_fields(keys: list[str]) -> list[str]:
    """
    Takes a list of strings and returns the set of distinct string values.
    """
    return list(set(keys))


for field in fields_to_be_tokenized:
    vault_path = VAULT_BASE_PATH + field
    vault = DeltaTable.forPath(spark, vault_path)
    
    projection = distinct_fields([field, key_field])
    projected_df = bronze.select(projection).distinct() # TODO: investigate if PII shared across subjects should also share tokens. right now these are not separated.
    merge = (
        vault.alias("target")
        .merge(
            projected_df.alias("source"),
            f"source.{field} = target.{field}",
        )
        .whenNotMatchedInsertAll()
    )
    merge.execute()

# COMMAND ----------

# DBTITLE 1,Anonymization
"""

This cell takes raw data and tokens and generates effectively anonymized data.

It does this by joining and substituting, where raw data and vaults are joined together, tokens are substituted for raw data, and this anonymized data is written out.

In this implementation, we disambiguate the token and raw field by appending a "_token" suffix to all tokenized fields e.g. ip_address becomes ip_address_token. This is optional but may help with user understanding, particularly with fields that are already integer types.

"""

assert fields_to_be_tokenized, "No fields to tokenize, this process is potentially leaking data"

anon = bronze # base step

for field in fields_to_be_tokenized:
    vault_path = VAULT_BASE_PATH + field
    vault = DeltaTable.forPath(spark, vault_path).toDF()
    anon = (
        anon.join(vault, [field], "left")
        .withColumn(f"{field}_token", F.col("token"))
        .drop("token", field, key_field)
    ) 

anon.display()
anon.write.mode("append").save(SILVER_DATA_PATH)

# COMMAND ----------

# DBTITLE 1,Deanonymization
"""

This cell takes our effectively anonymized data and deanonymizes it.

It does this by doing the same as the anonymization step, but in reverse - it joins and backsubstitutes tokens for raw data.

"""

anon = spark.read.load(SILVER_DATA_PATH)

for field in non_key_fields:
    vault_path = VAULT_BASE_PATH + field
    vault = DeltaTable.forPath(spark, vault_path).toDF()
    anon = (
        anon.alias("source")
        .join(
            vault.alias("vault"),
            F.col(f"source.{field}_token") == F.col(f"vault.token"),
        )
        .drop("token", f"{field}_token", key_field)
    )

vault_path = VAULT_BASE_PATH + key_field
vault = DeltaTable.forPath(spark, vault_path).toDF()
deanon = (
    anon.alias("source")
    .join(
        vault.alias("vault"),
        F.col(f"source.{key_field}_token") == F.col(f"vault.token"),
    )
    .drop("token", f"{key_field}_token")
)

deanon.display()

# COMMAND ----------

# DBTITLE 1,Basic Tests
assert sorted(bronze.schema.fieldNames()) == sorted(
    deanon.schema.fieldNames()
), "Detected mismatch in field names"


assert (
    bronze.count() == bronze.join(deanon, ["event_id"]).count()
), "Detected mismatch in events and matches"


hash_fields = sorted(list(set(bronze.schema.fieldNames()) - set(["event_id"])))
assert (
    bronze.count()
    == (
        bronze.withColumn("_hash", F.hash(*hash_fields)).join(
            deanon.withColumn("_hash", F.hash(*hash_fields)), ["event_id", "_hash"]
        )
    ).count()
), "Detected mismatch in event content"
