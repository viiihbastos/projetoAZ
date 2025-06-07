![image](https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo_wide.png)

[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

# Business Problem 

Working with various x12 EDI transactions in Spark on Databricks.

# Install

```python
pip install git+https://github.com/databricks-industry-solutions/x12-edi-parser
```

# Run 

### 837i and 837p sample data in Spark

```python
from databricksx12 import *
from databricksx12.hls import *
import json
from pyspark.sql.functions import input_file_name

hm = HealthcareManager()
df = spark.read.text("sampledata/837/*txt", wholetext = True)


rdd = (
 df.withColumn("filename", input_file_name()).rdd
  .map(lambda x: (x.asDict().get("filename"),x.asDict().get("value")))
  .map(lambda x: (x[0], EDI(x[1])))
  .map(lambda x: { **{'filename': x[0]}, **hm.to_json(x[1])} )
  .map(lambda x: json.dumps(x))
)
claims = spark.read.json(rdd)

#Create Claims tables from the EDI transactions
claims.createOrReplaceTempView("edi")
```
### Creating claim header and line tables from EDI 

``` SQL
--flatten EDI 
drop table if exists stg_claims;
CREATE TABLE stg_claims 
as 
select *
from 
(
select *, explode(trnx.Claims) as clms
from
(
select filename, edi_control_number, edi_date, edi_time, 
edi_recipient_qualifier_id, edi_sender_qualifier_id, edi_standard_version,
fgs.fg.`FunctionalGroup.control_number` as fg_control_number,
fgs.fg.`FunctionalGroup.date` as fg_date,
fgs.fg.`FunctionalGroup.time` as fg_time,
fgs.fg.`FunctionalGroup.receiver` as fg_receiver,
fgs.fg.`FunctionalGroup.sender` as fg_sender,
fgs.fg.`FunctionalGroup.standard_version` as fg_standard_version,
fgs.fg.`FunctionalGroup.transaction_type` as fg_transaction_type,
  explode(fgs.fg.`Transactions`) as trnx
from 
(
select
`edi.control_number` as edi_control_number,
`edi.date` as edi_date,
`edi.time` as edi_time,
`EDI.recipient_qualifier_id` as edi_recipient_qualifier_id,
`EDI.sender_qualifier_id` as edi_sender_qualifier_id,
`EDI.standard_version` as edi_standard_version,
explode(`FunctionalGroup`) as fg,
filename
from edi
) fgs
) trnx
) clms;

drop table if exists claim_header;
create table claim_header as 
select
filename, edi_control_number, edi_date, edi_time, edi_recipient_qualifier_id, edi_sender_qualifier_id, edi_standard_version,
fg_control_number, fg_date, fg_time, fg_receiver, fg_sender, fg_standard_version, fg_transaction_type,
clms.claim_header.*, 
clms.diagnosis.*,
clms.payer.*,
clms.providers.*,
  clms.patient.name as patient_name,  
  clms.patient.patient_relationship_cd,
  clms.patient.street as patient_street,
  clms.patient.city as patient_city,
  clms.patient.zip as patient_zip,
  clms.patient.dob as patient_dob,
  clms.patient.dob_format as patient_dob_format,
  clms.patient.gender_cd as patient_gender_cd,
  clms.subscriber.subsciber_identifier,
  clms.subscriber.name as subscriber_name,
  clms.subscriber.subscriber_relationship_cd,
  clms.subscriber.street as subscriber_street,
  clms.subscriber.city as subscriber_city,
  clms.subscriber.zip as subscriber_zip,
  clms.subscriber.dob as subscriber_dob,
  clms.subscriber.dob_format as subscriber_dob_format,
  clms.subscriber.gender_cd as subscriber_gender_cd
from stg_claims;

create table claim_line as 
select filename, claim_id, cl.*
from (
select filename, 
clms.claim_header.claim_id, 
explode(clms.claim_lines) as cl 
from stg_claims
) foo;

```

### Sample Output and Data Dictionary

![image](images/claim_header.png?raw=true)
![image](images/claim_line.png?raw=true)

### 835 sample

```python
df = spark.read.text("sampledata/835/*txt", wholetext = True)

rdd = (
 df.withColumn("filename", input_file_name()).rdd
  .map(lambda x: (x.asDict().get("filename"),x.asDict().get("value")))
  .map(lambda x: (x[0], EDI(x[1])))
  .map(lambda x: { **{'filename': x[0]}, **hm.to_json(x[1])} )
  .map(lambda x: json.dumps(x))
)
claims = spark.read.json(rdd)

#Create Claims tables from the EDI transactions
%sql
--flatten EDI 
drop table if exists stg_remittance;
CREATE TABLE stg_remittance 
as 
select *
from 
(
select *, explode(trnx.Claims) as clms
from
(
select filename, edi_control_number, edi_date, edi_time, 
edi_recipient_qualifier_id, edi_sender_qualifier_id, edi_standard_version,
fgs.fg.`FunctionalGroup.control_number` as fg_control_number,
fgs.fg.`FunctionalGroup.date` as fg_date,
fgs.fg.`FunctionalGroup.time` as fg_time,
fgs.fg.`FunctionalGroup.receiver` as fg_receiver,
fgs.fg.`FunctionalGroup.sender` as fg_sender,
fgs.fg.`FunctionalGroup.standard_version` as fg_standard_version,
fgs.fg.`FunctionalGroup.transaction_type` as fg_transaction_type,
  explode(fgs.fg.`Transactions`) as trnx
from 
(
select
`edi.control_number` as edi_control_number,
`edi.date` as edi_date,
`edi.time` as edi_time,
`EDI.recipient_qualifier_id` as edi_recipient_qualifier_id,
`EDI.sender_qualifier_id` as edi_sender_qualifier_id,
`EDI.standard_version` as edi_standard_version,
explode(`FunctionalGroup`) as fg,
filename
from edi
) fgs
) trnx
) clms ;

drop table if exists remittance;
create table remittance as 
select filename, edi_control_number, edi_date, edi_time, edi_recipient_qualifier_id, edi_sender_qualifier_id, edi_standard_version,
fg_control_number, fg_date, fg_time, fg_receiver, fg_sender, fg_standard_version, fg_transaction_type,
clms.*
from stg_remittance;
```
![image](images/remittance.png?raw=true)


## Different EDI Formats 

Default format used is AnsiX12 (* as a delim and ~ as segment separator)

```python
from databricksx12 import *

#EDI format type
ediFormat = AnsiX12Delim #specifying formats of data, ansi is also the default if nothing is specified
#can also specify customer formats (below is the same as AnsiX12Delim)
ediFormat = type("", (), dict({'SEGMENT_DELIM': '~', 'ELEMENT_DELIM': '*', 'SUB_DELIM': ':'}))

df = spark.read.text("sampledata/837/*txt", wholetext = True)

(df.rdd
  .map(lambda x: x.asDict().get("value"))
  .map(lambda x: EDI(x, delim_cls = ediFormat))
  .map(lambda x: {"transaction_count": x.num_transactions()})
).toDF().show()
"""
+-----------------+
|transaction_count|
+-----------------+
|                5|
|                1|
|                1|
|                1|
|                1|
+-----------------+
"""
```

## Reading & Parsing Healthcare Transactions

Currently supports 837s. Records in each format type should be saved separately, e.g. do not mix 835s & 837s in the df.save() command.

## Sample data outside of Spark

```python
from databricksx12 import *
from databricksx12.hls import *
import json

hm = HealthcareManager()
edi =  EDI(open("sampledata/837/CHPW_Claimdata.txt", "rb").read().decode("utf-8"))

#Returns parsed claim data
hm.from_edi(edi) 
#[<databricksx12.hls.claim.Claim837p object at 0x106e57b50>, <databricksx12.hls.claim.Claim837p object at 0x106e57c40>, <databricksx12.hls.claim.Claim837p object at 0x106e57eb0>, <databricksx12.hls.claim.Claim837p object at 0x106e57b20>, <databricksx12.hls.claim.Claim837p object at 0x106e721f0>]

#Print in json format
print(json.dumps(hm.to_json(edi), indent=4)) 

"""
{
    "EDI.sender_tax_id": "ZZ",
    "FuncitonalGroup": [
        {
            "FunctionalGroup.receiver": "123456789",
            "FunctionalGroup.sender": "CLEARINGHOUSE",
            "FunctionalGroup.transaction_datetime": "20180508:0833",
            "FunctionalGroup.transaction_type": "222",
            "Transactions": [
                {
                    "Transaction.transaction_type": "222",
                    "Claims": [
                        {
                            "submitter": {
                                "contact_name": "CLEARINGHOUSE CLIENT SERVICES",
                                "contacts": {
                                    "primary": [
                                        {
                                            "contact_method": "Telephone",
                                            "contact_number": "8005551212",
...
"""

#print the raw EDI Segments of one claim
one_claim = hm.from_edi(edi)[0]
print("\n".join([y.data for y in one_claim.data])) #Print one claim to look at the segments of it
"""
BHT*0019*00*7349063984*20180508*0833*CH
NM1*41*2*CLEARINGHOUSE LLC*****46*987654321
PER*IC*CLEARINGHOUSE CLIENT SERVICES*TE*8005551212*FX*8005551212
NM1*40*2*123456789*****46*CHPWA
HL*1**20*1
NM1*85*2*BH CLINIC OF VANCOUVER*****XX*1122334455
N3*12345 MAIN ST
N4*VANCOUVER*WA*98662
REF*EI*720000000
PER*IC*CONTACT*TE*9185551212
NM1*87*2
N3*PO BOX 1234
N4*VANCOUVER*WA*986681234
HL*2*1*22*0
SBR*P*18**COMMUNITY HLTH PLAN OF WASH*****CI
NM1*IL*1*SUBSCRIBER*JOHN*J***MI*987321
N3*987 65TH PL
...
"""
```

## Raw EDI as a Table 

```python
""""
Look at all data refernce -> https://justransform.com/edi-essentials/edi-structure/
  (1) Including control header / ISA & IEA segments
"""
from pyspark.sql.functions import input_file_name

( df.withColumn("filename", input_file_name()).rdd
  .map(lambda x: (x.asDict().get("filename"),x.asDict().get("value")))
  .map(lambda x: (x[0], EDI(x[1])))
  .map(lambda x: [{**{"filename": x[0]}, **y} for y in x[1].toRows()])
  .flatMap(lambda x: x)
  .toDF()).show()

"""
+--------------------+----------+--------------------------+--------------+------------+-----------------------------+--------+
|            row_data|row_number|segment_element_delim_char|segment_length|segment_name|segment_subelement_delim_char|filename|
+--------------------+----------+--------------------------+--------------+------------+-----------------------------+--------+
|ISA*00*          ...|         0|                         *|            17|         ISA|                            :|file:///|
|GS*HC*CLEARINGHOU...|         1|                         *|             9|          GS|                            :|file:///|
|ST*837*000000001*...|         2|                         *|             4|          ST|                            :|file:///|
|BHT*0019*00*73490...|         3|                         *|             7|         BHT|                            :|file:///|
|NM1*41*2*CLEARING...|         4|                         *|            10|         NM1|                            :|file:///|
|PER*IC*CLEARINGHO...|         5|                         *|             7|         PER|                            :|file:///|
|NM1*40*2*12345678...|         6|                         *|            10|         NM1|                            :|file:///|
```

## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support. 
