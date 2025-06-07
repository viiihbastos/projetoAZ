# GameAnalytics Data Ingestion Accelerator

## Overview
This project is designed to help GameAnalytics customers onboard to Databricks by providing an example of a data ingestion pipeline based on the GameAnalytics event data.  It leverages Databricks on AWS to ingest the data exported from GameAnalytics DataSuite(link) into s3 and load it into Databricks through Delta Live Tables.

## Installation Instructions
- Fork the subdirectory into a versions control system (VCS)
- Use [Databricks Git Folders](https://docs.databricks.com/en/repos/repos-setup.html) to pull the repo from your VCS into Databricks 
- Use the 00_GameAnalytics_Data_Export_Integration notebook to link the s3 bucket that GameAnalytics DataSuite is exporting data to as an external location

## Configuration
The config.py file uses the following variables

- S3_PATH : Path that GameAnalytics pushes data to. GameAnalytics [Documentation](https://docs.gameanalytics.com/datasuite/data-export/configuration)
