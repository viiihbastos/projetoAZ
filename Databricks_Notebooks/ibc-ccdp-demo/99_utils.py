# Databricks notebook source
from pyspark.sql.functions import rand, when, col
from pyspark.sql.types import DoubleType, StringType
from pyspark.sql.functions import struct, coalesce
import mlflow
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd
import re
from hyperopt import fmin, tpe, rand, hp, Trials, STATUS_OK, SparkTrials, space_eval
from hyperopt.pyll.base import scope
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import average_precision_score, f1_score
