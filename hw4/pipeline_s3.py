#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 11:35:39 2025

@author: dinglin
"""

import boto3
import csv
import os
import random
import pandas as pd
from io import StringIO
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime

import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, LinearSVC, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from pyspark.ml import Pipeline

# AWS S3 Configuration
S3_BUCKET = os.getenv("matt-lab8", "matt-lab8")  # Replace with your actual S3 bucket
S3_PREFIX = ""  # Folder within S3 bucket
MY_KEY_ID = 'ASIAYAAO5HRMKQZOIGF3'
MY_ACCESS_KEY = 'TiNX2D2Jtufn3S3uwlrVYHySJh+kjvQZBbg728/F'
MY_SESSION_TOKEN = 'IQoJb3JpZ2luX2VjEDoaCXVzLWVhc3QtMiJHMEUCIBQ8mLq/JRpEF9PP8BfJD/Ickx9jKxJoWTJO3+JaL5TFAiEAxaOFMmWgDBGB6OlSG0ikKcFbRaTy0i7n/XIA6j1kr4Yq9AIIhP//////////ARAAGgw1NDk3ODcwOTAwMDgiDA2MOLD2jVh5grwPUCrIAonpENWBL6gaWsfxVk9i8Z31BK3AaL6IubAdCO78Gw6LQ8gnx9o52/GnomAPqdPjq0Fin0wPCv8lJv5cZMWARDkE8jbaDXOeryoXtQtM3wlWXKn+neOCU8sAgwuhesfla/ORs5Zy16l8ZjQjh8a88kjGSXXfJ4jfJoPYSkjA/KMUqZwOqOQknh6iU8DBHacdWhc+fpdiIimj6MPMPZCRgqqwy8lotGtPC/WW3xQJl9183lWdi9Qu2nh6lgRUdaigynYKiILcJ9TmQ8huVobnIp0LYbl+RsIUfG+fnpoXs6hzvcKK0sb/onVzSRxnFxrXXL+dCE5+rAafJgwLe14+f4LIo0b3un3+ypB0/0ZaDTlQsxRXag9dsyjOFWOW6vL8+JTgh4NeLyicsGULctxKgaCCl/KnCssUVCLovf7iz7u8e4+5txIliFgw0Z25vgY6pwFHlMW6UwyG5QmrYZR/zHXKW7/i+ppoO1J2JgBUQGxfTG2MzfTuJVjksAyDy/shOoQB8faBw+oOufR8yVNKt2ALBMFOdpcYz+PK2OjsN6AhnfAR53jbj4XPlKgCyDZE0hVMGQWaKedh3NWzWgljBPt4r8/AtgIDmjTIDUI7WKfxBTJ7hrl4vVMrcNSNt3D92ilsKluZ0Li5soK+tEEr/4N5qbstRXfijQ=='

s3_client = boto3.client('s3',
                         aws_access_key_id=MY_KEY_ID,
                         aws_secret_access_key=MY_ACCESS_KEY,
                         aws_session_token=MY_SESSION_TOKEN)

# Create Spark session
spark = SparkSession.builder.appName("HeartDiseasePrediction").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Helper function to upload CSV to S3
def upload_to_s3(data, filename):
    csv_buffer = StringIO()
    data.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=S3_BUCKET, Key=f"{filename}", Body=csv_buffer.getvalue())

# Helper function to download CSV from S3
def download_from_s3(filename):
    obj = s3_client.get_object(Bucket=S3_BUCKET, Key=f"{filename}")
    return pd.read_csv(obj['Body'])

# DAG Definition
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 3, 1),
    'retries': 1,
}

dag = DAG(
    'ml_pipeline_s3',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
)

# Install pandas
install_pandas = BashOperator(
    task_id="install_pandas",
    bash_command="pip install pandas boto3 scikit-learn",
    dag=dag,
)

install_pyspark = BashOperator(
    task_id="install_pyspark",
    bash_command="pip install pyspark",
    dag=dag,
)

# Task: Generate and upload random data to S3
def load_data():
    df = download_from_s3("heart_disease.csv")
    if df.empty:
        print("DataFrame is empty!")
    else:
        print("DataFrame has data.")
    local_file = 'heart_disease.csv'
    s3_client.download_file(S3_BUCKET, 'heart_disease.csv', local_file)
    df_spark = spark.read.csv(local_file, header=True, inferSchema=True)
    if df_spark.count() == 0:
        print("Spark DataFrame is empty!")
    else:
        print("Spark DataFrame has data.")

load_task = PythonOperator(
    task_id="load_data",
    python_callable=load_data,
    dag=dag,
)

def sklearn_impute_smoking_1(url, df_cleaned):
    # Fetch the webpage
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the page with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the table containing the data (Inspect the page to identify the correct table)
        tables = soup.find_all('table', {'class': 'responsive-enabled'})
        table = tables[1]  # Assuming this is the correct table

        # Initialize a dictionary to store the data
        smoking_rate_by_age = {}

        # Use data from 2011-12 column
        # Loop through each row in the table to populate the dictionary
        for row in table.find_all('tr')[1:]:  # Skip the header row
            columns = row.find_all('td')
            if len(columns) > 1:
                age_group = row.find('th').text.strip()  # Get the age group
                second_value = columns[0].text.strip()  # Get the second value (the second percentage)
                
                # Add the data to the dictionary
                smoking_rate_by_age[age_group] = float(second_value) / 100

        #print(smoking_rate_by_age)

        # Step 1: Create a new 'smoke_source_1' column and copy the values from the 'smoke' column
        df_cleaned.loc[:, 'smoke_source_1'] = df_cleaned['smoke']

        # Step 2: Function to map age to the corresponding age group
        def get_age_group_sklearn_1(age):
            if age >= 15 and age <= 17:
                return '15–17'
            elif age >= 18 and age <= 24:
                return '18–24'
            elif age >= 25 and age <= 34:
                return '25–34'
            elif age >= 35 and age <= 44:
                return '35–44'
            elif age >= 45 and age <= 54:
                return '45–54'
            elif age >= 55 and age <= 64:
                return '55–64'
            elif age >= 65 and age <= 74:
                return '65–74'
            elif age >= 75:
                return '75 years and over'
            return None  # In case the age doesn't match any group

        # Step 3: Impute missing values in the 'smoke_source_1' column based on the 'age'
        def impute_smoking_rate_sklearn_1(row):
            if pd.isna(row['smoke_source_1']):  # If the value is missing
                age_group = get_age_group_sklearn_1(int(row['age']))  # Map age to age group
                if age_group and age_group in smoking_rate_by_age:
                    return smoking_rate_by_age[age_group]  # Get the corresponding smoking rate
                else:
                    return None  # If no corresponding smoking rate is found
            return row['smoke_source_1']  # If not missing, return the original value

        # Apply the function to the dataframe to impute missing values
        df_cleaned['smoke_source_1'] = df_cleaned.apply(impute_smoking_rate_sklearn_1, axis=1)

        # Return the dataframe with the new imputed values
        return df_cleaned

def sklearn_impute_smoking_2(df_cleaned):
    # Hardcoded smoking rates for males and females
    smoking_rate_female = 0.1  # Female smoking rate
    smoking_rate_male = 0.132  # Male smoking rate

    # Hardcoded smoking rates by age group
    smoking_rate_by_age = {
        '18–24': 0.048,  # 4.8% -> 0.048
        '25–44': 0.125,  # 12.5% -> 0.125
        '45–64': 0.151,  # 15.1% -> 0.151
        '65 and above': 0.087  # 8.7% -> 0.087
    }

    # Step 1: Create a new 'smoke_source_2' column and copy the values from the 'smoke' column
    df_cleaned.loc[:, 'smoke_source_2'] = df_cleaned['smoke']

    # Step 2: Function to map age to the corresponding age group
    def get_age_group_sklearn_2(age):
        if age >= 18 and age <= 24:
            return '18–24'
        elif age >= 25 and age <= 44:
            return '25–44'
        elif age >= 45 and age <= 64:
            return '45–64'
        elif age >= 65:
            return '65 and above'
        return None  # In case the age doesn't match any group

    # Step 3: Impute missing values in the 'smoke_source_2' column based on the 'age' and 'sex'
    def impute_smoking_rate_sklearn_2(row):
        if pd.isna(row['smoke_source_2']):  # If the value is missing
            age_group = get_age_group_sklearn_2(int(row['age']))  # Map age to age group
            if age_group:
                # Impute for females
                if row['sex'] == 0:
                    return smoking_rate_by_age[age_group]
                # Impute for males
                elif row['sex'] == 1:
                    return smoking_rate_by_age[age_group] * (smoking_rate_male / smoking_rate_female)
        return row['smoke_source_2']  # If not missing, return the original value

    # Apply the function to the dataframe to impute missing values
    df_cleaned['smoke_source_2'] = df_cleaned.apply(impute_smoking_rate_sklearn_2, axis=1)

    # Return the dataframe with the new imputed values
    return df_cleaned

# Task: Read data from S3 and clean
def clean_impute_sklearn():
    df = download_from_s3("heart_disease.csv")
    
    columns_to_retain = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'smoke', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 'exang', 'oldpeak', 'slope', 'target']
    df_cleaned = df[columns_to_retain]
    #print("\nCleaned Data:")
    #print(df_cleaned.head())  # Print first few rows of the cleaned dataset

    # Impute painloc and painexer using the mode b/c binary data
    df_cleaned.loc[:, 'painloc'] = df_cleaned['painloc'].fillna(df_cleaned['painloc'].mode()[0])
    df_cleaned.loc[:, 'painexer'] = df_cleaned['painexer'].fillna(df_cleaned['painexer'].mode()[0])

    # Impute trestbps values less than 100 mm Hg using 100 mm Hg
    df_cleaned.loc[:, 'trestbps'] = df_cleaned['trestbps'].clip(lower=100)

    # Impute oldpeak values less than 0 with 0 and those greater than 4 with 4
    df_cleaned.loc[:, 'oldpeak'] = df_cleaned['oldpeak'].clip(lower=0, upper=4)

    # Impute thaldur and thalach using the mean because values are pretty random
    df_cleaned.loc[:, 'thaldur'] = df_cleaned['thaldur'].fillna(df_cleaned['thaldur'].mean())
    df_cleaned.loc[:, 'thalach'] = df_cleaned['thalach'].fillna(df_cleaned['thalach'].mean())

    # Impute fbs, prop, nitr, pro, and diuretic missing values with mode and clip values greater than 1 to 1
    columns_to_fix = ['fbs', 'prop', 'nitr', 'pro', 'diuretic']

    for col in columns_to_fix:
        # Replace missing values with the mode (most frequent value)
        mode_value = df_cleaned[col].mode()[0]
        df_cleaned.loc[:, col] = df_cleaned[col].fillna(mode_value)
        
        # Clip values greater than 1 to 1
        df_cleaned.loc[:, col] = df_cleaned[col].clip(upper=1)

    # Check for missing exang values
    # print(df_cleaned['exang'].unique())
    # Impute exang values using the mode b/c binary data
    df_cleaned.loc[:, 'exang'] = df_cleaned['exang'].fillna(df_cleaned['exang'].mode()[0])
    # Verify missing values have been imputed
    # print(df_cleaned['exang'].unique())

    # Check for missing slope values
    # print(df_cleaned['slope'].unique())
    # Impute slope values using the mode b/c only 4 possible values
    df_cleaned.loc[:, 'slope'] = df_cleaned['slope'].fillna(df_cleaned['slope'].mode()[0])
    # Verify missing slope values have been imputed
    # print(df_cleaned['slope'].unique())

    # Imput missing age values with the mode
    df_cleaned.loc[:, 'age'] = df_cleaned['age'].fillna(df_cleaned['age'].mode()[0])

    # Imput missing sex values with the mode
    df_cleaned.loc[:, 'sex'] = df_cleaned['sex'].fillna(df_cleaned['sex'].mode()[0])

    # Imput missing cp values with the mode
    df_cleaned.loc[:, 'cp'] = df_cleaned['cp'].fillna(df_cleaned['cp'].mode()[0])

    # Imput missing trestbps values with the mode
    df_cleaned.loc[:, 'trestbps'] = df_cleaned['trestbps'].fillna(df_cleaned['trestbps'].mode()[0])

    # Imput missing oldpeak values with the mean
    df_cleaned.loc[:, 'oldpeak'] = df_cleaned['oldpeak'].fillna(df_cleaned['oldpeak'].mean())

    # Imput missing target values with the mode
    df_cleaned.loc[:, 'target'] = df_cleaned['target'].fillna(df_cleaned['target'].mode()[0])

    # Verify there are no missing values in the columns were missing values were replaced
    #missing_values = df_cleaned.isnull().sum()
    #missing_columns = missing_values[missing_values > 0]
    #print("Columns with missing values:")
    #print(missing_columns)

    # Only keep the first 899 rows because the rows that follow are not in the correct format
    df_cleaned = df_cleaned.iloc[:899]
    
    #print(df_cleaned)

    url = 'https://www.abs.gov.au/statistics/health/health-conditions-and-risks/smoking-and-vaping/latest-release'
    sklearn_smoking_data_one = sklearn_impute_smoking_1(url, df_cleaned)
    sklearn_smoking_data_two = sklearn_impute_smoking_2(df_cleaned)

    print(sklearn_smoking_data_two)
    
    ### RE-UPLOAD AS NEW SKLEARN CSV
    upload_to_s3(sklearn_smoking_data_two, "sklearn_cleaned_data.csv")

clean_sklearn_task = PythonOperator(
    task_id="clean_impute_sklearn",
    python_callable=clean_impute_sklearn,
    dag=dag,
)

def pyspark_impute_smoking_1(url, df_cleaned):
    # Fetch the webpage
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the page with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the table containing the data (Inspect the page to identify the correct table)
        tables = soup.find_all('table', {'class': 'responsive-enabled'})
        table = tables[1]  # Assuming this is the correct table

        # Initialize a dictionary to store the data
        smoking_rate_by_age = {}

        # Use data from 2011-12 column
        # Loop through each row in the table to populate the dictionary
        for row in table.find_all('tr')[1:]:  # Skip the header row
            columns = row.find_all('td')
            if len(columns) > 1:
                age_group = row.find('th').text.strip()  # Get the age group
                second_value = columns[0].text.strip()  # Get the second value (the second percentage)
                
                # Add the data to the dictionary
                smoking_rate_by_age[age_group] = float(second_value) / 100

        #print(smoking_rate_by_age)

        # Step 1: Create a new 'smoke_source_1' column and copy the values from the 'smoke' column
        df_cleaned = df_cleaned.withColumn("smoke_source_1", col("smoke"))

        # Step 2: Function to map age to the corresponding age group
        def get_age_group(age):
            age = int(age)
            if age >= 15 and age <= 17:
                return '15–17'
            elif age >= 18 and age <= 24:
                return '18–24'
            elif age >= 25 and age <= 34:
                return '25–34'
            elif age >= 35 and age <= 44:
                return '35–44'
            elif age >= 45 and age <= 54:
                return '45–54'
            elif age >= 55 and age <= 64:
                return '55–64'
            elif age >= 65 and age <= 74:
                return '65–74'
            elif age >= 75:
                return '75 years and over'
            return None  # In case the age doesn't match any group

        # Step 3: Define a UDF to apply the imputation logic
        def impute_smoking_rate(smoke_value, age):
            if smoke_value is None:  # If the value is missing
                age_group = get_age_group(age)  # Map age to age group
                if age_group and age_group in smoking_rate_by_age:
                    return float(smoking_rate_by_age[age_group])  # Get the corresponding smoking rate
                else:
                    return None  # If no corresponding smoking rate is found
            return float(smoke_value)  # If not missing, return the original value

        # Convert the function to a UDF
        impute_smoking_rate_udf = udf(impute_smoking_rate, FloatType())

        # Apply the UDF to to impute the missing values
        df_cleaned = df_cleaned.withColumn("smoke_source_1", impute_smoking_rate_udf(col("smoke_source_1"), col("age")))
        
        # Show the first 5 rows
        df_cleaned.show(5)

        # Return the dataframe with the new imputed values
        return df_cleaned

def pyspark_impute_smoking_2(df_cleaned):
    # Hardcoded smoking rates for males and females
    smoking_rate_female = 0.1  # Female smoking rate
    smoking_rate_male = 0.132  # Male smoking rate

    # Hardcoded smoking rates by age group
    smoking_rate_by_age = {
        '18–24': 0.048,  # 4.8% -> 0.048
        '25–44': 0.125,  # 12.5% -> 0.125
        '45–64': 0.151,  # 15.1% -> 0.151
        '65 and above': 0.087  # 8.7% -> 0.087
    }

    # Step 1: Create a new 'smoke_source_2' column and copy the values from the 'smoke' column
    df_cleaned = df_cleaned.withColumn("smoke_source_2", col("smoke"))

    # Step 2: Function to map age to the corresponding age group
    def get_age_group(age):
        age = int(age)
        if age >= 18 and age <= 24:
            return '18–24'
        elif age >= 25 and age <= 44:
            return '25–44'
        elif age >= 45 and age <= 64:
            return '45–64'
        elif age >= 65:
            return '65 and above'
        return None  # In case the age doesn't match any group

    # Step 3: Impute missing values in the 'smoke_source_2' column based on the 'age' and 'sex'
    def impute_smoking_rate(smoke_value, age, sex):
        if smoke_value is None:  # If the value is missing
            age_group = get_age_group(age)  # Map age to age group
            if age_group:
                # Impute for females
                if sex == 0:
                    return smoking_rate_by_age[age_group]
                # Impute for males
                elif sex == 1:
                    return smoking_rate_by_age[age_group] * (smoking_rate_male / smoking_rate_female)
        return smoke_value  # If not missing, return the original value

    # Convert the function to a UDF
    impute_smoking_rate_udf = udf(impute_smoking_rate, FloatType())

    # Apply the UDF to to impute the missing values
    df_cleaned = df_cleaned.withColumn("smoke_source_2", impute_smoking_rate_udf(col("smoke_source_2"), col("age"), col("sex")))

    df_cleaned = df_cleaned.withColumn(
        "smoke_source_2", 
        when(col("smoke_source_2").isNull(), col("smoke")).otherwise(col("smoke_source_2"))
    )
    
    # Show the first 5 rows
    df_cleaned.show(5)

    # Return the dataframe with the new imputed values
    return df_cleaned


# Task: Read data from S3 and clean
def clean_impute_pyspark():
    local_file = 'heart_disease.csv'
    s3_client.download_file(S3_BUCKET, 'heart_disease.csv', local_file)
    df_cleaned = spark.read.csv(local_file, header=True, inferSchema=True)

    # List of columns to retain
    columns_to_retain = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'smoke', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 'exang', 'oldpeak', 'slope', 'target']
    
    # Select only the columns that are in the list
    df_cleaned = df_cleaned.select(*columns_to_retain)
    
    # Calculate mode for 'painloc' column
    painloc_mode = df_cleaned.filter(F.col('painloc').isNotNull()) \
        .groupBy('painloc') \
        .agg(F.count('*').alias('count')) \
        .orderBy(F.desc('count')) \
        .first()['painloc']
    
    # Fill missing values in 'painloc' column with the mode
    df_cleaned = df_cleaned.fillna({'painloc': painloc_mode})
    
    # Calculate mode for 'painexer' column
    painexer_mode = df_cleaned.filter(F.col('painexer').isNotNull()) \
        .groupBy('painexer') \
        .agg(F.count('*').alias('count')) \
        .orderBy(F.desc('count')) \
        .first()['painexer']
    
    # Fill missing values in 'painexer' column with the mode
    df_cleaned = df_cleaned.fillna({'painexer': painexer_mode})

    # Impute trestbps values less than 100 mm Hg using 100 mm Hg
    df_cleaned = df_cleaned.withColumn(
        'trestbps',
        F.when(F.col('trestbps') < 100, 100).otherwise(F.col('trestbps')))

    # Impute oldpeak values less than 0 with 0 and those greater than 4 with 4
    df_cleaned = df_cleaned.withColumn(
        'oldpeak',
        F.when(F.col('oldpeak') < 0, 0).when(F.col('oldpeak') > 4, 4).otherwise(F.col('oldpeak')))
    
    # Impute thaldur and thalach using the mean because values are pretty random
    thaldur_mean = df_cleaned.agg(F.mean('thaldur')).collect()[0][0]
    thalach_mean = df_cleaned.agg(F.mean('thalach')).collect()[0][0]
    df_cleaned = df_cleaned.fillna({'thaldur': thaldur_mean, 'thalach': thalach_mean})

    # Impute fbs, prop, nitr, pro, and diuretic missing values with mode and clip values greater than 1 to 1
    # List of columns to fix
    columns_to_fix = ['fbs', 'prop', 'nitr', 'pro', 'diuretic']

    # Loop through each column to impute and clip
    for col in columns_to_fix:
        # Calculate the mode (most frequent value) for the column
        mode_value = df_cleaned.groupBy(col).count().orderBy(F.desc('count')).first()[0]
        
        # Impute missing values with the mode value
        df_cleaned = df_cleaned.fillna({col: mode_value})
        
        # Clip values greater than 1 to 1
        df_cleaned = df_cleaned.withColumn(col, F.when(F.col(col) > 1, 1).otherwise(F.col(col)))
    
    # Calculate mode for 'exang' column
    exang_mode = df_cleaned.filter(F.col('exang').isNotNull()) \
        .groupBy('exang') \
        .agg(F.count('*').alias('count')) \
        .orderBy(F.desc('count')) \
        .first()['exang']
    
    # Impute exang values using the mode b/c binary data
    df_cleaned = df_cleaned.fillna({'exang': exang_mode})

    # Calculate mode for 'slope' column
    slope_mode = df_cleaned.filter(F.col('slope').isNotNull()) \
        .groupBy('slope') \
        .agg(F.count('*').alias('count')) \
        .orderBy(F.desc('count')) \
        .first()['slope']
    
    # Impute slope values using the mode b/c only 4 possible values
    df_cleaned = df_cleaned.fillna({'slope': slope_mode})

    # Impute missing age, sex, cp, trestbps, and target values with the mode
    columns_to_fix = ['age', 'sex', 'cp', 'trestbps', 'target']

    # Loop through each column to calculate mode and impute missing values
    for col in columns_to_fix:
        # Calculate mode for the current column
        mode_value = df_cleaned.filter(F.col(col).isNotNull()) \
            .groupBy(col) \
            .agg(F.count('*').alias('count')) \
            .orderBy(F.desc('count')) \
            .first()[col]
        
        # Impute missing values in the current column with the mode value
        df_cleaned = df_cleaned.fillna({col: mode_value})
    
    # Impute missing oldpeak values with the mean
    oldpeak_mean = df_cleaned.agg(F.mean('oldpeak')).collect()[0][0]
    df_cleaned = df_cleaned.fillna({'oldpeak': oldpeak_mean})

    # Only keep the first 899 rows because the rows that follow are not in the correct format
    df_cleaned = df_cleaned.limit(899)

    df_cleaned.show(5)

    # Verify that only the first 899 rows are in the cleaned dataframe
    # print(f"Number of rows: {df_cleaned.count()}")

    # Verify all trestbps values less than 100 mm Hg have been imputed
    # df_cleaned.filter(F.col('trestbps') < 100).show()
    
    # Verify all oldpak values less than 0 and greater than 4 have been imputed
    # df_cleaned.filter(F.col('oldpeak') < 0).show()
    # df_cleaned.filter(F.col('oldpeak') > 4).show()
    
    # Verify all missing thaldur and thalach values have been imputed
    # df_cleaned.filter(F.col('thaldur').isNull()).show()
    # df_cleaned.filter(F.col('thalach').isNull()).show()
    
    # Verify that all missing fbs, prop, nitr, pro, and diuretic values have been imputed
    # df_cleaned.filter(F.col('fbs').isNull()).show()
    # df_cleaned.filter(F.col('prop').isNull()).show()
    # df_cleaned.filter(F.col('nitr').isNull()).show()
    # df_cleaned.filter(F.col('pro').isNull()).show()
    # df_cleaned.filter(F.col('diuretic').isNull()).show()
   
    # Verify that all missing exang values have been imputed
    # df_cleaned.filter(F.col('exang').isNull()).show()
    
    # Verify that all missing slope values have been imputed
    # df_cleaned.filter(F.col('slope').isNull()).show()
    
    # Verify missing age, sex, cp, trestbps, and target values have been imputed
    # df_cleaned.filter(F.col('age').isNull()).show()
    # df_cleaned.filter(F.col('sex').isNull()).show()
    # df_cleaned.filter(F.col('cp').isNull()).show()
    # df_cleaned.filter(F.col('trestbps').isNull()).show()
    # df_cleaned.filter(F.col('target').isNull()).show()
    
    # Verify missing oldpeak values have been imputed
    # df_cleaned.filter(F.col('oldpeak').isNull()).show()

    
    url = 'https://www.abs.gov.au/statistics/health/health-conditions-and-risks/smoking-and-vaping/latest-release'
    pyspark_smoking_data_one = pyspark_impute_smoking_1(url, df_cleaned)
    pyspark_smoking_data_two = pyspark_impute_smoking_2(pyspark_smoking_data_one)

    print(pyspark_smoking_data_two)
    
    ### RE-UPLOAD AS NEW PYSPARK CSV
    upload_to_s3(pyspark_smoking_data_two.toPandas(), "pyspark_cleaned_data.csv")
    


clean_pyspark_task = PythonOperator(
    task_id="clean_impute_pyspark",
    python_callable=clean_impute_pyspark,
    dag=dag,
)


# Task: Perform feature engineering for sklearn - maximum heart rate
def feature_engineering_1():
    df = download_from_s3("sklearn_cleaned_data.csv")
    df["max_HR"] = 206.9 - (0.67 * df["age"])
    upload_to_s3(df, "fe_data_1.csv")

feature_task_1 = PythonOperator(
    task_id="feature_engineering_1",
    python_callable=feature_engineering_1,
    dag=dag,
)

# Task: Perform feature engineering for pyspark - blood pressure difference from normal
def feature_engineering_2():
    local_file = 'pyspark_cleaned_data.csv'
    s3_client.download_file(S3_BUCKET, 'pyspark_cleaned_data.csv', local_file)
    df = spark.read.csv(local_file, header=True, inferSchema=True)
    df = df.withColumn("bp_diff_from_norm", col("trestbps") - 120)
    upload_to_s3(df.toPandas(), "fe_data_2.csv")

feature_task_2 = PythonOperator(
    task_id="feature_engineering_2",
    python_callable=feature_engineering_2,
    dag=dag,
)

# Task: Train SVM Model
def train_svm_fe1(**kwargs):
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    ti = kwargs['ti']  # Task instance for XCom

    df = download_from_s3("fe_data_1.csv")

    X = df.drop(columns=['target', 'smoke'])  # Features + smoke column removed because there are NaN values and we will use two new smoke columns instead
    X = X.apply(pd.to_numeric, errors='coerce')
    y = df['target']  # Target variable
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    model = SVC(probability=True, random_state=42)
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    #model.fit(X_train, y_train)
    #acc = accuracy_score(y_test, model.predict(X_test))
    best_model = grid_search.best_estimator_
    acc = accuracy_score(y_test, best_model.predict(X_test))
    print(f"SVM Accuracy: {acc}")

    # Push accuracy to XCom
    ti.xcom_push(key="SVM_FE1_accuracy", value=acc)

svm_task_fe1 = PythonOperator(
    task_id="train_svm_fe1",
    python_callable=train_svm_fe1,
    provide_context=True,
    dag=dag,
)

# Task: Train Logistic Regression Model
def train_logistic_fe1(**kwargs):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    ti = kwargs['ti']  # Task instance for XCom

    df = download_from_s3("fe_data_1.csv")
    X = df.drop(columns=['target', 'smoke'])  # Features + smoke column removed because there are NaN values and we will use two new smoke columns instead
    X = X.apply(pd.to_numeric, errors='coerce')
    y = df['target']  # Target variable
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=10000, random_state=42)
    param_grid = {'C': [0.1, 1, 10]}  # Hyperparameter tuning for Logistic Regression
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    #model.fit(X_train, y_train)
    
    #acc = accuracy_score(y_test, model.predict(X_test))
    best_model = grid_search.best_estimator_
    acc = accuracy_score(y_test, best_model.predict(X_test))
    print(f"Logistic Regression Accuracy: {acc}")

    # Push accuracy to XCom
    ti.xcom_push(key="Logistic_FE1_accuracy", value=acc)

logistic_task_fe1 = PythonOperator(
    task_id="train_logistic_fe1",
    python_callable=train_logistic_fe1,
    provide_context=True,
    dag=dag,
)

# Task: Train SVM Model
def train_svm_fe2(**kwargs):
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    ti = kwargs['ti']  # Task instance for XCom
    
    local_file = 'fe_data_2.csv'
    s3_client.download_file(S3_BUCKET, 'fe_data_2.csv', local_file)
    df = spark.read.csv(local_file, header=True, inferSchema=True)
    df = df.drop("smoke")
    feature_columns = [col for col in df.columns if col != 'target']
    df = df.select(*[col(c).cast('double') for c in feature_columns] + ['target'])
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df = assembler.transform(df)
    train_df, test_df = df.randomSplit([0.9, 0.1], seed=42)
    svm = LinearSVC(labelCol="target", featuresCol="features", maxIter=1000, regParam=0.1)
    param_grid = ParamGridBuilder().addGrid(svm.regParam, [0.1, 1.0]).build()
    evaluator = BinaryClassificationEvaluator(labelCol="target", metricName="areaUnderROC")

    cv = CrossValidator(estimator=svm, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)

    # Step 6: Train the model
    svm_model = cv.fit(train_df)  # Using CrossValidator for tuning
    # svm_model = svm.fit(train_df)  # Direct training without CrossValidator
    
    # Step 7: Evaluate the model on the test data
    svm_pred = svm_model.transform(test_df)  # Use the fitted model to predict on the test set
    svm_auc = evaluator.evaluate(svm_pred)
    
    # Step 8: Get accuracy using MulticlassClassificationEvaluator (for classification tasks)
    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="target", metricName="accuracy")
    svm_accuracy = accuracy_evaluator.evaluate(svm_pred)
    
    print(f"SVM AUC: {svm_auc:.4f}")
    print(f"SVM Accuracy: {svm_accuracy:.4f}")
    
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    #model.fit(X_train, y_train)
    
    #acc = accuracy_score(y_test, model.predict(X_test))
    #best_model = grid_search.best_estimator_
    #acc = accuracy_score(y_test, best_model.predict(X_test))

    # Push accuracy to XCom
    ti.xcom_push(key="SVM_FE2_accuracy", value=svm_accuracy)

svm_task_fe2 = PythonOperator(
    task_id="train_svm_fe2",
    python_callable=train_svm_fe2,
    provide_context=True,
    dag=dag,
)

# Task: Train Logistic Regression Model
def train_logistic_fe2(**kwargs):
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    ti = kwargs['ti']  # Task instance for XCom
    
    local_file = 'fe_data_2.csv'
    s3_client.download_file(S3_BUCKET, 'fe_data_2.csv', local_file)
    df = spark.read.csv(local_file, header=True, inferSchema=True)
    df = df.drop("smoke")
    feature_columns = [col for col in df.columns if col != 'target']
    df = df.select(*[col(c).cast('double') for c in feature_columns] + ['target'])
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df = assembler.transform(df)
    train_df, test_df = df.randomSplit([0.9, 0.1], seed=42)
    lr = LogisticRegression(labelCol="target", featuresCol="features", maxIter=10000)
    lr_param_grid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 1.0]).build()
    
    auc_evaluator = BinaryClassificationEvaluator(labelCol="target", metricName="areaUnderROC")
    f1_evaluator = MulticlassClassificationEvaluator(labelCol="target", metricName="f1")
    
    lr_cv = CrossValidator(estimator=lr, estimatorParamMaps=lr_param_grid, evaluator=auc_evaluator, numFolds=5)
    lr_model = lr_cv.fit(train_df)
    
    lr_pred = lr_model.transform(test_df)
    lr_auc = auc_evaluator.evaluate(lr_pred)
    lr_f1 = f1_evaluator.evaluate(lr_pred)
    # print(f"Logistic Regression AUC: {lr_auc}")
    # print(f"Logistic Regression - AUC: {lr_auc:.4f}, F1: {lr_f1:.4f}")
    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="target", metricName="accuracy")
    
    # Logistic Regression Accuracy
    lr_accuracy = accuracy_evaluator.evaluate(lr_pred)
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    #model.fit(X_train, y_train)
    
    #acc = accuracy_score(y_test, model.predict(X_test))
    #best_model = grid_search.best_estimator_
    #acc = accuracy_score(y_test, best_model.predict(X_test))
    print(f"LR Accuracy: {lr_accuracy}")

    # Push accuracy to XCom
    ti.xcom_push(key="Logistic_FE2_accuracy", value=lr_accuracy)

logistic_task_fe2 = PythonOperator(
    task_id="train_logistic_fe2",
    python_callable=train_logistic_fe2,
    provide_context=True,
    dag=dag,
)


# Task: Merge Model Results
def merge_results(**kwargs):
    print("Merging model results and selecting the best model.")
    ti = kwargs['ti']

    # Retrieve accuracy values from XCom
    accuracies = {
        "SVM_FE1": ti.xcom_pull(task_ids="train_svm_fe1", key="SVM_FE1_accuracy"),
        "Logistic_FE1": ti.xcom_pull(task_ids="train_logistic_fe1", key="Logistic_FE1_accuracy"),
        "SVM_FE2": ti.xcom_pull(task_ids="train_svm_fe2", key="SVM_FE2_accuracy"),
        "Logistic_FE2": ti.xcom_pull(task_ids="train_logistic_fe2", key="Logistic_FE2_accuracy"),
    }

    # Filter out None values
    accuracies = {k: v for k, v in accuracies.items() if v is not None}

    print(f"Model Accuracies: {accuracies}")

    if accuracies:  # Ensure there is at least one valid accuracy value
        best_model = max(accuracies, key=accuracies.get)
        best_accuracy = accuracies[best_model]
        print(f"Best Model: {best_model} | Accuracy: {best_accuracy}")
    else:
        print("No valid accuracies found.")

    # Push best model and accuracy to XCom
    ti.xcom_push(key="best_model", value=best_model)
    ti.xcom_push(key="best_accuracy", value=best_accuracy)

merge_task = PythonOperator(
    task_id="merge_results",
    python_callable=merge_results,
    provide_context=True,
    dag=dag,
)

# Task: Final Evaluation
def evaluate_test(**kwargs):
    print("Evaluating final model performance.")
    ti = kwargs['ti']

    # Retrieve best model from XCom
    best_model = ti.xcom_pull(task_ids="merge_results", key="best_model")
    best_accuracy = ti.xcom_pull(task_ids="merge_results", key="best_accuracy")

    print(f"Final Model: {best_model} | Accuracy: {best_accuracy}")

evaluate_task = PythonOperator(
    task_id="evaluate_test",
    python_callable=evaluate_test,
    provide_context=True,
    dag=dag,
)

# DAG dependencies
install_pandas >> install_pyspark >> load_task  # Install dependencies before starting
load_task >> clean_sklearn_task  # Sklearn Data Cleaning
load_task >> clean_pyspark_task  # Pyspark Data Cleaning
clean_sklearn_task >> feature_task_1  # Feature Engineering
clean_pyspark_task >> feature_task_2  # Feature Engineering
feature_task_1 >> [svm_task_fe1, logistic_task_fe1]   # Train models in parallel
feature_task_2 >> [logistic_task_fe2, svm_task_fe2]   # Train models in parallel
[svm_task_fe1, logistic_task_fe1, svm_task_fe2, logistic_task_fe2] >> merge_task >> evaluate_task # Merge results + Final evaluation