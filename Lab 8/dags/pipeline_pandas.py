#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 11:26:13 2025

@author: dinglin
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import random

# DAG Definition
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 3, 1),
    'retries': 1,
}

dag = DAG(
    'ml_pipeline_pandas',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
)

# Install pandas
install_pandas = BashOperator(
    task_id="install_pandas",
    bash_command="pip install pandas",
    dag=dag,
)

# Simulated in-memory dataset
def load_data(**kwargs):
    import pandas as pd

    data = {
        'feature1': [random.randint(0, 100) for _ in range(100)],
        'feature2': [random.randint(0, 100) for _ in range(100)],
        'label': [random.choice([0, 1]) for _ in range(100)]
    }
    
    df = pd.DataFrame(data)
    kwargs['ti'].xcom_push(key='data', value=df.to_dict(orient="records"))  # Store data in Airflow XCom

def clean_impute(**kwargs):
    import pandas as pd

    ti = kwargs['ti']
    data_dict = ti.xcom_pull(task_ids='load_data', key='data')
    df = pd.DataFrame(data_dict)

    df.fillna(df.mean(), inplace=True)
    ti.xcom_push(key='cleaned_data', value=df.to_dict(orient="records"))

def feature_engineering(**kwargs):
    import pandas as pd

    ti = kwargs['ti']
    data_dict = ti.xcom_pull(task_ids='clean_impute', key='cleaned_data')
    df = pd.DataFrame(data_dict)

    # Feature Engineering: Create a new derived feature
    df['new_feature'] = df['feature1'] * 0.5
    ti.xcom_push(key='fe_data', value=df.to_dict(orient="records"))

def train_svm(**kwargs):
    import pandas as pd
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    ti = kwargs['ti']
    data_dict = ti.xcom_pull(task_ids='feature_engineering', key='fe_data')
    df = pd.DataFrame(data_dict)

    X = df[['feature1', 'new_feature']]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = SVC()
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"SVM Accuracy: {acc}")

def train_logistic(**kwargs):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    ti = kwargs['ti']
    data_dict = ti.xcom_pull(task_ids='feature_engineering', key='fe_data')
    df = pd.DataFrame(data_dict)

    X = df[['feature1', 'new_feature']]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Logistic Regression Accuracy: {acc}")

def merge_results():
    print("Merging model results and selecting best model.")

def evaluate_test():
    print("Evaluating final best model on test set.")

# Define Airflow tasks
t1 = PythonOperator(task_id="load_data", python_callable=load_data, provide_context=True, dag=dag)
t2 = PythonOperator(task_id="clean_impute", python_callable=clean_impute, provide_context=True, dag=dag)
t3 = PythonOperator(task_id="feature_engineering", python_callable=feature_engineering, provide_context=True, dag=dag)
t4a = PythonOperator(task_id="train_svm", python_callable=train_svm, provide_context=True, dag=dag)
t4b = PythonOperator(task_id="train_logistic", python_callable=train_logistic, provide_context=True, dag=dag)
t5 = PythonOperator(task_id="merge_results", python_callable=merge_results, dag=dag)
t6 = PythonOperator(task_id="evaluate_test", python_callable=evaluate_test, dag=dag)

# DAG dependencies
install_pandas >> t1  # Install dependencies before loading data
t1 >> t2  # Data Cleaning
t2 >> t3  # Feature Engineering
t3 >> [t4a, t4b]  # Train models in parallel
[t4a, t4b] >> t5  # Merge results
t5 >> t6  # Final evaluation
