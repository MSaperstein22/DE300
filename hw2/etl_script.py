import os
import boto3
import pandas as pd
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

# AWS S3 Setup
S3_BUCKET = 'de300winter2025'
S3_SUBFOLDER = 'matt_saperstein'
S3_FILE = f'{S3_SUBFOLDER}/heart_disease.csv'

my_aws_access_key_id='ASIAYAAO5HRMKBLELQSK'
my_aws_secret_access_key='Cks7TV1fiFDyHbHP1gWv605T5ookV9QtWrV9fPo1'
my_aws_session_token='IQoJb3JpZ2luX2VjEK7//////////wEaCXVzLWVhc3QtMiJIMEYCIQDsDq1fSN+OQdzEw7HaRky40+M/vXt2/dE+1rz4tRr/jQIhAM7exCMv+mt7MtThgk4CvcXsj+qTR0eXhbR88AZPI36tKvQCCMf//////////wEQABoMNTQ5Nzg3MDkwMDA4IgykG/EdJmXmqUtsbFMqyALnxptRqcV8pUATIXYnK9ABerjsgs0cKXcvytUC0vP7uzyK4HssddiKbgdUpCj+Vocv3RKKDfpjYH09Bw9CVnz3yww2WO80wdnohcxugUYOxwPybMmJDGtJLum2P+U2J7qTUs2pp99uLLObRsTqVVXf+ZwsgLoOY7vqeVyOiMUzKa0gLsXJUcI6yxrAAgystXjSXw8tbY0p3DO5h5JkqlS91ccKXPX2MqOWKUf2GNeXrPT/z3XV834oLtvbXMnzMOVt6rcTIpCzD61Ev6nni5Jv/yMbloAvrdDYLvdMgBcVvIXDkglHmW3F66sGxBhhjn/SSiF1dTQ/ciL7TJhDGQ43WKf9HshJYHddYnJbhocHVn+qCUdCibn+/AZkPtCHdeWyjelOpPgrVdqeM7EeLsbQSUBaaWi1+RimUYgiCbHZegTMzSO4XL4VMP7rqb0GOqYBV/P70DrJskjthfQX0YcGwgqDqWr8469iKSPlinexk/xDIW8W/dKSCHJVzcYar6euGzf0zkJ91/BPy9IHTjGu6CAxm6NqcgOQ/iE+EClS214NJMiapiwwUFTWvfTEKIe8viS+6eEzSVXnkZrDlg8uyaVK8zYF8g/GKw3XyZx9fAM0zpwBbV8rHsmLYQwB0EOxAkvAYt03yh/jIgOdOXiSVMymDo1smQ=='

s3 = boto3.client('s3',
                    aws_access_key_id=my_aws_access_key_id,
                    aws_secret_access_key=my_aws_secret_access_key,
                    aws_session_token=my_aws_session_token)

# Load data from S3 and print content
def load_data_from_s3():
    local_file = 'heart_disease.csv'
    s3.download_file(S3_BUCKET, S3_FILE, local_file)
    df = pd.read_csv(local_file)
    #print(df.head())  # Print first few rows of the dataset
    return df

# Clean data by retaining only the specified columns
def clean_data(df):
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
    return df_cleaned


def impute_smoking_1(url, df_cleaned):
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
        def get_age_group(age):
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
        def impute_smoking_rate(row):
            if pd.isna(row['smoke_source_1']):  # If the value is missing
                age_group = get_age_group(int(row['age']))  # Map age to age group
                if age_group and age_group in smoking_rate_by_age:
                    return smoking_rate_by_age[age_group]  # Get the corresponding smoking rate
                else:
                    return None  # If no corresponding smoking rate is found
            return row['smoke_source_1']  # If not missing, return the original value

        # Apply the function to the dataframe to impute missing values
        df_cleaned['smoke_source_1'] = df_cleaned.apply(impute_smoking_rate, axis=1)

        # Return the dataframe with the new imputed values
        return df_cleaned

def impute_smoking_2(df_cleaned):
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
    def get_age_group(age):
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
    def impute_smoking_rate(row):
        if pd.isna(row['smoke_source_2']):  # If the value is missing
            age_group = get_age_group(int(row['age']))  # Map age to age group
            if age_group:
                # Impute for females
                if row['sex'] == 0:
                    return smoking_rate_by_age[age_group]
                # Impute for males
                elif row['sex'] == 1:
                    return smoking_rate_by_age[age_group] * (smoking_rate_male / smoking_rate_female)
        return row['smoke_source_2']  # If not missing, return the original value

    # Apply the function to the dataframe to impute missing values
    df_cleaned['smoke_source_2'] = df_cleaned.apply(impute_smoking_rate, axis=1)

    # Return the dataframe with the new imputed values
    return df_cleaned


def train_heart_disease_model(df):
    """
    Function to train a heart disease prediction model on the provided DataFrame.
    It performs data preprocessing, splitting, and evaluates multiple classification models.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame containing the heart disease data with 'target' as the label column.
        
    Returns:
        final_model: The best trained model based on cross-validation results.
        classification_report: A string containing classification report of the final model on the test set.
        roc_auc_score: The ROC-AUC score of the final model on the test set.
    """
    
    # Step 1: Extract features and target from the DataFrame
    X = df.drop(columns=['target', 'smoke'])  # Features + smoke column removed because there are NaN values and we will use two new smoke columns instead
    X = X.apply(pd.to_numeric, errors='coerce')
    y = df['target']  # Target variable
    
    # Step 2: Split the data into 90% training and 10% test with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    # Step 3: Train multiple classifiers and evaluate with 5-fold cross-validation
    models = {
        "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "XGBoost": XGBClassifier(enable_categorical=True, eval_metric='logloss', random_state=42)
    }

    # Initialize 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Prepare a dictionary to store cross-validation scores
    cv_results = {}
    best_models = {}

    # Loop through the models and perform cross-validation
    for model_name, model in models.items():
        # Hyperparameter tuning for Logistic Regression
        if model_name == "Logistic Regression":
            param_grid = {'C': [0.1, 1, 10]}  # Hyperparameter tuning for Logistic Regression
            grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy')
        
        # Hyperparameter tuning for Random Forest
        elif model_name == "Random Forest":
            param_grid = {'max_depth': [10, 20, None], 'n_estimators': [50, 100, 200]}  # Hyperparameter tuning for Random Forest
            grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy')
        
        elif model_name == "Support Vector Machine":
            param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

        elif model_name == "XGBoost":  # ADD THIS BLOCK
            param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}

        # Run GridSearchCV to find the best parameters
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Get the best model and its cross-validation scores
        best_model = grid_search.best_estimator_
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy')

        # Store results
        best_models[model_name] = best_model
        cv_results[model_name] = cv_scores.mean()

        # Store the cross-validation scores for the model
        cv_results[model_name] = cv_scores
        print(f"{model_name} - Best Params: {grid_search.best_params_}")
        print(f"{model_name} - Cross-validation scores: {cv_scores}")
        print(f"{model_name} - Average cross-validation score: {cv_scores.mean()}\n")

    # Step 4: Evaluate the best model on the test data and select the final model
    ###final_model_name = max(cv_results, key=lambda k: cv_results[k].mean())  # Choose model with highest cross-validation score
    ###final_model = grid_search.best_estimator_  # Best model from grid search
    final_model_name = max(cv_results, key=lambda k: cv_results[k].mean())
    final_model = best_models[final_model_name]

    # Evaluate the final model on the test set
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    # Calculate performance metrics
    print(f"Final model: {final_model_name}")
    classification_rep = classification_report(y_test, y_pred)
    print("Classification Report (Test Data):\n", classification_rep)
    roc_auc = roc_auc_score(y_test, final_model.predict_proba(X_test)[:, 1])
    print("ROC-AUC Score (Test Data):", roc_auc)
    
    # Return the final model and evaluation metrics
    return final_model, classification_rep, roc_auc


def main():
    df = load_data_from_s3()
    df_cleaned = clean_data(df)

    url = 'https://www.abs.gov.au/statistics/health/health-conditions-and-risks/smoking-and-vaping/latest-release'
    smoking_data_one = impute_smoking_1(url, df_cleaned)
    #print(smoking_data_one)
    smoking_data_two = impute_smoking_2(df_cleaned)
    #print(smoking_data_two)
    train_heart_disease_model(df_cleaned)

main()