import os
import boto3
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, LinearSVC, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
# import pickle
from pyspark.sql import functions as F
import requests
from bs4 import BeautifulSoup
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from pyspark.ml import Pipeline

# AWS S3 Setup
S3_BUCKET = 'de300winter2025'
S3_SUBFOLDER = 'matt_saperstein'
S3_FILE = f'{S3_SUBFOLDER}/heart_disease.csv'

my_aws_access_key_id='ASIAYAAO5HRMPAZ4NWLP'
my_aws_secret_access_key='FjU0XSHQVBQcL1re6yP5DSbSa2XdKFpdQRDuP9kK'
my_aws_session_token='IQoJb3JpZ2luX2VjEBcaCXVzLWVhc3QtMiJHMEUCIE6B8SrZbkkBfS/IxgyxoChhyTIXH/7CPPfmWmVkyoC1AiEArhbTRq98LKOtePRAHz7h3FGVCyzDXhF97gr2Zgmkhhwq6wIIUBAAGgw1NDk3ODcwOTAwMDgiDAw3ZkGL+ygm+ulaYCrIAkP3c19jkQuPxsL74+uWteRCoyRmJzTwhupyAEroI2WGM1WBV1Klq9oUvKhyQDdCKOHlhMCAeWJgESotYP7K/aDbE05NKpn/GxriWnGk8RbejnCjvybLtITjjG3iJIPgbxXSITg7gJ9mzMtX4rzRfSc0+/ACVVItx7Yfp+arQLjosuF8IZ3qIR3QoRakHYt4BqcobNYfxGBzw7jaSx22lvevoU3OpZeeESNxyxG5bwQVSNL9ZaDfCNLHG6yh5ymFTntjEZsNVKvYdSSksseNdJwOGP2ITw3s3ZmIs1mSGx2em3JtLTQymsXfEYkigElMdz6e6cMXKL9dkRFDu820hvEffshejUFHhZRQlKVfJ3pQpPEH8Pwz0oWLA0Qbn8oW95y/cNreDHbQp9K9xbKXqqiEsxh/l3aIcSDPfFWFTrbHSqj8ajSP4HAwppr5vQY6pwGDSl10GuP0qMy/vnwPTBQ1zIv03xbapUfBiZMqhVjc89l2hoc1HkboyPqXXmMpFkhthCOwPK6LKTNmXqzNYQLbIORY8htQdljQAWGGXiXLTBFy7AovSOsexQanCaEuSatO5sG5AcLFXPh+uy+Zw/xENkr2DqhBcEZesHbHor/r34owd4IW4XB+xhcIP432jCeD+cfrShxtokgL3lD1sWU5Y1BGoE1MYw=='

s3 = boto3.client('s3',
                    aws_access_key_id=my_aws_access_key_id,
                    aws_secret_access_key=my_aws_secret_access_key,
                    aws_session_token=my_aws_session_token)

# Create Spark session
spark = SparkSession.builder.appName("HeartDiseasePrediction").getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Load data from S3 and return as Spark DataFrame
def load_data_from_s3():
    local_file = 'heart_disease.csv'
    s3.download_file(S3_BUCKET, S3_FILE, local_file)
    df = spark.read.csv(local_file, header=True, inferSchema=True)
    # Show the first few rows
    # df.show(5)
    return df


def clean_data(df):
    # List of columns to retain
    columns_to_retain = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'smoke', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 'exang', 'oldpeak', 'slope', 'target']
    
    # Select only the columns that are in the list
    df_cleaned = df.select(*columns_to_retain)
    
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

    # Show the first 5 rows
    df_cleaned.show(5)

    # Return the dataframe with the new imputed values
    return df_cleaned

def train_heart_disease_model(df):
    """
    Function to train a heart disease prediction model on the provided Spark DataFrame.
    It performs data preprocessing, splitting, and evaluates multiple classification models using Spark MLlib.
    
    Parameters:
        df (pyspark.sql.DataFrame): Input DataFrame containing the heart disease data with 'target' as the label column.
        
    Returns:
        final_model: The best trained model based on cross-validation results.
        classification_report: A string containing classification report of the final model on the test set.
        roc_auc_score: The ROC-AUC score of the final model on the test set.
    """
    
    # Initialize Spark session
    # spark = SparkSession.builder.appName("HeartDiseaseModel").getOrCreate()

    # Step 1: Preprocess data
    # Drop 'smoke' column as we will handle smoke as two new columns (you may need to impute missing values here if necessary)
    df = df.drop("smoke")
    
    # Step 2: Convert features to numeric and create a feature vector
    # Convert columns to numeric types
    feature_columns = [col for col in df.columns if col != 'target']
    df = df.select(*[col(c).cast('double') for c in feature_columns] + ['target'])

    # Step 3: Assemble features into a feature vector using VectorAssembler
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df = assembler.transform(df)

    # Step 4: Split the data into 90% training and 10% test with stratification
    train_df, test_df = df.randomSplit([0.9, 0.1], seed=42)
    
    # Step 5: Set up classifiers
    lr = LogisticRegression(labelCol="target", featuresCol="features", maxIter=10000)
    rf = RandomForestClassifier(labelCol="target", featuresCol="features", numTrees=100)
    gbt = GBTClassifier(labelCol="target", featuresCol="features", maxIter=100)
    
    # Step 6: Define the parameter grids for hyperparameter tuning
    lr_param_grid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 1.0]).build()
    rf_param_grid = ParamGridBuilder().addGrid(rf.maxDepth, [10, 20]).addGrid(rf.numTrees, [50, 100, 200]).build()
    gbt_param_grid = ParamGridBuilder().addGrid(gbt.maxDepth, [5, 10]).addGrid(gbt.maxIter, [10, 20]).build()
    
    # Step 7: Set up CrossValidator for hyperparameter tuning and model selection
    evaluator = BinaryClassificationEvaluator(labelCol="target", metricName="areaUnderROC")
    
    lr_cv = CrossValidator(estimator=lr, estimatorParamMaps=lr_param_grid, evaluator=evaluator, numFolds=5)
    rf_cv = CrossValidator(estimator=rf, estimatorParamMaps=rf_param_grid, evaluator=evaluator, numFolds=5)
    gbt_cv = CrossValidator(estimator=gbt, estimatorParamMaps=gbt_param_grid, evaluator=evaluator, numFolds=5)
    
    # Step 8: Train the models using cross-validation
    lr_model = lr_cv.fit(train_df)
    rf_model = rf_cv.fit(train_df)
    gbt_model = gbt_cv.fit(train_df)

    # Step 9: Evaluate models on the test set
    lr_pred = lr_model.transform(test_df)
    rf_pred = rf_model.transform(test_df)
    gbt_pred = gbt_model.transform(test_df)
    
    lr_auc = evaluator.evaluate(lr_pred)
    rf_auc = evaluator.evaluate(rf_pred)
    gbt_auc = evaluator.evaluate(gbt_pred)

    print(f"Logistic Regression AUC: {lr_auc}")
    print(f"Random Forest AUC: {rf_auc}")
    print(f"Gradient Boosted Tree AUC: {gbt_auc}")
    
    # Step 10: Select the best model based on AUC
    auc_scores = {'Logistic Regression': lr_auc, 'Random Forest': rf_auc, 'GBT': gbt_auc}
    best_model_name = max(auc_scores, key=auc_scores.get)
    print(f"Best Model: {best_model_name}")
    
    # Step 11: Final model training and evaluation
    final_model = None
    if best_model_name == "Logistic Regression":
        final_model = lr_model.bestModel
    elif best_model_name == "Random Forest":
        final_model = rf_model.bestModel
    elif best_model_name == "GBT":
        final_model = gbt_model.bestModel
    
    final_pred = final_model.transform(test_df)
    final_auc = evaluator.evaluate(final_pred)
    print(f"Final Model AUC: {final_auc}")
    
    # Return the final model, classification report, and ROC-AUC score
    classification_report = f"ROC-AUC Score: {final_auc}"
    return final_model, classification_report, final_auc
    
def main():
    df = load_data_from_s3()
    cleaned_df = clean_data(df)
    url = 'https://www.abs.gov.au/statistics/health/health-conditions-and-risks/smoking-and-vaping/latest-release'
    smoking_data_one = impute_smoking_1(url, cleaned_df)
    smoking_data_two = impute_smoking_2(cleaned_df)
    train_heart_disease_model(cleaned_df)

main()
