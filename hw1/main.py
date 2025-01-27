import pandas as pd
from sqlalchemy import create_engine, text
import os
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import seaborn as sns

# Database connection settings
DB_USER = "user"
DB_PASSWORD = "password"
DB_HOST = "postgres"
DB_PORT = "5432"
DB_NAME = "mydatabase"

# File paths
CSV_FILE = "data/data.csv"

# Create a database engine
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

def create_initial_table_from_csv(csv_file, table_name):
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Create table schema based on the DataFrame (no manipulation)
    print(f"Creating initial table {table_name}...")
    df.head(0).to_sql(table_name, engine, if_exists="replace", index=False)
    
    # Load data into the table
    print(f"Loading initial data into {table_name}...")
    df.to_sql(table_name, engine, if_exists="append", index=False)
    print(f"Initial data successfully loaded into {table_name}!")

def create_final_table_from_csv(csv_file, table_name):
    # Read CSV file
    df = pd.read_csv(csv_file)

    # Convert DAYS_EMPLOYED and DAYS_BIRTH to positive values
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].abs()  # Use the absolute value
    df['DAYS_BIRTH'] = df['DAYS_BIRTH'].abs()

    print("Missing values before imputation:")
    print(df.isnull().sum())
    
    # Impute missing values
    df = impute_missing_values(df)

    print("Missing values after imputation:")
    print(df.isnull().sum())

    df = identify_and_handle_outliers(df)

    compute_statistics(df)

    perform_feature_transformations(df)

    generate_plots(df)

    # Create table schema based on the DataFrame
    print(f"Creating table {table_name}...")
    df.head(0).to_sql(table_name, engine, if_exists="replace", index=False)

    # Load data into the table
    print(f"Loading data into {table_name}...")
    df.to_sql(table_name, engine, if_exists="append", index=False)
    print(f"Data successfully loaded into {table_name}!")
    
    # Analyze missing values after imputation (should now be all zero)
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    missing_info = pd.DataFrame({
        'Missing Values': missing_data,
        'Percentage': missing_percentage
    })
    print("Missing Values Analysis:")
    print(missing_info[missing_info['Missing Values'] > 0])

def impute_missing_values(df):
    # Impute categorical column 'HOUSETYPE_MODE' with mode (most frequent value)
    df['HOUSETYPE_MODE'] = df['HOUSETYPE_MODE'].fillna(df['HOUSETYPE_MODE'].mode()[0])
    
    # Impute numerical columns with median (for normalized values between 0 and 1)
    df['EXT_SOURCE_1'] = df['EXT_SOURCE_1'].fillna(df['EXT_SOURCE_1'].median())
    df['EXT_SOURCE_2'] = df['EXT_SOURCE_2'].fillna(df['EXT_SOURCE_2'].median())
    df['EXT_SOURCE_3'] = df['EXT_SOURCE_3'].fillna(df['EXT_SOURCE_3'].median())
    df['TOTALAREA_MODE'] = df['TOTALAREA_MODE'].fillna(df['TOTALAREA_MODE'].median())
    
    # Impute 'AMT_REQ_CREDIT_BUREAU_YEAR' with 0
    df['AMT_REQ_CREDIT_BUREAU_YEAR'] = df['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(0)
    
    return df

def identify_and_handle_outliers(df):
    # Numerical columns
    numerical_cols = ['CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'AMT_INCOME_TOTAL', 'AMT_CREDIT',
                      'TOTALAREA_MODE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 
                      'DAYS_EMPLOYED', 'DAYS_BIRTH', 'AMT_REQ_CREDIT_BUREAU_YEAR']

    print(f"Number of rows where 'TARGET' is neither 0 nor 1: {df[~df['TARGET'].isin([0, 1])].shape[0]}")
    
    for col in numerical_cols:
        # Calculate the IQR for the column
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Calculate lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identifying outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"Outliers detected in {col}:")
        print(outliers[[col]])

        # Handle outliers by capping them to the upper and lower bounds
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    return df

def compute_statistics(df):
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)  # Adjust the width
    pd.set_option('display.max_colwidth', None)  # Display full column content
    # Numerical columns
    numerical_cols = ['CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'AMT_INCOME_TOTAL', 'AMT_CREDIT',
                      'TOTALAREA_MODE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 
                      'DAYS_EMPLOYED', 'DAYS_BIRTH', 'AMT_REQ_CREDIT_BUREAU_YEAR']
    
    # Calculate and print various statistical measures for numerical columns
    statistical_measures = df[numerical_cols].describe().T  # Transpose for better readability

    # Add additional statistical measures
    statistical_measures['skewness'] = df[numerical_cols].skew()
    statistical_measures['kurtosis'] = df[numerical_cols].kurtosis()

    # Print the statistics
    print("Statistical Measures for Numerical Columns:")
    print(statistical_measures[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis']])

    categorical_cols = ['TARGET', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
                    'NAME_INCOME_TYPE', 'HOUSETYPE_MODE', 'NAME_EDUCATION_TYPE', 
                    'NAME_HOUSING_TYPE', 'NAME_FAMILY_STATUS']

    # Calculate mode and value counts for each categorical column
    for col in categorical_cols:
        print(f"Statistical Measures for Categorical Column: {col}")
        print(f"Mode: {df[col].mode()[0]}")
        print(f"Value Counts:\n{df[col].value_counts()}")
        print("\n")

def class_proportion_encode(df, columns, target_column):
    for column in columns:
        # Calculate the proportion of the target being 1 for each category in the column
        class_proportions = df.groupby(column)[target_column].mean()  # Proportion of 1's in target
        
        # Replace the categorical values with the corresponding class proportions
        df[column] = df[column].map(class_proportions)
    
    return df

def perform_feature_transformations(df):
    # Target column and binary or ordinal categorical columns to apply target encoding
    target_column = 'TARGET'
    target_encoding_columns = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']

    # Apply target-based label encoding (since we assume all target_encoding_columns are categorical)
    for col in target_encoding_columns:
        # Calculate the mean of the target for each category in the feature column
        category_means = df.groupby(col)[target_column].mean()
        # Map the category means back to the column, replacing the original values
        df[col] = df[col].map(category_means)
    print("Target-based label encoding applied to categorical columns.")
    
    # Nominal categorical columns for one-hot encoding
    one_hot_encoding_columns = ['NAME_INCOME_TYPE', 'HOUSETYPE_MODE', 'NAME_EDUCATION_TYPE', 'NAME_HOUSING_TYPE', 'NAME_FAMILY_STATUS']

    # Apply one-hot encoding
    df = pd.get_dummies(df, columns=one_hot_encoding_columns, drop_first=True)
    print("One-hot encoding applied to nominal categorical columns.")
    
    return df

def generate_plots(df):
    # Set the style for the plots
    sns.set(style="whitegrid")

    # Box plots for certain numerical columns
    numerical_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT',
                      'TOTALAREA_MODE', 'AMT_REQ_CREDIT_BUREAU_YEAR']

    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[col])
        plt.title(f"Box Plot for {col}")
        plt.show()
        plt.savefig(f"Box Plot for {col}.png")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], orient="h")
    plt.title("Box Plot for EXT_SOURCE Values")
    plt.tight_layout()
    plt.savefig("Box Plot for EXT_SOURCE Values")
    plt.show()

    plt.figure(figsize=(10, 6))   
    sns.boxplot(data=df[['AMT_CREDIT', 'AMT_INCOME_TOTAL']], orient="h")
    plt.title("Box Plot for Credit and Income  Values")
    plt.tight_layout()
    plt.savefig("Box Plot for Credit and Income Values.png")
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.boxplot(data=df[['CNT_CHILDREN', 'CNT_FAM_MEMBERS']], orient="h")
    plt.title("Box Plot for Number of Children and Family Members")
    plt.tight_layout()
    plt.xlabel("Number of People")
    plt.tight_layout() 
    plt.savefig("Box Plot for Number of Children and Family Members.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[['DAYS_EMPLOYED', 'DAYS_BIRTH']], orient="h")
    plt.title("Box Plot for DAYS_EMPLOYED and DAYS_BIRTH")
    plt.xlabel("Days (positive values)")
    plt.tight_layout()
    plt.savefig("Box Plot for DAYS_EMPLOYED and DAYS_BIRTH.png")
    plt.show()

    # Scatter Plot: EXT_SOURCE_2 vs. EXT_SOURCE_3
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['EXT_SOURCE_2'], y=df['EXT_SOURCE_3'])
    plt.title("Scatter Plot of EXT_SOURCE_2 vs EXT_SOURCE_3")
    plt.xlabel('EXT_SOURCE_2')
    plt.ylabel('EXT_SOURCE_3')
    plt.show()
    plt.savefig("Scatter Plot of EXT_SOURCE_2 vs EXT_SOURCE_3")

    # Scatter Plot: Income vs. Credit Amount
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="AMT_INCOME_TOTAL", y="AMT_CREDIT", data=df)
    plt.title("Income vs. Credit Amount")
    plt.savefig("Income_vs_Credit_Amount.png")
    plt.close()  # Close the plot to avoid overlap

    # Scatter Plot: Age (DAYS_BIRTH) vs. Credit Amount
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="DAYS_BIRTH", y="AMT_CREDIT", data=df)
    plt.title("Age vs. Credit Amount")
    plt.savefig("Age_vs_Credit_Amount.png")
    plt.close()

    # Scatter Plot: Employment Duration vs. Age (DAYS_EMPLOYED vs. DAYS_BIRTH)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="DAYS_EMPLOYED", y="DAYS_BIRTH", data=df)
    plt.title("Employment Duration vs. Age")
    plt.savefig("Employment_Duration_vs_Age.png")
    plt.close()

def main():
    # Ensure data directory exists
    if not os.path.exists(CSV_FILE):
        print(f"Error: File {CSV_FILE} not found.")
        return

    # Create and populate the initial table
    create_initial_table_from_csv(CSV_FILE, "initial_data_table")

    # Create and populate the final table
    create_final_table_from_csv(CSV_FILE, "final_data_table")

    # Test the data retrieval for initial data table
    with engine.connect() as connection:
        result = connection.execute(text("SELECT * FROM initial_data_table LIMIT 5;"))
        print("\nSample data from the initial table database:")
        for row in result:
            print(row)

    # Test the data retrieval for final data table
    with engine.connect() as connection:
        result = connection.execute(text("SELECT * FROM final_data_table LIMIT 5;"))
        print("\nSample data from the final table database:")
        for row in result:
            print(row)

if __name__ == "__main__":
    main()

