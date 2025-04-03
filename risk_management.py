import numpy as np   # Calculate multi-arrays / matrices 
import pandas as pd  # Data manipulation (dataframe)
import os, json      # os: function interact with os


def load_csv(file_name):
    """
    Load a csv file 
    Return pandas.DataFrame
    """
    file_path = os.path.join('dataset', file_name)
    # Read the CSV file
    data = pd.read_csv(file_path)
    print(f"Successfully loaded {file_name}, shape: {data.shape}")
    return data

def analyze_missing_values(df):
    """
    Analyze missing values in a dataframe
    Returns: datagrame with missing value statistics
    """
    print("\n-- MISSING VALUES ANALYSIS --\n")
    
    # Calculate missing values statistics 
    missing_stats = pd.DataFrame({
        'missing_count': df.isnull().sum(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    
    # Add data types for reference
    missing_stats['dtype'] = df.dtypes.values
    
    # Categorize features by missing percentage
    bins = [0, 0.001, 5, 25, 100] # Define the bins(intervals)
    labels = ["none", "low", "medium", "high"] # The labels for the bins
    missing_stats['missing_category'] = pd.cut(
        missing_stats['missing_percentage'],
        bins=bins,
        labels=labels
    ) # pd.cut divides a df column into the bins and assins labels to each bin

    print(f"Total features: {len(missing_stats)}")
    print(f"Features with no missing (0%): {missing_stats['missing_category'].isna().sum()}")
    print(f"Features with low missing (< 5%): {sum(missing_stats['missing_category'] == 'low')}")
    print(f"Features with medium missing (5-25%): {sum(missing_stats['missing_category'] == 'medium')}")
    print(f"Features with high missing (> 25%): {sum(missing_stats['missing_category'] == 'high')}\n")

    return missing_stats

def identify_feature_types(df):
    """
    Identify neumerical and categorical features
    """
    return

def data_Preprocessing():
    """
    Preprocess data
    - Handle categorical features 
    - Encode categorical features
    - Transform numerical features
    """
    
    return  

def data_exploratory(df):
    """
    Performs exploratory data analysis on the credit default risk datasets
    """
    print("Starting Exploring Datasets...")

    
    # Calculate target(label: 0/1) distribution 
    target_counts = df['TARGET'].value_counts()
    total_records = len(df)
    target_percentages = target_counts / total_records * 100
    print("\n-- TARGET (LABELS) DISTRIBUTION --")
    print("TARGET(0): no repayments difficulties")
    print("TARGET(1): has repayments difficulties")
    print("\nCounts:")
    print(target_counts)
    print("\nPercentages:")
    print(f"TARGET(0): {target_percentages[0]:.2f}%")
    print(f"TARGET(1): {target_percentages[1]:.2f}%") 
    
    # Handle Missing Values in each columns in application_tain.csv
    missing_stats = analyze_missing_values(df)
    
    # Identify Feature Types 
    numerical_features, categorical_features = identify_feature_types(df)
    
    return missing_stats, numerical_features, categorical_features
    

if __name__ == "__main__":
    csv_files = [
        "application_train.csv",
        "application_test.csv",
        "bureau.csv",
        "bureau_balance.csv",
        "credit_card_balance.csv",
        "installments_payments.csv",
        "POS_CASH_balance.csv",
        "previous_application.csv"
    ]
    
    # Identify class imbalance in the training dataset
    df = load_csv(csv_files[0]) # application_train.csv
    
    # Detailed EDA - Exploratory Data Analysis
    missing_stats, numerical_features, categorical_features = data_exploratory(df)
    
    # Data Preprocessing 
    # processed_df = data_Preprocessing(missing_stats, numerical_features, categorical_features)
    
    # Feature Engineering ...