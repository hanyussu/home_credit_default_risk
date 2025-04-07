import numpy as np   # Calculate multi-arrays / matrices 
import pandas as pd  # Data manipulation (dataframe)
import os, json      # os: function interact with os
from sklearn.preprocessing import OneHotEncoder # Encoding Categorical Features

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
    Returns: lists of numerical and categorical features 
    """
    print("\n-- FEATURE TYPE IDENTICATION --\n")
    
    # Initialize lists to store feature types
    numerical_features = []
    categorical_features = []
    
    # Count unique data types in DataFrame
    print("Count of each data type:")
    print(df.dtypes.value_counts())
    
    for col in df.columns:
        # skip id and label
        if col in ['SK_ID_CURR', 'TARGET']:
            continue
        # Rule 1: Object type -> categorial 
        if df[col].dtype == "object":
            categorical_features.append(col)
            
        # Rule 2: Binary or very few unique values -> likely categorical
        elif df[col].nunique() < 10:
            categorical_features.append(col)
        # Rule 3: Numerical Values
        else:
            numerical_features.append(col)
            
    print(f"Numerical features: {len(numerical_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    return numerical_features, categorical_features 

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
  
def analyze_categorical_cardinality(df, features):
    """
    Analyze the cardinality (number of unique values) of categorical features.
    
    Encoding Strategy: 
    - Low cardinality (<=10): 'one_hot'
    - High cardinality (>10): 'target'
    """
    encoder_choice = {}
    low_card, high_card = 0, 0
    for col in [col for col in features if col in df.columns]:
        # Count unique values
        n_unique = df[col].nunique()
        if n_unique > 10:
            encoder_choice[col] = 'high' 
            high_card += 1
        else: 
            encoder_choice[col] = 'low'
            low_card += 1
    print(f"Low Categorical Cardinality Feature Counts: {low_card}")        
    print(f"High Categorical Cardinality Feature Counts: {high_card}\n")
    return encoder_choice 
  
def data_Preprocessing(df, missing_stats, numerical_features, categorical_features):
    """
    Preprocess data
    - Drop features with excessive missing values (>75%)
    - Handle missing values in remaining features
    - Encode categorical features
    - Transform numerical features
    
    Returns: processed dataframe
    """
    print("\n-- Data Preprocessing --\n")
    # Create a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # 1. Drop features with excessive missing values
    high_missing_features = missing_stats[missing_stats["missing_category"] == "high"].index.tolist()
    processed_df = processed_df.drop(columns=high_missing_features)
    print(f"Dropped {len(high_missing_features)} high missing features")
    
    # 2. Handle missing values in reaming features (Data Imputation)
    # Numerical Features: imputes with median 
    for col in [col for col in numerical_features if col in processed_df.columns]:
        median_value = processed_df[col].median()
        processed_df[col] = processed_df[col].fillna(median_value)
        # print(f"Imputed missing values in '{col}' with median: {median_value}")
    
    # Categorical Features: imputes with most frequest value
    for col in [col for col in categorical_features if col in processed_df.columns]:
        most_frequent = processed_df[col].mode()[0]
        processed_df[col] = processed_df[col].fillna(most_frequent)
    
    # 3. Encode Categorical Features 
    # Categorical Features Cardinality Analysis ---> decide which encoder should be used
    cardinality_stats = analyze_categorical_cardinality(processed_df, categorical_features)
    
    # Separate features by encoding type
    low_cardinality_features = [col for col in categorical_features if col in processed_df.columns 
                      and cardinality_stats.get(col) == 'low']
    high_cardinality_features = [col for col in categorical_features if col in processed_df.columns 
                      and cardinality_stats.get(col) == 'high']
    
    # Drop these features from the dataframe
    if high_cardinality_features:
        processed_df = processed_df.drop(columns=high_cardinality_features)
        print(f"Dropped {len(high_cardinality_features)} high cardinality features")
    print(f"Original shape: {df.shape}, New shape after dropping: {processed_df.shape}")
    
    # Use One-Hot encoding for categorical features
    
    
    
    
    # 4. Transform Numerical Features
    
    return 

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
    processed_df = data_Preprocessing(df, missing_stats, numerical_features, categorical_features)
    
    # Feature Engineering ...