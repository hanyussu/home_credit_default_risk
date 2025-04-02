import numpy as np
import pandas as pd
import os, json


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

def data_exploratory():
    """
    Performs exploratory data analysis on the credit default risk datasets
    """
    print("Starting Exploring Datasets...")
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
    application_train_df = load_csv(csv_files[0])
    
    # Calculate target(label: 0/1) distribution 
    target_counts = application_train_df['TARGET'].value_counts()
    total_records = len(application_train_df)
    target_percentages = target_counts / total_records * 100
    print("\n-- TARGET (LABELS) DISTRIBUTION --")
    print("TARGET(0): no repayments difficulties")
    print("TARGET(1): has repayments difficulties")
    print("\nCounts:")
    print(target_counts)
    print("\nPercentages:")
    print(f"TARGET(0): {target_percentages[0]:.2f}%")
    print(f"TARGET(1): {target_percentages[1]:.2f}%")
    
    # Feature Correlations with Target value 
    
    
    
        
    return 
    

if __name__ == "__main__":
    # Detailed EDA - Exploratory Data Analysis
    data_exploratory()