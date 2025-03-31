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
    
    Returns: Dictionary containing:
        - loaded datagrames
        - basic info for each dataframe
        - missing value stats
        - descriptive statistics
        - data types
    """
    print("Starting Exploring Datasets...")
    # csv_files = [
    #     "application_test.csv",
    #     "application_train.csv",
    #     "bureau.csv",
    #     "bureau_balance.csv",
    #     "credit_card_balance.csv",
    #     "installments_payments.csv",
    #     "POS_CASH_balance.csv",
    #     "previous_application.csv"
    # ]
    
    csv_files = [
        "bureau.csv"
    ]
    
    # Dictionary to store all exploratory information
    results = {
        "dataframes": {},
        "stats": {}
    }
    
    for filename in csv_files:
        df = load_csv(filename)
        results["dataframes"][filename] = df
        
        # Create entry for this dataframe's stats
        df_stats = {}
        
        # Basic info 
        df_stats["shape"] = df.shape
        df_stats["columns"] = list(df.columns) 
        
        # Missing Values
        missing = df.isnull().sum()
        # Create a dictionary of columns with missing values
        missing_by_column = {}
        for column, count in missing.items():
            if count > 0:  # Only include columns that have missing values
                missing_by_column[column] = {
                    'count': int(count),
                }

        df_stats["missing"] = {
             'total': int(missing.sum()),
             'columns': missing_by_column
        }
         
        # Descriptive Stats
        categorical_stats = None
        # if the categorical data exists
        # df.select_dtypes(include=['object']) selects columns with categorical data type
        # and returns a dataframe (rows, cols)
        if df.select_dtypes(include=['object']).shape[1] > 0:
            categorical_stats = df.describe(include=['object']).to_dict()
        
        df_stats["describe"] = {
            "numeric": df.describe().to_dict(),
            "categorical": categorical_stats
        }
    
        # Store dataframe's stats into results
        results["stats"][filename] = df_stats   

    return results

if __name__ == "__main__":
    results = data_exploratory()
    # Create a better formatted outputs using JSON
    formatted_results = json.dumps(results["stats"], indent=4)
    print(formatted_results)