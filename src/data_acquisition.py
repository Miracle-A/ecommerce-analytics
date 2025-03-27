# src/data_acquisition.py
import pandas as pd
import numpy as np
import os

def download_data():
    """
    Download the UCI Online Retail dataset and save it to the data folder.
    """
    # URL for the UCI Online Retail dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Download and save the data
    print("Downloading UCI Online Retail dataset...")
    df = pd.read_excel(url)
    
    # Save to CSV for easier handling
    df.to_csv("data/online_retail_raw.csv", index=False)
    print(f"Data downloaded and saved to data/online_retail_raw.csv")
    
    return df

if __name__ == "__main__":
    # Download the data
    retail_data = download_data()
    
    # Display basic information
    print("\nDataset Information:")
    print(f"Shape: {retail_data.shape}")
    print("\nColumn Names:")
    print(retail_data.columns.tolist())
    print("\nData Types:")
    print(retail_data.dtypes)
    print("\nSample Data:")
    print(retail_data.head())