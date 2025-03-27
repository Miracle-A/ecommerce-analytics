# src/data_preprocessing.py
import pandas as pd
import numpy as np
from datetime import datetime

def clean_retail_data(df):
    """
    Clean and preprocess the retail dataset.
    """
    print("Cleaning and preprocessing data...")
    
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Remove rows with missing values
    df_clean = df_clean.dropna()
    
    # Convert InvoiceDate to datetime
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    
    # Extract date components
    df_clean['Year'] = df_clean['InvoiceDate'].dt.year
    df_clean['Month'] = df_clean['InvoiceDate'].dt.month
    df_clean['Day'] = df_clean['InvoiceDate'].dt.day
    df_clean['DayOfWeek'] = df_clean['InvoiceDate'].dt.dayofweek
    df_clean['Hour'] = df_clean['InvoiceDate'].dt.hour
    
    # Filter out rows with quantity less than or equal to 0
    df_clean = df_clean[df_clean['Quantity'] > 0]
    
    # Filter out rows with unit price less than or equal to 0
    df_clean = df_clean[df_clean['UnitPrice'] > 0]
    
    # Create TotalPrice column
    df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']
    
    # Convert InvoiceNo to string to handle numeric invoice numbers
    df_clean['InvoiceNo'] = df_clean['InvoiceNo'].astype(str)
    
    # Flag returns (invoices starting with 'C')
    df_clean['IsReturn'] = df_clean['InvoiceNo'].str.startswith('C').astype(int)
    
    # Filter out returns for main analysis (but keep them in a separate DataFrame)
    returns = df_clean[df_clean['IsReturn'] == 1]
    df_clean = df_clean[df_clean['IsReturn'] == 0]
    
    print(f"Data cleaning complete. Shape after cleaning: {df_clean.shape}")
    
    return df_clean, returns

def prepare_customer_features(df_clean):
    """
    Prepare customer-level features for segmentation.
    """
    print("Preparing customer-level features...")
    
    # Get the latest date in the dataset to calculate recency
    latest_date = df_clean['InvoiceDate'].max()
    
    # Group by customer and calculate RFM metrics
    customer_features = df_clean.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (latest_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',  # Frequency
        'TotalPrice': 'sum',  # Monetary
        'Quantity': 'sum',  # Total items purchased
        'StockCode': 'nunique'  # Number of unique products purchased
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalPrice': 'Monetary',
        'Quantity': 'TotalItems',
        'StockCode': 'UniqueProducts'
    })
    
    # Calculate additional features
    # Average purchase value
    customer_features['AvgPurchaseValue'] = customer_features['Monetary'] / customer_features['Frequency']
    
    # Average items per purchase
    customer_features['AvgItemsPerPurchase'] = customer_features['TotalItems'] / customer_features['Frequency']
    
    # Save the customer features
    customer_features.to_csv("data/customer_features.csv")
    print(f"Customer features saved to data/customer_features.csv")
    
    return customer_features

if __name__ == "__main__":
    # Load the raw data
    print("Loading raw data...")
    try:
        df = pd.read_csv("data/online_retail_raw.csv")
    except FileNotFoundError:
        print("Raw data not found. Please run data_acquisition.py first.")
        exit(1)
    
    # Clean the data
    df_clean, returns = clean_retail_data(df)
    
    # Save cleaned data
    df_clean.to_csv("data/online_retail_clean.csv", index=False)
    returns.to_csv("data/online_retail_returns.csv", index=False)
    
    # Prepare customer features for segmentation
    customer_features = prepare_customer_features(df_clean)
    
    print("Data preprocessing completed successfully!")