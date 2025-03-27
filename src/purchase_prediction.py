# src/purchase_prediction.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
import pickle
import os

def prepare_purchase_prediction_data(df_clean):
    """
    Prepare data for purchase prediction modeling.
    """
    print("Preparing purchase prediction data...")
    
    # Group transactions by customer and date
    purchase_data = df_clean.groupby(['CustomerID', 'InvoiceDate']).agg({
        'InvoiceNo': 'first',
        'Quantity': 'sum',
        'TotalPrice': 'sum',
        'StockCode': 'nunique'
    }).reset_index()
    
    # Sort by customer and date
    purchase_data = purchase_data.sort_values(['CustomerID', 'InvoiceDate'])
    
    # Calculate days since previous purchase
    purchase_data['PrevPurchaseDate'] = purchase_data.groupby('CustomerID')['InvoiceDate'].shift(1)
    purchase_data['DaysSincePrevPurchase'] = (purchase_data['InvoiceDate'] - 
                                             purchase_data['PrevPurchaseDate']).dt.days
    
    # Calculate features from previous purchases
    purchase_data['PrevQuantity'] = purchase_data.groupby('CustomerID')['Quantity'].shift(1)
    purchase_data['PrevTotalPrice'] = purchase_data.groupby('CustomerID')['TotalPrice'].shift(1)
    purchase_data['PrevUniqueProducts'] = purchase_data.groupby('CustomerID')['StockCode'].shift(1)
    
    # Calculate average metrics from previous purchases
    customer_purchase_counts = purchase_data.groupby('CustomerID').size().reset_index(name='PurchaseCount')
    purchase_data = purchase_data.merge(customer_purchase_counts, on='CustomerID')
    
    # Remove first purchase for each customer (no previous purchase data)
    purchase_data = purchase_data.dropna(subset=['DaysSincePrevPurchase'])
    
    # Create target variable: will purchase within next 30 days?
    purchase_data['NextPurchaseDate'] = purchase_data.groupby('CustomerID')['InvoiceDate'].shift(-1)
    purchase_data['DaysUntilNextPurchase'] = (purchase_data['NextPurchaseDate'] - 
                                             purchase_data['InvoiceDate']).dt.days
    
    # Fill NaN values for customers' last purchases
    purchase_data['DaysUntilNextPurchase'] = purchase_data['DaysUntilNextPurchase'].fillna(999)
    
    # Create binary target: 1 if next purchase within 30 days, 0 otherwise
    purchase_data['WillPurchaseNextMonth'] = (purchase_data['DaysUntilNextPurchase'] <= 30).astype(int)
    
    # Create features for time-based patterns
    purchase_data['DayOfWeek'] = purchase_data['InvoiceDate'].dt.dayofweek
    purchase_data['Month'] = purchase_data['InvoiceDate'].dt.month
    purchase_data['IsWeekend'] = purchase_data['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Save the prepared data
    purchase_data.to_csv("data/purchase_prediction_data.csv", index=False)
    
    return purchase_data

def train_purchase_probability_model(purchase_data):
    """
    Train a model to predict the probability of purchase in the next 30 days.
    """
    print("Training purchase probability model...")
    
    # Select features and target
    features = [
        'DaysSincePrevPurchase', 'PrevQuantity', 'PrevTotalPrice', 'PrevUniqueProducts',
        'PurchaseCount', 'DayOfWeek', 'Month', 'IsWeekend'
    ]
    
    X = purchase_data[features]
    y = purchase_data['WillPurchaseNextMonth']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    with open("models/purchase_probability_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # Save feature importances
    feature_importances = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    feature_importances.to_csv("data/feature_importances.csv", index=False)
    
    return model, feature_importances

def train_purchase_value_model(purchase_data):
    """
    Train a model to predict the value of the next purchase.
    """
    print("Training purchase value model...")
    
    # Filter to only include customers with a next purchase
    next_purchase_data = purchase_data[purchase_data['DaysUntilNextPurchase'] < 999].copy()
    
    # Get the value of the next purchase
    next_purchase_values = next_purchase_data.groupby('CustomerID').apply(
        lambda x: x.sort_values('InvoiceDate')['TotalPrice'].values
    )
    
    next_purchase_data['NextPurchaseValue'] = None
    
    for customer_id, values in next_purchase_values.items():
        if len(values) > 1:
            customer_indices = next_purchase_data[next_purchase_data['CustomerID'] == customer_id].index
            if len(customer_indices) > 0 and len(values) > 1:
                next_purchase_data.loc[customer_indices[:-1], 'NextPurchaseValue'] = values[1:]
    
    # Drop rows without a next purchase value
    next_purchase_data = next_purchase_data.dropna(subset=['NextPurchaseValue'])
    
    # Select features and target
    features = [
        'DaysSincePrevPurchase', 'PrevQuantity', 'PrevTotalPrice', 'PrevUniqueProducts',
        'PurchaseCount', 'DayOfWeek', 'Month', 'IsWeekend'
    ]
    
    X = next_purchase_data[features]
    y = next_purchase_data['NextPurchaseValue']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Train a Gradient Boosting Regressor
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Purchase Value Model MAE: ${mae:.2f}")
    
    # Save the model
    with open("models/purchase_value_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    return model

if __name__ == "__main__":
    # Load cleaned retail data
    print("Loading cleaned retail data...")
    try:
        df_clean = pd.read_csv("data/online_retail_clean.csv")
        df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    except FileNotFoundError:
        print("Cleaned data not found. Please run data_preprocessing.py first.")
        exit(1)
    
    # Prepare data for purchase prediction
    purchase_data = prepare_purchase_prediction_data(df_clean)
    
    # Train purchase probability model
    prob_model, feature_importances = train_purchase_probability_model(purchase_data)
    
    # Train purchase value model
    value_model = train_purchase_value_model(purchase_data)
    
    print("Purchase prediction models trained successfully!")