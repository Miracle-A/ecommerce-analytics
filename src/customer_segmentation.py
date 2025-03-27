# src/customer_segmentation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import os

def perform_rfm_analysis(customer_features):
    """
    Perform RFM (Recency, Frequency, Monetary) analysis with robust handling of duplicate values.
    """
    print("Performing RFM analysis...")
    
    # Create a copy of the dataframe
    rfm = customer_features[['Recency', 'Frequency', 'Monetary']].copy()
    
    # Create RFM quartiles with robust handling of duplicates
    # For Recency - lower is better, so reverse the labels
    try:
        rfm['R_Quartile'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1], duplicates='drop')
    except ValueError:
        # If too many duplicates, use custom logic
        recency_median = rfm['Recency'].median()
        rfm['R_Quartile'] = 1  # Default low value
        rfm.loc[rfm['Recency'] <= recency_median/2, 'R_Quartile'] = 4  # Very recent
        rfm.loc[(rfm['Recency'] > recency_median/2) & (rfm['Recency'] <= recency_median), 'R_Quartile'] = 3  # Recent
        rfm.loc[(rfm['Recency'] > recency_median) & (rfm['Recency'] <= recency_median*2), 'R_Quartile'] = 2  # Less recent
    
    # For Frequency - higher is better
    try:
        rfm['F_Quartile'] = pd.qcut(rfm['Frequency'], 4, labels=[1, 2, 3, 4], duplicates='drop')
    except ValueError:
        # If too many duplicates, use custom logic
        frequency_median = rfm['Frequency'].median()
        rfm['F_Quartile'] = 1  # Default low value
        rfm.loc[rfm['Frequency'] > frequency_median*2, 'F_Quartile'] = 4  # Very frequent
        rfm.loc[(rfm['Frequency'] > frequency_median) & (rfm['Frequency'] <= frequency_median*2), 'F_Quartile'] = 3  # Frequent
        rfm.loc[(rfm['Frequency'] > 1) & (rfm['Frequency'] <= frequency_median), 'F_Quartile'] = 2  # Less frequent
    
    # For Monetary - higher is better
    try:
        rfm['M_Quartile'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4], duplicates='drop')
    except ValueError:
        # If too many duplicates, use custom logic
        monetary_median = rfm['Monetary'].median()
        rfm['M_Quartile'] = 1  # Default low value
        rfm.loc[rfm['Monetary'] > monetary_median*2, 'M_Quartile'] = 4  # High value
        rfm.loc[(rfm['Monetary'] > monetary_median) & (rfm['Monetary'] <= monetary_median*2), 'M_Quartile'] = 3  # Good value
        rfm.loc[(rfm['Monetary'] > 0) & (rfm['Monetary'] <= monetary_median), 'M_Quartile'] = 2  # Low value
    
    # Ensure all quartiles are integer type
    rfm['R_Quartile'] = rfm['R_Quartile'].astype(int)
    rfm['F_Quartile'] = rfm['F_Quartile'].astype(int)
    rfm['M_Quartile'] = rfm['M_Quartile'].astype(int)
    
    # Calculate RFM score
    rfm['RFM_Score'] = rfm['R_Quartile'] + rfm['F_Quartile'] + rfm['M_Quartile']
    
    # Assign RFM segments
    rfm['RFM_Segment'] = 'Unknown'
    rfm.loc[rfm['RFM_Score'] >= 9, 'RFM_Segment'] = 'Champions'
    rfm.loc[(rfm['RFM_Score'] >= 7) & (rfm['RFM_Score'] < 9), 'RFM_Segment'] = 'Loyal Customers'
    rfm.loc[(rfm['RFM_Score'] >= 5) & (rfm['RFM_Score'] < 7), 'RFM_Segment'] = 'Potential Loyalists'
    rfm.loc[(rfm['RFM_Score'] >= 3) & (rfm['RFM_Score'] < 5), 'RFM_Segment'] = 'At Risk'
    rfm.loc[rfm['RFM_Score'] < 3, 'RFM_Segment'] = 'Lost'
    
    return rfm

def perform_kmeans_clustering(customer_features):
    """
    Perform K-means clustering on customer features.
    """
    print("Performing K-means clustering...")
    
    # Select features for clustering
    cluster_features = customer_features[['Recency', 'Frequency', 'Monetary', 
                                         'AvgPurchaseValue', 'TotalItems', 
                                         'UniqueProducts', 'AvgItemsPerPurchase']].copy()
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_features)
    
    # Save the scaler for future use
    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    # Find optimal number of clusters using silhouette score
    silhouette_scores = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        score = silhouette_score(scaled_features, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"K={k}, Silhouette Score={score:.4f}")
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different k Values')
    plt.savefig("data/silhouette_scores.png")
    
    # Choose the optimal k (for this example, let's just pick the one with highest score)
    optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    print(f"Optimal number of clusters: {optimal_k}")
    
    # Train final model with optimal k
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    final_kmeans.fit(scaled_features)
    
    # Save the model
    with open("models/kmeans_model.pkl", "wb") as f:
        pickle.dump(final_kmeans, f)
    
    # Add cluster labels to the customer features
    customer_features['Cluster'] = final_kmeans.labels_
    
    return customer_features, final_kmeans

def analyze_clusters(clustered_customers, kmeans_model):
    """
    Analyze the characteristics of each cluster.
    """
    print("Analyzing clusters...")
    
    # Reset index to make CustomerID a column
    df_with_customerid = clustered_customers.reset_index()
    
    # Calculate cluster statistics
    cluster_stats = df_with_customerid.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'AvgPurchaseValue': 'mean',
        'TotalItems': 'mean',
        'UniqueProducts': 'mean',
        'AvgItemsPerPurchase': 'mean',
        'CustomerID': 'count'
    }).rename(columns={'CustomerID': 'Count'})
    
    cluster_stats['Percentage'] = (cluster_stats['Count'] / cluster_stats['Count'].sum()) * 100
    
    # Save cluster statistics
    cluster_stats.to_csv("data/cluster_statistics.csv")
    
    # Create cluster profiles based on their characteristics
    profiles = []
    for cluster in sorted(clustered_customers['Cluster'].unique()):
        stats = cluster_stats.loc[cluster]
        
        # Determine buying frequency category
        if stats['Frequency'] > cluster_stats['Frequency'].quantile(0.75):
            frequency = "High"
        elif stats['Frequency'] < cluster_stats['Frequency'].quantile(0.25):
            frequency = "Low"
        else:
            frequency = "Medium"
        
        # Determine spending category
        if stats['Monetary'] > cluster_stats['Monetary'].quantile(0.75):
            spending = "High"
        elif stats['Monetary'] < cluster_stats['Monetary'].quantile(0.25):
            spending = "Low"
        else:
            spending = "Medium"
        
        # Determine recency category (lower is better)
        if stats['Recency'] < cluster_stats['Recency'].quantile(0.25):
            recency = "Recent"
        elif stats['Recency'] > cluster_stats['Recency'].quantile(0.75):
            recency = "Inactive"
        else:
            recency = "Moderate"
        
        # Create profile
        profile = {
            'Cluster': cluster,
            'Size': int(stats['Count']),
            'Percentage': f"{stats['Percentage']:.1f}%",
            'Frequency': frequency,
            'Spending': spending,
            'Recency': recency,
            'AvgPurchaseValue': f"${stats['AvgPurchaseValue']:.2f}",
            'Description': f"{recency} customers with {frequency} purchase frequency and {spending} spending"
        }
        
        profiles.append(profile)
    
    # Convert to DataFrame and save
    profiles_df = pd.DataFrame(profiles)
    profiles_df.to_csv("data/cluster_profiles.csv", index=False)
    
    return profiles_df

if __name__ == "__main__":
    # Load customer features
    print("Loading customer features...")
    try:
        customer_features = pd.read_csv("data/customer_features.csv")
        customer_features.set_index('CustomerID', inplace=True)
    except FileNotFoundError:
        print("Customer features not found. Please run data_preprocessing.py first.")
        exit(1)
    
    # Perform RFM analysis
    rfm_results = perform_rfm_analysis(customer_features)
    rfm_results.to_csv("data/rfm_results.csv")
    
    # Perform K-means clustering
    clustered_customers, kmeans_model = perform_kmeans_clustering(customer_features)
    clustered_customers.to_csv("data/clustered_customers.csv")
    
    # Analyze clusters
    cluster_profiles = analyze_clusters(clustered_customers, kmeans_model)
    
    print("Customer segmentation completed successfully!")