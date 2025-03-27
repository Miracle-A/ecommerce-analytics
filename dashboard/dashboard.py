import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

# Load the models and data
@st.cache_data
def load_data():
    # Load customer data
    clustered_customers = pd.read_csv("data/clustered_customers.csv")
    clustered_customers.set_index('CustomerID', inplace=True)
    
    # Load RFM results
    rfm_results = pd.read_csv("data/rfm_results.csv")
    rfm_results.set_index('CustomerID', inplace=True)
    
    # Load cluster profiles
    cluster_profiles = pd.read_csv("data/cluster_profiles.csv")
    
    # Load feature importances
    feature_importances = pd.read_csv("data/feature_importances.csv")
    
    return clustered_customers, rfm_results, cluster_profiles, feature_importances

@st.cache_resource
def load_models():
    # Load segmentation model
    with open("models/kmeans_model.pkl", "rb") as f:
        kmeans_model = pickle.load(f)
    
    # Load purchase probability model
    with open("models/purchase_probability_model.pkl", "rb") as f:
        purchase_prob_model = pickle.load(f)
    
    # Load purchase value model
    with open("models/purchase_value_model.pkl", "rb") as f:
        purchase_value_model = pickle.load(f)
    
    return kmeans_model, purchase_prob_model, purchase_value_model

# Main function to run the app
def main():
    st.title("E-Commerce Customer Analytics Dashboard")
    
    # Load data and models
    try:
        clustered_customers, rfm_results, cluster_profiles, feature_importances = load_data()
        kmeans_model, purchase_prob_model, purchase_value_model = load_models()
        data_loaded = True
    except Exception as e:
        st.error(f"Error loading data or models: {str(e)}")
        st.warning("Please make sure you've run all the preprocessing scripts first.")
        data_loaded = False
    
    if not data_loaded:
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    pages = [
        "Overview",
        "Customer Segmentation",
        "Purchase Prediction",
        "Marketing Recommendations"
    ]
    selection = st.sidebar.radio("Go to", pages)
    
    # Overview page
    if selection == "Overview":
        st.header("E-Commerce Analytics Overview")
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Customers", f"{len(clustered_customers):,}")
        
        with col2:
            avg_monetary = clustered_customers['Monetary'].mean()
            st.metric("Avg. Customer Value", f"${avg_monetary:.2f}")
        
        with col3:
            avg_frequency = clustered_customers['Frequency'].mean()
            st.metric("Avg. Purchase Frequency", f"{avg_frequency:.1f}")
        
        # Display RFM segment distribution
        st.subheader("Customer RFM Segments")
        
        rfm_counts = rfm_results['RFM_Segment'].value_counts().reset_index()
        rfm_counts.columns = ['Segment', 'Count']
        rfm_counts['Percentage'] = rfm_counts['Count'] / rfm_counts['Count'].sum() * 100
        
        fig = px.bar(rfm_counts, x='Segment', y='Count', 
                    text=rfm_counts['Percentage'].apply(lambda x: f"{x:.1f}%"),
                    color='Segment', 
                    title="Distribution of RFM Segments")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display feature importances for purchase prediction
        st.subheader("Top Factors Influencing Purchase Probability")
        
        fig = px.bar(feature_importances.head(10), x='Importance', y='Feature', 
                    orientation='h', 
                    title="Feature Importance for Purchase Prediction")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer Segmentation page
    elif selection == "Customer Segmentation":
        st.header("Customer Segmentation Analysis")
        
        # Display cluster profiles
        st.subheader("Customer Segments")
        st.dataframe(cluster_profiles)
        
        # Display cluster comparison
        st.subheader("Segment Comparison")
        
        # Select metrics to compare
        metrics = ['Recency', 'Frequency', 'Monetary', 'AvgPurchaseValue', 
                   'TotalItems', 'UniqueProducts', 'AvgItemsPerPurchase']
        
        selected_metrics = st.multiselect(
            "Select metrics to compare",
            metrics,
            default=['Recency', 'Frequency', 'Monetary']
        )
        
        if selected_metrics:
            # Group by cluster and calculate mean of selected metrics
            cluster_comparison = clustered_customers.groupby('Cluster')[selected_metrics].mean()
            
            # Create radar chart
            categories = selected_metrics
            fig = go.Figure()
            
            for cluster in sorted(clustered_customers['Cluster'].unique()):
                values = cluster_comparison.loc[cluster].values.tolist()
                # Add the first value at the end to close the polygon
                values = values + [values[0]]
                
                # Normalize to range 0-1 for comparing different scales
                min_vals = clustered_customers[selected_metrics].min()
                max_vals = clustered_customers[selected_metrics].max()
                normalized_values = [(val - min_vals[i]) / (max_vals[i] - min_vals[i]) 
                                     for i, val in enumerate(values[:-1])]
                # Add the first value at the end to close the polygon
                normalized_values = normalized_values + [normalized_values[0]]
                
                fig.add_trace(go.Scatterpolar(
                    r=normalized_values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=f'Cluster {cluster}'
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Cluster Comparison (Normalized Values)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # RFM Analysis by Cluster
        st.subheader("RFM Analysis by Cluster")
        
        # Merge rfm_results with clustered_customers
        rfm_cluster = pd.merge(rfm_results[['RFM_Segment']], 
                              clustered_customers[['Cluster']], 
                              left_index=True, right_index=True)
        
        # Calculate percentage of each RFM segment in each cluster
        rfm_cluster_pct = pd.crosstab(rfm_cluster['Cluster'], 
                                     rfm_cluster['RFM_Segment'], 
                                     normalize='index') * 100
        
        # Plot heatmap
        fig = px.imshow(rfm_cluster_pct,
                       labels=dict(x="RFM Segment", y="Cluster", color="Percentage"),
                       x=rfm_cluster_pct.columns,
                       y=rfm_cluster_pct.index,
                       color_continuous_scale="Viridis",
                       title="Distribution of RFM Segments within Each Cluster (%)")
        
        fig.update_layout(coloraxis_colorbar=dict(title="Percentage"))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Purchase Prediction page
    elif selection == "Purchase Prediction":
        st.header("Purchase Prediction Tool")
        
        st.write("""
        This tool predicts the likelihood of a customer making a purchase in the next 30 days
        and estimates the value of that purchase.
        """)
        
        # Customer prediction form
        st.subheader("Enter Customer Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            days_since_purchase = st.slider("Days Since Last Purchase", 1, 365, 30)
            prev_quantity = st.number_input("Previous Purchase Quantity", 1, 1000, 5)
            prev_total_price = st.number_input("Previous Purchase Value ($)", 1.0, 10000.0, 100.0)
            prev_unique_products = st.number_input("Previous Unique Products", 1, 100, 3)
        
        with col2:
            purchase_count = st.number_input("Total Purchase Count", 1, 100, 5)
            day_of_week = st.selectbox("Day of Week", 
                                      ["Monday", "Tuesday", "Wednesday", "Thursday", 
                                       "Friday", "Saturday", "Sunday"])
            month = st.selectbox("Month", 
                                ["January", "February", "March", "April", "May", "June",
                                 "July", "August", "September", "October", "November", "December"])
            
            # Convert day of week to numeric
            day_mapping = {
                "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                "Friday": 4, "Saturday": 5, "Sunday": 6
            }
            day_of_week_num = day_mapping[day_of_week]
            
            # Convert month to numeric
            month_mapping = {
                "January": 1, "February": 2, "March": 3, "April": 4,
                "May": 5, "June": 6, "July": 7, "August": 8,
                "September": 9, "October": 10, "November": 11, "December": 12
            }
            month_num = month_mapping[month]
            
            # Is weekend flag
            is_weekend = 1 if day_of_week_num >= 5 else 0
        
        # Create feature array for prediction
        features = [
            days_since_purchase, prev_quantity, prev_total_price, prev_unique_products,
            purchase_count, day_of_week_num, month_num, is_weekend
        ]
        
        # Make prediction when button is clicked
        if st.button("Predict Purchase Behavior"):
            # Make predictions
            purchase_prob = purchase_prob_model.predict_proba([features])[0, 1]
            purchase_value = purchase_value_model.predict([features])[0]
            
            # Display predictions
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Purchase Probability", f"{purchase_prob:.1%}")
                
                # Show probability gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = purchase_prob * 100,
                    title = {'text': "Purchase Probability"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "red"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "green"}
                        ]
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric("Estimated Purchase Value", f"${purchase_value:.2f}")
                expected_value = purchase_prob * purchase_value
                st.metric("Expected Value (Prob × Value)", f"${expected_value:.2f}")
                
                # Show purchase value based on probability
                if purchase_prob < 0.3:
                    st.error("Low probability of purchase in the next 30 days.")
                elif purchase_prob < 0.7:
                    st.warning("Moderate probability of purchase in the next 30 days.")
                else:
                    st.success("High probability of purchase in the next 30 days.")
            
            # Recommendation based on prediction
            st.subheader("Marketing Recommendation")
            
            if purchase_prob < 0.3:
                st.markdown("""
                **Re-engagement Strategy**
                - Send a special discount or promotion
                - Highlight new products that match previous purchases
                - Consider a "We miss you" campaign
                """)
            elif purchase_prob < 0.7:
                st.markdown("""
                **Nurturing Strategy**
                - Send personalized product recommendations
                - Offer free shipping on next purchase
                - Share relevant content about product usage
                """)
            else:
                st.markdown("""
                **Upsell Strategy**
                - Recommend premium products or upgrades
                - Offer bundle deals to increase order value
                - Provide early access to new products
                """)
    
    # Marketing Recommendations page
    elif selection == "Marketing Recommendations":
        st.header("Marketing Recommendations by Segment")
        
        # Display cluster-based marketing recommendations
        for index, row in cluster_profiles.iterrows():
            cluster = row['Cluster']
            description = row['Description']
            
            st.subheader(f"Cluster {cluster}: {description}")
            
            # Generate recommendations based on segment characteristics
            if "Recent" in description and "High" in description:
                st.markdown("""
                **High-Value Active Customers** 
                - Focus on retention and loyalty programs
                - Upsell premium products and services
                - Exclusive VIP offers and early access to new products
                - Referral incentives to leverage their network
                """)
            
            elif "Recent" in description and "Low" in description:
                st.markdown("""
                **New or Low-Spend Customers**
                - Educate on product range and benefits
                - Entry-level promotions to encourage exploration
                - Targeted content to build relationship
                - Incentivize second purchase with special offers
                """)
            
            elif "Inactive" in description and "High" in description:
                st.markdown("""
                **Churning High-Value Customers**
                - Reactivation campaigns with personalized incentives
                - Gather feedback on why they stopped purchasing
                - Win-back promotions based on previous purchase history
                - Consider VIP service offerings to rebuild relationship
                """)
            
            elif "Inactive" in description:
                st.markdown("""
                **Lapsed Customers**
                - Reactivation campaigns with strong incentives
                - "We miss you" messaging with personalized recommendations
                - Consider whether reactivation is cost-effective based on past value
                - Final attempt offers before suppressing from regular marketing
                """)
            
            else:
                st.markdown("""
                **Moderate Value/Engagement Customers**
                - Targeted cross-sell based on purchase history
                - Incentivize increased purchase frequency
                - Educational content about product benefits
                - Surveys to better understand needs and preferences
                """)
            
            # Add a visual divider
            st.markdown("---")
        
        # Campaign ROI Calculator
        st.subheader("Campaign ROI Calculator")
        
        st.write("""
        Estimate the return on investment for targeted marketing campaigns
        based on customer segments.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_cluster = st.selectbox(
                "Select Customer Segment",
                sorted(cluster_profiles['Cluster'].unique())
            )
            
            campaign_cost = st.number_input(
                "Campaign Cost ($)",
                min_value=100.0,
                max_value=100000.0,
                value=5000.0,
                step=500.0
            )
            
            conversion_rate = st.slider(
                "Expected Conversion Rate (%)",
                min_value=1.0,
                max_value=30.0,
                value=5.0,
                step=0.5
            ) / 100
        
        with col2:
            # Get segment information
            segment_info = cluster_profiles[cluster_profiles['Cluster'] == selected_cluster].iloc[0]
            segment_size = segment_info['Size']
            
            # Get average purchase value for the segment
            avg_purchase = clustered_customers[clustered_customers['Cluster'] == selected_cluster]['AvgPurchaseValue'].mean()
            
            # Calculate expected results
            customers_reached = st.slider(
                "Customers Reached",
                min_value=int(segment_size * 0.1),
                max_value=segment_size,
                value=int(segment_size * 0.5),
                step=max(1, int(segment_size * 0.05))
            )
            
            expected_conversions = int(customers_reached * conversion_rate)
            expected_revenue = expected_conversions * avg_purchase
            expected_profit = expected_revenue - campaign_cost
            roi = (expected_profit / campaign_cost) * 100 if campaign_cost > 0 else 0
            
            st.metric("Expected Conversions", f"{expected_conversions:,}")
            st.metric("Expected Revenue", f"${expected_revenue:,.2f}")
            st.metric("Expected Profit", f"${expected_profit:,.2f}")
            st.metric("ROI", f"{roi:.1f}%", 
                     delta="Positive" if roi > 0 else "Negative")
        
        # Display ROI visualization
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Cost', 'Revenue', 'Profit'],
            y=[campaign_cost, expected_revenue, expected_profit],
            marker_color=['red', 'green', 'blue']
        ))
        
        fig.update_layout(
            title="Campaign Financial Projection",
            xaxis_title="Metric",
            yaxis_title="Amount ($)",
            yaxis=dict(tickprefix="$")
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Run the app
if __name__ == "__main__":
    main()