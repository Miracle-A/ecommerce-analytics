# E-Commerce Customer Analytics and Purchase Prediction

## Project Overview
This project develops a comprehensive analytics solution for an e-commerce business that combines customer segmentation with purchase prediction to drive personalized marketing strategies and increase revenue.

## Features
- RFM Analysis (Recency, Frequency, Monetary value)
- K-means clustering for customer segmentation
- Machine learning models for purchase prediction
- Interactive dashboard with Streamlit
- Marketing campaign ROI calculator

## Dashboard Sections
- **Overview**: Key metrics and customer segment distribution
- **Customer Segmentation**: Detailed analysis of customer segments and behaviors
- **Purchase Prediction**: Tool to predict future customer purchases
- **Marketing Recommendations**: Targeted strategies by segment with ROI projections

## Technologies Used
- Python, Pandas, NumPy, Scikit-learn
- Streamlit for interactive dashboard
- Plotly for data visualization
- Machine learning for predictive modeling

## How to Run
1. Clone this repository
2. Install requirements: \pip install -r requirements.txt\
3. Run the data pipeline:
   - \python src/data_acquisition.py\
   - \python src/data_preprocessing.py\
   - \python src/customer_segmentation.py\
   - \python src/purchase_prediction.py\
4. Launch the dashboard: \streamlit run dashboard/dashboard.py\

## Dataset
This project uses the UCI Online Retail dataset, which contains transactions from a UK-based online retailer over a one-year period.

