# E-Commerce Customer Analytics and Purchase Prediction

## Project Overview
This project develops a comprehensive analytics solution for an e-commerce business that combines customer segmentation with purchase prediction to drive personalized marketing strategies and increase revenue.

## Dashboard Screenshots

### Overview
![image](https://github.com/user-attachments/assets/98cf91c4-5f97-4cf3-a474-f6c53e0b4b30)


### Customer Segmentation
![image](https://github.com/user-attachments/assets/0a72b17c-b652-4396-8ff7-1adf22a6149d)


### Purchase Prediction
![image](https://github.com/user-attachments/assets/18360b10-02ea-4034-ab5c-e96f34ca1275)


### Marketing Recommendations
![image](https://github.com/user-attachments/assets/f878fa66-fc4e-4865-b703-33c6c1c9ee2c)


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

