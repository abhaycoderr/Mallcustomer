# Mall Customer Segmentation using Machine Learning

## Overview
This project focuses on customer segmentation using machine learning techniques to group mall customers based on their demographic and behavioral characteristics.

Customer segmentation is an important task in retail analytics, enabling businesses to understand customer behavior and design targeted marketing strategies. This project demonstrates an end-to-end workflow including data preprocessing, clustering, and visualization.

---

## Problem Statement
Given a dataset of mall customers, the objective is to segment customers into distinct groups based on their spending behavior and income patterns. These segments can help businesses identify high-value customers and optimize marketing strategies.

---

## Dataset Description

The project uses the **Mall Customers Dataset**, which is commonly used for learning clustering techniques.

### Dataset Characteristics:
 - Total Features: 5  
- No missing values  

### Features:
- **CustomerID** – Unique identifier for each customer  
- **Gender** – Male / Female  
- **Age** – Age of the customer  
- **Annual Income (k$)** – Yearly income in thousand dollars  
- **Spending Score (1–100)** – Score assigned based on purchasing behavior  

This dataset contains both demographic and behavioral data, making it suitable for clustering and segmentation tasks.

---

## Methodology

### Data Preprocessing
- Checked for missing values  
- Selected relevant features (Income & Spending Score)  
- Feature scaling for clustering  

### Exploratory Data Analysis
- Distribution of age, income, and spending score  
- Relationship between income and spending  

### Model Implementation
- Applied **K-Means Clustering algorithm**  
- Used **Elbow Method** to determine optimal number of clusters  

K-Means groups similar data points by minimizing the distance between data points and cluster centroids. :contentReference[oaicite:1]{index=1}  

---

## Results and Insights

The model identifies **approximately 5 customer segments**, such as:

- High income, high spending (premium customers)  
- High income, low spending  
- Low income, high spending  
- Average income and spending  
- Low income, low spending  

These clusters help businesses:
- Identify target customers  
- Improve marketing strategies  
- Increase customer retention  

Clustering enables discovery of hidden patterns in data without labeled outputs, making it a powerful unsupervised learning technique. :contentReference[oaicite:2]{index=2}  

---

## Performance Evaluation

Since this is an **unsupervised learning problem**, traditional accuracy metrics are not used.

Evaluation is based on:
- Cluster separation (visual clarity)  
- Elbow method optimization  
- Business interpretability of clusters  

---

## Tech Stack
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

