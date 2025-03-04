# Customer Segmentation with K-Means Clustering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)

## Overview

This project demonstrates how to segment retail customers based on their shopping behavior using a K-means clustering algorithm. We analyze a dataset containing **Age**, **Gender**, **Annual Income**, and **Spending Score** to identify distinct customer segments. The insights derived can help in developing tailored, data-driven marketing strategies.

## Dataset

The dataset used is `Mall_Customers.csv` obtained from **Kaggle** and includes:
- **CustomerID**: Unique identifier for each customer.
- **Gender**: Customer gender.
- **Age**: Customer's age.
- **Annual Income (k$)**: Annual income in thousands of dollars.
- **Spending Score (1-100)**: A score assigned based on customer behavior and spending habits.

[Elbow Method](elbow-method.png)

[Silhouette score for optimal K](Silhouette-score.png)
## Data Preprocessing
Remove irrelevant columns, encode categorical variables, and normalize features.
Optimal Clusters: Determine the ideal number of clusters using the Elbow Method and Silhouette Score.
Clustering & Visualization: Apply K-means clustering and visualize the results using PCA.
Insights: Analyze cluster centroids to derive actionable marketing strategies.

## Project Structure
Customer-Segmentation/
├── customer_segmentation.ipynb   # Notebook with full analysis and code
├── Mall_Customers.csv            # Dataset file
├── README.md                     # This file
├── requirements.txt              # List of dependencies
└── LICENSE                       # Project license (MIT)

## Results & Insights
Cluster Analysis: The optimal number of clusters is determined using both the Elbow Method and Silhouette Score.
Visualization: PCA is applied to reduce dimensions for a clear scatter plot visualization of customer segments.
Cluster Characteristics: Analysis of cluster centroids provides insight into customer demographics and spending behaviors.

## Marketing Strategies
Based on the cluster profiles, possible strategies include:
  <ul>
    <li>High Spending / Low Income: Target with promotions, discounts, and loyalty programs.</li>
<li>High Income / Low Spending: Offer exclusive premium services and personalized recommendations.</li>
<li>High Spending / High Income: Focus on premium experiences and reward programs.</li>
<li>Low Spending / Older Customers: Enhance engagement through targeted communications and community events.</li>
<li>Balanced Segments: Use cross-selling and bundling strategies tailored to moderate spenders.</li>
  </ul>

## License
This project is licensed under the MIT License.

## Contact
For questions or suggestions, feel free to contact me at [murithigad@gmail.com].
