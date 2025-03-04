import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 1. Load and preprocess the data
df = pd.read_csv(r"C:\Users\user\Desktop\INDOLIKE PROJECTS\CLUSTER\Mall_Customers.csv")
print("Initial Data Sample:")
print(df.head())

# Drop irrelevant columns (e.g., CustomerID)
if 'CustomerID' in df.columns:
    df.drop('CustomerID', axis=1, inplace=True)

# Encode categorical variable 'Gender'
if 'Gender' in df.columns:
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])  # E.g., Female:0, Male:1

# Select relevant features
features = ['Age', 'Gender', 'Annual Income (k$)', 'Spending Score (1-100)']

# Normalize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])
scaled_df = pd.DataFrame(scaled_features, columns=features)

# 2. Determine optimal number of clusters using Elbow Method and Silhouette Score
wcss = []
sil_scores = []
K_range = range(2, 11)  # testing k from 2 to 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_df)
    wcss.append(kmeans.inertia_)
    score = silhouette_score(scaled_df, kmeans.labels_)
    sil_scores.append(score)

# Plot Elbow Method and Silhouette Score side by side
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K_range, wcss, 'bo-', markersize=8)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal k")

plt.subplot(1, 2, 2)
plt.plot(K_range, sil_scores, 'ro-', markersize=8)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Optimal k")
plt.tight_layout()
plt.show()

# Based on the plots, let's assume the optimal number of clusters is 5
optimal_k = 5

# 3. Apply K-means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_df)
scaled_df['Cluster'] = cluster_labels
df['Cluster'] = cluster_labels  # Add cluster labels to original dataframe

# Visualize clusters using PCA for dimensionality reduction
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_df[features])
pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = cluster_labels

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette="Set2", s=100)
plt.title("Customer Segments (PCA Projection)")
plt.show()

# 4. Examine cluster centroids in original scale
centers = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(centers, columns=features)
print("Cluster Centroids (in original scale):")
print(centers_df)

# 5. Generate insights by summarizing each cluster
for i in range(optimal_k):
    print(f"\n--- Cluster {i} Analysis ---")
    cluster_data = df[df['Cluster'] == i]
    print(cluster_data.describe())
