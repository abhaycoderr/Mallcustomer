
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns


try:
    df = pd.read_csv('Mall_Customers.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'Mall_Customers.csv' not found. Please ensure the file is in the same directory.")
    exit()


print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nDataset Info:")
df.info()



X = df.iloc[:, [3, 4]].values


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)




print("\nCalculating WCSS for the Elbow Method...")
wcss = []
# Calculating WCSS for a range of clusters from 1 to 10
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the WCSS values to find the 'elbow'
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('The Elbow Method for Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()





print("\nApplying K-Means with K=5 and visualizing the clusters...")
# Creating and fitting the K-Means model with 5 clusters
kmeans_final = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
y_kmeans = kmeans_final.fit_predict(X_scaled)



# Adding  the cluster labels to the original DataFrame
df['Cluster'] = y_kmeans
print("\nDataFrame with cluster labels:")
print(df.head())



# Visualizing  the clusters using a scatter plot
plt.figure(figsize=(12, 8))
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=df['Cluster'], palette='viridis', s=100)



# Plotting the centroids
centroids = scaler.inverse_transform(kmeans_final.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')

plt.title('Customer Segments based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()