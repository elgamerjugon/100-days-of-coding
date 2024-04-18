%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
sns.set()

# Esta función genera un array de 300, 2 y los guarda en la variable points, también guarda los index de los clusters de predicción
points, cluster_indexes = make_blobs(n_samples = 300, centers = 4, cluster_std = 0.8, random_state = 0)

x = points[:, 0]
y = points[:, 1]

# La s significa marker size, c = color, marker = "o" el tipo de marcador, cmap = el color de los puntos, alpha = transparencia
plt.scatter(x,y , s= 50, alpha = 0.7)

# Entrenamiento del modelo kmeans con 4 clústers
kmeans = KMeans(n_clusters = 4, random_state = 0)
kmeans.fit(points)
predicted_cluster_indexes = kmeans.predict(points)

plt.scatter(x, y, c= predicted_cluster_indexes, s = 50, alpha = 0.7, cmap = "viridis")

# Centroides que el modelo generó
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c = "red", s = 100)

# Elbow plot, inertias (sum of sqared distances of the data points to the closes cluster center)
inertias = []

for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, random_state = 0)
    kmeans.fit(points)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 10), inertias)
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")

# Customers data
customers = pd.read_csv("data/customers.csv")
customers.head()

points = customers.iloc[:, 3:5].values
x = points[:, 0]
y = points[:, 1]
plt.scatter(x, y, s = 50, alpha = 0.7)
plt.xlabel("Annual Income (k$))")
plt.ylabel("Spending score")

kmeans = KMeans(n_clusters = 5, random_state = 0)
kmeans.fit(points)
predicted_cluster_indexes = kmeans.predict(points)

plt.scatter(x, y, c = predicted_cluster_indexes, s = 50, alpha = 0.7, cmap = "viridis")
plt.xlabel("Annual Income (K$)")
plt.ylabel("Sepdning Score")

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c = "red", s = 100)

df = customers.copy()
df["Cluster"] = kmeans.predict(points)
df.head()

# Get the clsuter index for a customer with a high income and low spending score
cluster = kmeans.predict(np.array([[120, 20]]))[0]

# Filter the dataframe to include only customers in that cluster
clustered_df = df[df["Cluster"] == cluster]

# Show the customer IDs
clustered_df["CustomerID"].values

# Multivariable clustering

df = customers.copy()
encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])
df.head()

points = df.iloc[:, 1:5].values
inertias = []

for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, random_state = 0)
    kmeans.fit(points)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 10), inertias)
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")

kmeans = KMeans(n_clusters = 5, random_state = 0)
kmeans.fit(points)

df["Cluster"] = kmeans.predict(points)
df.head()

results = pd.DataFrame(columns = [
    "Cluster", "Average Age", "Average Income", "Average Spending Index",
    "Number of females", "Number of males"])

for i, center in enumerate(kmeans.cluster_centers_):
    age = center[1]
    income = center[2]
    spend = center[3]
    gdf = df[df["Cluster"] == i]
    females = df[df["Gender"] == 0].shape[0]
    males = df[df["Gender"] == 1].shape[0]
    results.loc[i] = ([i, age, income, spend, females, males])

results.head()