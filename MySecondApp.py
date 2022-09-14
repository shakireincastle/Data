#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv(r"C:\Users\user\Desktop\AirAsia Sept 5-14-2022\AirAsia Sept 10-14\datasets\mall_customer.csv")
df.head()

features = ['Annual_Income_(k$)', 'Spending_Score']
X = df[features]
plt.scatter(X['Annual_Income_(k$)'], X['Spending_Score']);

kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

step_size = 0.01

x_min, x_max = min(X.iloc[:,0]) - 1, max(X.iloc[:,0]) + 1
y_min, y_max = min(X.iloc[:,1]) - 1, max(X.iloc[:,1]) + 1
x_values, y_values = np.meshgrid(np.arange(x_min,x_max,step_size), np.arange(y_min,y_max,step_size))

predictions = kmeans.predict(np.c_[x_values.ravel(), y_values.ravel()])

predictions = predictions.reshape(x_values.shape)
plt.figure(figsize=(8,6))
plt.imshow(predictions, interpolation='nearest', extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()), 
           cmap=plt.cm.Spectral, aspect='auto', origin='lower')

plt.scatter(X.iloc[:,0],X.iloc[:,1], marker='o', facecolors='grey',edgecolors='w',s=30)
# Plot the centroids of the clusters
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1], marker='o', s=200, linewidths=3, 
           color='k', zorder=10, facecolors='black')

plt.title('Centroids and boundaries calculated using KMeans Clustering', fontsize=16)
plt.show()

