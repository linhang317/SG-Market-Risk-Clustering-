#!/usr/bin/env python
# coding: utf-8

# In[286]:


from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[383]:


df_data1_X = pd.read_excel("/Users/linhang/Desktop/data/v1.1/data_used_v1.1.xlsx", index_col = 0, usecols = np.arange(0,19))


# In[466]:


df_data1_all = pd.read_excel("/Users/linhang/Desktop/data/v1.1/data_used_v1.1.xlsx", index_col = 0)


# In[388]:


df_data1_y = pd.read_excel("/Users/linhang/Desktop/data/v1.1/data_used_v1.1.xlsx", index_col = 0, usecols = [0,19,24])


# In[390]:


np_data1_X = df_data1_X.to_numpy()


# In[391]:


X_normalized = StandardScaler().fit_transform(df_data1_X)


# In[392]:


#A reduction in dimensions is performed by using t-SNE, which is an dimensional reduction algorithm that
#gives much more weight to preserving information about distances between points that are neighbors and
#that transforms data into 2-dimensional points for clustering visualization purposes.
#The term "perplexity" refers to a guess of the number of close neighbor each data point has.


# In[434]:


tsne_p5_iter1000 = TSNE(random_state = 1, perplexity =5, n_iter = 1000)
tsne_p15_iter1000 = TSNE(random_state = 2, perplexity =15, n_iter = 1000)
tsne_p30_iter1000 = TSNE(random_state = 3, perplexity =30, n_iter = 1000)
tsne_p50_iter1000 = TSNE(random_state = 4, perplexity =50, n_iter = 1000)

tsne_p5_iter3000 = TSNE(random_state = 5, perplexity =5, n_iter = 3000)
tsne_p15_iter3000 = TSNE(random_state = 6, perplexity =15, n_iter = 3000)
tsne_p30_iter3000 = TSNE(random_state = 7, perplexity =30, n_iter = 3000)
tsne_p50_iter3000 = TSNE(random_state = 8, perplexity =50, n_iter = 3000)

tsne_p5_iter250 = TSNE(random_state = 9, perplexity =5, n_iter = 250)
tsne_p15_iter250 = TSNE(random_state = 10, perplexity =15, n_iter = 250)
tsne_p30_iter250 = TSNE(random_state = 11, perplexity =30, n_iter = 250)
tsne_p50_iter250 = TSNE(random_state = 12, perplexity =50, n_iter = 250)


# In[435]:


X_tsne_p5_iter1000 = tsne_p5_iter1000.fit_transform(X_normalized)
X_tsne_p15_iter1000 = tsne_p15_iter1000.fit_transform(X_normalized)
X_tsne_p30_iter1000 = tsne_p30_iter1000.fit_transform(X_normalized)
X_tsne_p50_iter1000 = tsne_p50_iter1000.fit_transform(X_normalized)

X_tsne_p5_iter3000 = tsne_p5_iter3000.fit_transform(X_normalized)
X_tsne_p15_iter3000 = tsne_p15_iter3000.fit_transform(X_normalized)
X_tsne_p30_iter3000 = tsne_p30_iter3000.fit_transform(X_normalized)
X_tsne_p50_iter3000 = tsne_p50_iter3000.fit_transform(X_normalized)

X_tsne_p5_iter250 = tsne_p5_iter250.fit_transform(X_normalized)
X_tsne_p15_iter250 = tsne_p15_iter250.fit_transform(X_normalized)
X_tsne_p30_iter250 = tsne_p30_iter250.fit_transform(X_normalized)
X_tsne_p50_iter250 = tsne_p50_iter250.fit_transform(X_normalized)


# In[436]:


plt.scatter(X_tsne_p5_iter1000[:,0],X_tsne_p5_iter1000[:,1],c = df_data1_y[df_data1_y.columns[0]])
plt.title("X_tsne_p5_iter1000")


# In[437]:


plt.scatter(X_tsne_p15_iter1000[:,0],X_tsne_p15_iter1000[:,1],c = df_data1_y[df_data1_y.columns[0]])
plt.title("X_tsne_p15_iter1000")


# In[438]:


plt.scatter(X_tsne_p30_iter1000[:,0],X_tsne_p30_iter1000[:,1],c = df_data1_y[df_data1_y.columns[0]])
plt.title("X_tsne_p30_iter1000")


# In[439]:


plt.scatter(X_tsne_p50_iter1000[:,0],X_tsne_p50_iter1000[:,1],c = df_data1_y[df_data1_y.columns[0]])
plt.title("X_tsne_p50_iter1000")


# In[440]:


plt.scatter(X_tsne_p5_iter3000[:,0],X_tsne_p5_iter3000[:,1],c = df_data1_y[df_data1_y.columns[0]])
plt.title("X_tsne_p5_iter3000")


# In[441]:


plt.scatter(X_tsne_p15_iter3000[:,0],X_tsne_p15_iter3000[:,1],c = df_data1_y[df_data1_y.columns[0]])
plt.title("X_tsne_p15_iter3000")


# In[442]:


plt.scatter(X_tsne_p30_iter3000[:,0],X_tsne_p30_iter3000[:,1],c = df_data1_y[df_data1_y.columns[0]])
plt.title("X_tsne_p30_iter3000")


# In[443]:


plt.scatter(X_tsne_p50_iter3000[:,0],X_tsne_p50_iter3000[:,1],c = df_data1_y[df_data1_y.columns[0]])
plt.title("X_tsne_p50_iter3000")


# In[444]:


plt.scatter(X_tsne_p5_iter250[:,0],X_tsne_p5_iter250[:,1],c = df_data1_y[df_data1_y.columns[0]])
plt.title("X_tsne_p5_iter250")


# In[445]:


plt.scatter(X_tsne_p15_iter250[:,0],X_tsne_p15_iter250[:,1],c = df_data1_y[df_data1_y.columns[0]])
plt.title("X_tsne_p15_iter250")


# In[446]:


plt.scatter(X_tsne_p30_iter250[:,0],X_tsne_p30_iter250[:,1],c = df_data1_y[df_data1_y.columns[0]])
plt.title("X_tsne_p30_iter250")


# In[447]:


plt.scatter(X_tsne_p50_iter250[:,0],X_tsne_p50_iter250[:,1],c = df_data1_y[df_data1_y.columns[0]])
plt.title("X_tsne_p50_iter250")


# In[ ]:


#it can be seen that at a perplexity level of 15 and iteration of 1,000, clear clustering is presented, and
#such clustering is related to sectors


# In[448]:


kmeans = pd.Series(np.zeros(15), index = np.arange(1,16))
costs = kmeans.copy()
kmeans[1] = KMeans(n_clusters = 1).fit(X_tsne_p15_iter1000)
kmeans[2] = KMeans(n_clusters = 2).fit(X_tsne_p15_iter1000)
kmeans[3] = KMeans(n_clusters = 3).fit(X_tsne_p15_iter1000)
kmeans[4] = KMeans(n_clusters = 4).fit(X_tsne_p15_iter1000)
kmeans[5] = KMeans(n_clusters = 5).fit(X_tsne_p15_iter1000)
kmeans[6] = KMeans(n_clusters = 6).fit(X_tsne_p15_iter1000)
kmeans[7] = KMeans(n_clusters = 7).fit(X_tsne_p15_iter1000)
kmeans[8] = KMeans(n_clusters = 8).fit(X_tsne_p15_iter1000)
kmeans[9] = KMeans(n_clusters = 9).fit(X_tsne_p15_iter1000)
kmeans[10] = KMeans(n_clusters = 10).fit(X_tsne_p15_iter1000)
kmeans[11] = KMeans(n_clusters = 11).fit(X_tsne_p15_iter1000)
kmeans[12] = KMeans(n_clusters = 12).fit(X_tsne_p15_iter1000)
kmeans[13] = KMeans(n_clusters = 13).fit(X_tsne_p15_iter1000)
kmeans[14] = KMeans(n_clusters = 12).fit(X_tsne_p15_iter1000)
kmeans[15] = KMeans(n_clusters = 13).fit(X_tsne_p15_iter1000)


# In[449]:


count = 1
for x in costs:
    costs[count] = kmeans[count].inertia_
    count = count + 1


# In[450]:


costs


# In[458]:


plt.plot(costs.index, costs)
plt.title("Costs vs Number of clusters")


# In[ ]:


#6 and 12 are used as cluster numbers because at 6 number of clusters, cost of the clustering tends to converge,
#and at 12 numbers of clusters or more, the clustering is stuck to local optimum


# In[459]:


df_data1_X["12-Cluster Label"] = kmeans[12].labels_


# In[460]:


df_data1_X["6-Cluster Label"] = kmeans[6].labels_


# In[469]:


df_data1_all[df_data1_all.columns[-5]]


# In[470]:


df_data1_X["Sector"] = df_data1_all[df_data1_all.columns[-2]]
df_data1_X["Industry"] = df_data1_all[df_data1_all.columns[-1]]
df_data1_X["Name"] = df_data1_all[df_data1_all.columns[-5]]
df_data1_X["Chinese Name"] = df_data1_all[df_data1_all.columns[-4]]


# In[462]:


plt.scatter(X_tsne_p15_iter1000[:,0],X_tsne_p15_iter1000[:,1],c = df_data1_X["6-Cluster Label"])
plt.title("X_tsne_p15_iter1000 - 6 Clusters")


# In[463]:


plt.scatter(X_tsne_p15_iter1000[:,0],X_tsne_p15_iter1000[:,1],c = df_data1_X["12-Cluster Label"])
plt.title("X_tsne_p15_iter1000 - 12 Clusters")


# In[464]:


plt.scatter(X_tsne_p15_iter1000[:,0],X_tsne_p15_iter1000[:,1],c = df_data1_y[df_data1_y.columns[0]])
plt.title("X_tsne_p15_iter1000")


# In[472]:


df_data1_X.to_excel("/Users/linhang/Desktop/data/v1.1/data_clustered_v1.1.xlsx")


# In[ ]:




