import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.datasets import load_iris
from scipy.stats import skew,kurtosis,zscore
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans,AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram,linkage

iris=load_iris()
irisdb=pd.DataFrame(iris.data, columns=iris.feature_names)
print(irisdb.info())
print(irisdb.describe())

print("Iris null check"+str(irisdb.isnull().sum()))
print("Iris duplicate check"+str(irisdb.duplicated().sum()))
irisdb=irisdb.drop_duplicates()
print("Iris duplicate check"+str(irisdb.duplicated().sum()))
print(irisdb.skew())

scaler=StandardScaler()
scaled_feature= scaler.fit_transform(irisdb)



##################################################################### KMEAN ALGORITHAM
modelkmean= KMeans(n_clusters=3)
irisdb['Cluster']=modelkmean.fit_predict(scaled_feature)
wcss=[]
k_values= range(1,11)
for k in k_values: 
   kmeans = KMeans(n_clusters=k)
   kmeans.fit(scaled_feature)
   wcss.append(kmeans.inertia_)
print(wcss)
print(irisdb)

plt.plot(k_values,wcss)
# plt.show()

############################################################## Hierarchical
hc= AgglomerativeClustering(n_clusters=3,metric='euclidean',linkage='ward')
irisdb['Cluster']= hc.fit_predict(scaled_feature)
print(irisdb)
sns.scatterplot(x=irisdb['sepal length (cm)'], y=irisdb['sepal width (cm)'], hue=irisdb['Cluster'], palette='deep')

plt.title("Agglomerative Clustering Results")
# plt.show()
z=linkage(scaled_feature,method='ward')
lab=irisdb['Cluster'].tolist()
dendrogram(z, labels=lab,leaf_rotation=90)
plt.show()
# final=irisdb['target'].tolist()

















