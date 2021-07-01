# import packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#a) load iris dataset
from sklearn.datasets import load_iris

# # iris data
iris = load_iris().data
# iris = load_iris()

####alternative way to do it, converting array into dataframe
# data = load_iris()
# df = pd.DataFrame(data.data, columns=data.feature_names)
# df.head()
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(df)
# df_scaled = scaler.transform(df)
# df_scaled.std(axis=0)
#
######################################
# # get the target names(variable names)
target_vals = ["Sepal_length", "Sepal_width", "Petal_length", "Petal_width"]
iris_dat = iris.data
target_iris = iris.target
#
#
# # now build up the dataset
df = pd.DataFrame(iris_dat, columns=target_vals)
df["Species"] = target_iris
#
#
# # now replace species with their respective values
 df["Species"] = df["Species"].replace(to_replace=[0,1,2],value=["setosa","versicolor", "virginica"])

############################################

# b) scale the data such that each variable has unit variance

#####from sklearn.preprocessing import MinMaxScaler
##### scaler = MinMaxScaler()
# scaler=MinMaxScaler()
# scaler.fit(df)
# print(df_scaled.max())
# print(df_scaled.min())


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(iris)
df_scaled = scaler.transform(iris)
df_scaled.std(axis=0) #to check the variance

# DataFrame method to assert that all variables have unit variance
print(df_scaled.max())
print(df_scaled.min())


#########################################
#c)cluster the scales variables

###1c). Kmeans Model
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3,random_state=3).fit(df_scaled)


#check the labels
kmeans.labels_

#store it in the dataframe
df = pd.DataFrame(df, columns=data.feature_names)
df["kmeans"] = kmeans.labels_
df.head()
df["kmeans"].value_counts()

###2c). Agglomerative Model

from sklearn.cluster import AgglomerativeClustering as AC
agg =  AC(n_clusters=3).fit(df_scaled)

#store it in the dataframe
df["agg"] = agg.labels_
df.head()
df["agg"].value_counts()

###3c). DBSCAN model
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(min_samples=2,metric="euclidean", eps =1).fit(df_scaled)

#store it in the dataframe
df["dbscan"] = dbscan.labels_
df.head()
df["dbscan"].value_counts()


####to create another DataFrame
data=[df["kmeans"], df["agg"], df["dbscan"]]

headers=["kmeans","agg","dbscan"]

newdf=pd.concat(data, axis=1, keys=headers)

####the alternative one to create
####new = pd.DataFrame([df.kmeans, df.agg, df.dbscan]).transpose() ####for some reason, this does not work.


#############################################
#d) compute the silhoutte scores
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
print(silhouette_score(df_scaled,kmeans.labels_))
print(silhouette_score(df_scaled,agg.labels_))
print(silhouette_score(df_scaled,dbscan.labels_))

###DBSCAN has the highest Silhouette score with 0.504645610832545, bc we have only 2 clustes in DBSCAN, and the silhoutte scores of Kmeans is better than in DBSCAN, DBSCAN just indicates that there should be 2
###clusters than 3 as in reality.

###############################################
#e)

























df= pd.DataFrame(iris)
df.keys()
df["kmeans"] = kmeans.labels