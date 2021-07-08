# Jian Hu is writing the code here
# ''' Day2 Exercise for Unsupervised Machine Learning '''

# import the methods and packages we need
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# 1. Principal Component Analysis

# a) Read the textile
# read the csv file
FilePath = "./data/Olympics.csv"

# read data with the first column id
df = pd.read_csv(FilePath, index_col="id")

# whether to drop the score
# check the columns data types
df.info()
# store summary statistics in a dataset to better have a look at them
x = df.describe()  # i am really not certain whether it makes senses or not to delte the column from my dataset before
# going to build a model. But seeing the data may not resemble that much of the rest.
# It seems the first features are predictors of the score variable.
print(x)

# delete the last column
df = df.drop("score", 1)

# prepare the statistics individually
df.corr()


# b) Unit variance with standard scaler

# scale the variable
scaler = StandardScaler()
# scaled df
df_scaled = pd.DataFrame(scaler.fit_transform(df))

# to check if we scaled unit variance in our data set, we can use std() function from pandas.DataFrame
df_scaled.std()  # unit variance is 1.015505 for all features


# c) PCA MODEL

# plain vanilla PCA model
pca = PCA(random_state=42)
pca.fit(df_scaled)

# pca.fit(df_scaled)_components_, Store the components in a new variable
Loadings_pca = pd.DataFrame(pca.components_, columns=list(df.columns))
# none of the variables load most prominently on the first component
# none of the variables load most prominently on the second component
# variable haut load most prominently on the third component
# there seems to be in total 10 components in the dataset, however, we have not checked how many of these components
# make sense and how many of these would be enough to describe our data. Of course, this depends a little bit on the
# knowledge on the subject matter. Because we do not have a limit of meaningful components (have not set a limit ),
# some variables load more than one component and for some components, there is no predominant variable
# although the explained variance is higher than the other components(e.g. first and second component).

# now we can check explain variance, or get the ratio
exp_var = pd. DataFrame(pca.explained_variance_ratio_, columns=["Explained_variance"])
exp_var.index.name = "Principal Component"

# now get a new column
exp_var["Cum.Explained"] = exp_var["Explained_variance"].cumsum()
print(pca.explained_variance_ratio_)

# we still need at least 7 component to explain at least 90% of variance


# 2. Clustering


# a) load iris data set

# iris data
iris = load_iris()
X = iris["data"]


# b) scaling the data
scaler = StandardScaler()

scaler.fit(X)
df_scaled = scaler.transform(X)
#  to check the variance if it has been scaled properly
df_scaled.std(axis=0)  # to check the variance


# c)cluster the scales variables

# 1c). Kmeans Model
kmeans = KMeans(n_clusters=3, random_state=3).fit(df_scaled)

# check the labels
print(kmeans.labels_)

# store it in the DataFrame
df = pd.DataFrame()
df["kmeans"] = kmeans.labels_
df.head()
df["kmeans"].value_counts()

# 2c). Agglomerative Model

# get the model instance
agg = AgglomerativeClustering(n_clusters=3).fit(df_scaled)

# store it in the DataFrame, add the labels to the data set we created
df["agg"] = agg.labels_
df.head()
df["agg"].value_counts()

# 3c). DBSCAN model
dbscan = DBSCAN(min_samples=2, metric="euclidean", eps=1).fit(df_scaled)

# store it in the DataFrame
df["dbscan"] = dbscan.labels_  # add the labels to the data set
df.head()
df["dbscan"].value_counts()   # to see how many observations are in the noise cluster

# we have to treat the noise differently as these observations are not assigned to any clusters existing in our
# data set. They are outside of the cluster borders and they themselves do not compose a standalone cluster.


# d) compute the silhoutte scores
print(silhouette_score(df_scaled, kmeans.labels_))
print(silhouette_score(df_scaled, agg.labels_))
print(silhouette_score(df_scaled, dbscan.labels_))

# DBSCAN has the highest Silhouette score with 0.504645610832545, bc we have only 2 clusters in DBSCAN,
# and the silhoutte scores of Kmeans is better than in DBSCAN, DBSCAN just indicates that there should be 2
# clusters than 3 as in reality.


# e) add variables "sepal width" and "petal length to the DataFrame that contains  the cluster assignments
# get the dataset from the original source
X = pd.DataFrame(iris["data"], columns=iris["feature_names"])
# add the sepal width and petal length to the DataFrame we created
df[["sepal_width", "petal_length"]] = X[["sepal width (cm)", "petal length (cm)"]]


# f) rename possible noise assignments to "Noises"
df['dbscan'] = df["dbscan"].replace([-1], ['Noise'])

# melt the data set  (cluster variables)
df_melted = pd.melt(df, id_vars=["sepal_width", "petal_length"], var_name="Cluster_algorithm")
# rename the column
df_melted = df_melted.rename(columns={"value": "Cluster_assignment"})


# g) plot a three-fold scatter plot using sepal width as X-variable, petal length as Y-variable
g = sns.FacetGrid(df_melted, col="Cluster_algorithm")
g = g.map_dataframe(sns.scatterplot, x="sepal_width", y="petal_length", hue="Cluster_assignment")
g = g.add_legend()

# save the plot as ./output/cluster_petal.pdf
g.savefig("./output/cluster_petal.pdf")

# Noise assignment does not make sense intuitively, if someone does not know this dataset, and does not
# know what we have done. He/she might wonder why there is the observation called noise. However, for someone
# who knows clustering and these different approaches, it makes sense because the person checking these plots can
# immediately understand that there are Noise Variables that cannot be assigned to any cluster and when we
# plot the results, we can see this observation.
