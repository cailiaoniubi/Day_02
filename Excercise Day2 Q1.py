# import packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#read the csv file
FilePath = "C:/Users/pc1/Desktop/Machine learning/Excercise/TUM_2021_Exercises/data/Olympics.csv"


#read data with the first column id
df = pd.read_csv(FilePath, index_col="id")

# df= pd.read_csv(FilePath)
# df= df.set_index(“id”)


#whether to drop the score
# store sumarry statistics in a dataset to better have a look at them
df.info()
x = df.describe()
print(x)
df = df.drop("score", 1)

# prepare the statistics individually
df.corr()


# Unit variance with min and max
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df)
df_scaled = scaler.fit_transform(df)
print(df_scaled.max())
print(df_scaled.min())


# plain vanilla PCA model
from sklearn.decomposition import PCA
pca = PCA(random_state=42)
pca.fit(df_scaled)
# pca.fit(df_scaled)_components_
pca.components_
loadings_pca = pd.DataFrame(pca.components_)
# on the first components, 0
# on the 2nd component, 6
# on the thrid component, 9


# now we can check explain variance
exp_var = pd. DataFrame(pca.explained_variance_ratio_, columns=["Explained_variance"])
exp_var.index.name ="Principal Component"
exp_var["Cum.Explained"] = exp_var["Explained_variance"].cumsum()
pca.explained_variance_ratio_

# we still need at least 5 component to explain at least 90% of variance


mean =
print(mean)