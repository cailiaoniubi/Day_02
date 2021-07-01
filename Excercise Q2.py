#let us do the second one

# load the packages
import pandas as pd

# 2.get the dataset
DPATH = "http://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user"

# get the data
df = pd.read_csv(DPATH, sep= "|")

# print the last 10 entries and 25 first entries
df.tail(10)
df.head(25)

# check what each column data type is
df.info()

# user_id is integer
# age is integer
# gender is object(categorical)
# occupation is object(categorical)
# Zip_code is object(categorical)

# let us create an object and add the counts of occupations in it
Oc_Counts = pd.DataFrame(df["occupation"].value_counts())
# Oc_Counts = df["occupation"].value_counts()
# the most common one is student with 196

# to check hwo many different occupations there are in the dataset and what is the most frequent
# Oc_Counts.unique()
# Oc_Counts.max()
# converted series data to dataframe
# Oc_counts= pd.DataFrame(Oc_Counts)

# chanage the column name
Oc_Counts = Oc_Counts.rename(columns={"population":"counts"})
Oc_Count["occupation"]
# this checks which occupation has the most values
Oc_counts.occupation.idxmax()  #the most common one is student
Oc_Counts.index.unique().sum() #check the unique one



# sort the new object by index

# df =df.sort_values(by="Oc_Counts")
# df