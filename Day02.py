import pandas as pd


# Create DataFrame
d = {"column1": 1, "column2": "string"}
df = pd.DataFrame.from_dict(d, orient="index")

pd.DataFrame()

df.head()

# x= "90"
#
# y= 100
# z= x+y

# int("9.0")

# try except example

try:
    average = sum(a_list)/len(a_list)
except (ZerodivsionEroor,ValueError):
    print(a_list)
    average = none